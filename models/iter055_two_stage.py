"""Iteration 055: Two-Stage Input-Split Prediction.

Decomposes the 46-channel input into two specialized pathways:
  Stage 1: Linear map from 19 around-ear channels -> 12 in-ear (direct spatial coupling)
  Stage 2: TinyDeep from 27 scalp channels -> 12 in-ear (longer-range temporal patterns)
  Fusion:  Learned per-channel sigmoid-weighted average of both predictions

Hypothesis: Separating input sources lets each stage specialize. Around-ear channels
are physically closest to in-ear and should capture direct spatial coupling via a simple
linear map. Scalp channels capture longer-range dynamics via a deeper temporal model.
The fusion layer learns which source is more informative per output channel.

This differs from ensemble (iter043) which splits by MODEL type; here we split by INPUT type.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# --- Channel split constants ---
# In broadband_46ch.h5: channels 0-26 are scalp, 27-45 are around-ear
N_SCALP = 27
N_AROUND = 19


class MultiScaleConv(nn.Module):
    """Multi-scale temporal convolution bank."""

    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(h),
                nn.GELU(),
            )
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


class TinyDeep(nn.Module):
    """Compact transformer with multi-scale conv front-end."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(
            nn.Conv1d(H, H, 4, stride=4, bias=False),
            nn.BatchNorm1d(H),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.temporal(x)
        h = self.down(h).transpose(1, 2)
        h = self.transformer(h)
        h = h.transpose(1, 2)
        h = self.up(h)[:, :, :self.T]
        h = self.out_norm(h.transpose(1, 2))
        h = self.out_proj(h).transpose(1, 2)
        return h + skip


class TwoStageModel(nn.Module):
    """Two-stage input-split model with learned per-channel fusion.

    Stage 1: Conv1d(19, 12, 1) -- linear around-ear -> in-ear
    Stage 2: TinyDeep(27, 12)  -- deep scalp -> in-ear
    Fusion:  sigmoid(alpha) * stage1 + (1 - sigmoid(alpha)) * stage2
    """

    def __init__(self, C_total, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        assert C_total == N_SCALP + N_AROUND, (
            f"Expected {N_SCALP + N_AROUND} input channels, got {C_total}"
        )
        # Stage 1: simple linear from around-ear (closest physically)
        self.stage1 = nn.Conv1d(N_AROUND, C_out, kernel_size=1, bias=True)

        # Stage 2: deeper model from scalp channels
        self.stage2 = TinyDeep(
            C_in=N_SCALP, C_out=C_out, T=T, H=H,
            n_blocks=n_blocks, dropout=dropout,
        )

        # Fusion: per-channel learned weight (initialized to 0 -> sigmoid=0.5)
        self.alpha = nn.Parameter(torch.zeros(1, C_out, 1))

    def forward(self, x):
        x_scalp = x[:, :N_SCALP, :]       # (B, 27, T)
        x_around = x[:, N_SCALP:, :]       # (B, 19, T)

        pred_around = self.stage1(x_around)  # (B, 12, T)
        pred_scalp = self.stage2(x_scalp)    # (B, 12, T)

        w = torch.sigmoid(self.alpha)        # (1, 12, 1)
        return w * pred_around + (1 - w) * pred_scalp


class CorrMSELoss(nn.Module):
    """Combined MSE + (1 - Pearson r) loss."""

    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / (
            (pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8
        )
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    """Compute mean Pearson r on a data loader."""
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / (
                (pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8
            )
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    """Build and train two-stage input-split model.

    Args:
        train_ds: Training EEGDataset
        val_ds: Validation EEGDataset
        C_scalp: Total input channels (46 for broadband_46ch)
        C_inear: Number of in-ear output channels (12)
        device: torch device
    """
    T = train_ds.scalp.shape[-1]  # temporal dimension (256 for 2s @ 128 Hz)

    # --- Initialize Stage 1 (around-ear) with closed-form solution ---
    around_train = train_ds.scalp[:, N_SCALP:, :].numpy()
    inear_train = train_ds.inear.numpy()
    cf_around = ClosedFormLinear(C_in=N_AROUND, C_out=C_inear)
    cf_around.fit(around_train, inear_train)
    print(f"CF around-ear baseline fitted (around-ear -> in-ear)")

    # --- Initialize Stage 2 skip connection with closed-form solution ---
    scalp_train = train_ds.scalp[:, :N_SCALP, :].numpy()
    cf_scalp = ClosedFormLinear(C_in=N_SCALP, C_out=C_inear)
    cf_scalp.fit(scalp_train, inear_train)
    print(f"CF scalp baseline fitted (scalp -> in-ear)")

    # --- Build two-stage model ---
    model = TwoStageModel(
        C_total=C_scalp, C_out=C_inear, T=T,
        H=64, n_blocks=2, dropout=0.1,
    ).to(device)

    # Initialize stage1 weights from CF around-ear solution
    with torch.no_grad():
        model.stage1.weight.copy_(cf_around.W.float().unsqueeze(-1))
        if model.stage1.bias is not None:
            model.stage1.bias.zero_()

    # Initialize stage2 skip connection from CF scalp solution
    with torch.no_grad():
        model.stage2.skip.weight.copy_(cf_scalp.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TwoStageModel params: {n_params:,}")

    # --- Training setup ---
    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout (applied to full input, affects both stages)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vr = validate_correlation(model, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 25 == 0:
            w = torch.sigmoid(model.alpha).mean().item()
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f}), "
                  f"around_weight={w:.3f}")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Report final fusion weights
    final_w = torch.sigmoid(model.alpha).detach().cpu().squeeze()
    print(f"Two-stage best val_r: {best_r:.4f}")
    print(f"Fusion weights (around-ear): {final_w.numpy().round(3)}")
    print(f"Mean around-ear weight: {final_w.mean().item():.3f}")

    return model
