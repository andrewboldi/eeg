"""Iteration 051: Per-channel decoder heads on broadband 46ch data.

Hypothesis: Each in-ear channel has different noise characteristics and
different relationships to the scalp/around-ear input channels. A shared
output projection forces all 12 channels through the same bottleneck.
Separate small decoder heads let each channel specialize its own mapping.

Architecture:
  1. Shared encoder: MultiScaleConv(46, 64) + stride-4 downsample
  2. Shared transformer: 2 TransformerEncoder blocks (H=64, 4 heads)
  3. Shared upsample: ConvTranspose1d back to T=256
  4. Per-channel heads: 12 independent Linear(64,32)->GELU->Linear(32,1)
  5. CF skip: Conv1d(46, 12, 1)
  6. Channel-weighted loss: harder channels get MORE weight

Channel weights derived from iter039 LOSO per-channel analysis:
  - Easy channels (high baseline r) get lower weight
  - Hard channels (low baseline r) get higher weight
  This pushes the model to improve where there is most headroom.

Expected: r > 0.64 (vs 0.638 iter039 shared head)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class MultiScaleConv(nn.Module):
    """Extract temporal features at multiple scales."""

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


class PerChannelHead(nn.Module):
    """Small independent decoder for a single output channel."""

    def __init__(self, H, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(H, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, T, H) -> (B, T, 1)
        return self.net(x)


class PerChannelHeadModel(nn.Module):
    """Shared encoder + per-channel decoder heads."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.C_out = C_out
        self.T = T

        # Shared encoder
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(
            nn.Conv1d(H, H, 4, stride=4, bias=False),
            nn.BatchNorm1d(H),
            nn.GELU(),
        )

        # Shared transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # Shared upsample
        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)

        # Per-channel decoder heads (12 independent small MLPs)
        self.heads = nn.ModuleList([
            PerChannelHead(H, hidden=32) for _ in range(C_out)
        ])

        # CF skip connection
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x):
        # x: (B, C_in, T)
        skip = self.skip(x)  # (B, C_out, T)

        # Shared encoder path
        h = self.temporal(x)          # (B, H, T)
        h = self.down(h)              # (B, H, T/4)
        h = h.transpose(1, 2)         # (B, T/4, H)
        h = self.transformer(h)       # (B, T/4, H)
        h = h.transpose(1, 2)         # (B, H, T/4)
        h = self.up(h)[:, :, :self.T] # (B, H, T)
        h = self.out_norm(h.transpose(1, 2))  # (B, T, H)

        # Per-channel heads
        outs = []
        for head in self.heads:
            outs.append(head(h))  # each: (B, T, 1)
        out = torch.cat(outs, dim=2)  # (B, T, C_out)
        out = out.transpose(1, 2)     # (B, C_out, T)

        return out + skip


class ChannelWeightedCorrMSELoss(nn.Module):
    """Combined MSE + correlation loss with per-channel importance weights.

    Channels with lower baseline r (harder) get MORE weight so the model
    focuses optimization effort where there is headroom for improvement.
    Weights are inversely proportional to approximate baseline difficulty.
    """

    def __init__(self, alpha=0.5, channel_weights=None):
        super().__init__()
        self.alpha = alpha
        # channel_weights: (C_out,) tensor, higher = more important
        if channel_weights is not None:
            self.register_buffer("w", channel_weights)
        else:
            self.w = None

    def forward(self, pred, target):
        # pred, target: (B, C_out, T)

        # Per-channel MSE: (B, C_out)
        mse_per_ch = ((pred - target) ** 2).mean(dim=-1)

        # Per-channel correlation: (B, C_out)
        pm = pred - pred.mean(dim=-1, keepdim=True)
        tm = target - target.mean(dim=-1, keepdim=True)
        cov = (pm * tm).sum(dim=-1)
        r = cov / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
        corr_loss_per_ch = 1.0 - r  # (B, C_out)

        if self.w is not None:
            # Weighted mean across channels
            w = self.w.unsqueeze(0)  # (1, C_out)
            mse = (mse_per_ch * w).sum(dim=-1).mean()
            corr_loss = (corr_loss_per_ch * w).sum(dim=-1).mean()
        else:
            mse = mse_per_ch.mean()
            corr_loss = corr_loss_per_ch.mean()

        return self.alpha * mse + (1 - self.alpha) * corr_loss


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    # Step 1: Fit closed-form baseline for skip init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Build per-channel head model
    model = PerChannelHeadModel(
        C_in=C_scalp, C_out=C_inear, T=256,
        H=64, n_blocks=2, dropout=0.1,
    ).to(device)

    # Init skip with CF weights
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PerChannelHead params: {n_params:,}")

    # Channel difficulty weights (inverse of approximate baseline r per channel).
    # Derived from CF baseline on broadband 46ch LOSO: overall mean r ~ 0.577.
    # Approximate per-channel r from typical Ear-SAAD patterns:
    #   ELA/ERA channels (~0.65-0.70) = easy, ELC/ERT channels (~0.35-0.45) = hard
    # Channels ordered: ELA1,ELA2,ELA3,ELB1,ELB2,ELB3,ERA1,ERA2,ERA3,ERB1,ERB2,ERB3
    # Approximate baseline r for each (from cross-subject CF):
    approx_baseline_r = torch.tensor([
        0.65, 0.60, 0.55,  # ELA1, ELA2, ELA3
        0.50, 0.45, 0.40,  # ELB1, ELB2, ELB3
        0.65, 0.60, 0.55,  # ERA1, ERA2, ERA3
        0.50, 0.45, 0.40,  # ERB1, ERB2, ERB3
    ], dtype=torch.float32)
    # Inverse weighting: harder channels get more weight
    # w_i = 1/r_i, then normalize so sum = C_out (preserves loss scale)
    channel_weights = 1.0 / (approx_baseline_r + 0.1)
    channel_weights = channel_weights / channel_weights.sum() * C_inear

    loss_fn = ChannelWeightedCorrMSELoss(alpha=0.5, channel_weights=channel_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # OneCycleLR for stable training
    steps_per_epoch = len(tl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=200, steps_per_epoch=steps_per_epoch,
    )

    best_r, best_state, no_imp = -1.0, None, 0

    for ep in range(1, 201):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)

            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]

            # Channel dropout (15%)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        vr = validate_correlation(model, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")

        if no_imp >= 40:
            print(f"  Early stopping at epoch {ep}")
            break

    print(f"PerChannelHead best val_r: {best_r:.4f}")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model
