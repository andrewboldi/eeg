"""Iteration 068: Channel-Weighted Loss.

Hypothesis: Weighting loss by per-channel CF correlation focuses optimization
on well-coupled channels (where improvement is possible) rather than dead
channels (where scalp signal doesn't reach in-ear). This should increase
mean Pearson r by concentrating capacity where it matters.

Method:
1. Fit CF baseline, compute per-channel correlation on training data
2. Weight channels via softmax(r_ch / temperature) with temperature=0.5
3. Per-channel CorrMSE loss weighted by these channel importances
4. TinyDeep architecture (H=64, 2 blocks) from iter043

Confidence: 70% — focused loss should help if some channels are truly dead.
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
    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(h), nn.GELU())
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


class TinyDeep(nn.Module):
    """Tiny deep model (H=64, 2 blocks) with Flash Attention via SDPA."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.C_out = C_out
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                  nn.BatchNorm1d(H), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
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


class ChannelWeightedCorrMSE(nn.Module):
    """Per-channel CorrMSE with channel importance weights."""

    def __init__(self, channel_weights, a=0.5):
        super().__init__()
        # channel_weights: (C_out,) tensor of importance weights (sum to 1)
        self.register_buffer("w", channel_weights)
        self.a = a

    def forward(self, p, t):
        # p, t: (B, C, T)
        # Per-channel MSE: (B, C)
        mse_ch = ((p - t) ** 2).mean(dim=-1).mean(dim=0)  # (C,)

        # Per-channel correlation: (B, C)
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r_ch = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
        corr_loss_ch = (1 - r_ch).mean(dim=0)  # (C,)

        # Combined per-channel loss
        loss_ch = self.a * mse_ch + (1 - self.a) * corr_loss_ch  # (C,)

        # Weighted sum across channels
        return (self.w * loss_ch).sum()


def compute_channel_weights(cf_model, train_ds, device, temperature=0.5):
    """Compute per-channel correlation of CF predictions on training data,
    then convert to softmax weights."""
    cf_model.eval()
    with torch.no_grad():
        x = train_ds.scalp.to(device)  # (N, C_scalp, T)
        y = train_ds.inear.to(device)  # (N, C_inear, T)

        # Process in chunks to avoid OOM
        chunk = 512
        all_r = []
        for i in range(0, x.shape[0], chunk):
            p = cf_model(x[i:i + chunk])
            yi = y[i:i + chunk]
            pm = p - p.mean(-1, keepdim=True)
            tm = yi - yi.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
            all_r.append(r)
        # all_r: list of (chunk, C_inear) -> cat -> (N, C_inear)
        r_per_sample = torch.cat(all_r, dim=0)  # (N, C_inear)
        r_mean = r_per_sample.mean(dim=0)  # (C_inear,)

    # Softmax weighting with temperature
    weights = F.softmax(r_mean / temperature, dim=0)  # (C_inear,)

    print(f"Channel CF correlations: {r_mean.cpu().numpy().round(3)}")
    print(f"Channel weights (T={temperature}): {weights.cpu().numpy().round(3)}")
    return weights


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


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit CF baseline
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)

    # Step 2: Compute channel weights from CF performance
    channel_weights = compute_channel_weights(cf, train_ds, device, temperature=0.5)

    # Step 3: Build TinyDeep with CF skip init
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeep params: {n_params:,}")

    # Step 4: Channel-weighted CorrMSE loss
    loss_fn = ChannelWeightedCorrMSE(channel_weights, a=0.5)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            # Mixup augmentation (from iter030 findings)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout (from iter011 findings)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad()
            loss_fn(model(x), y).backward()
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
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")
    return model
