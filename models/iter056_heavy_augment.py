"""Iteration 056: Heavy Augmentation Suite on TinyDeep.

Hypothesis: The tiny model overfits on 12 training subjects because the data
is too small. More aggressive augmentation creates more diverse training
samples, which should help cross-subject generalization.

Architecture: Same TinyDeep as iter043 (H=64, 2 blocks) -- proven architecture.

Key difference: Apply the FULL augmentation suite from src/augmentations.py
during training, each with probability 0.5:
  1. mixup(alpha=0.4)
  2. channel_dropout(p=0.15)
  3. temporal_shift(max_shift=10)
  4. gaussian_noise(std=0.05)
  5. amplitude_scale(range=(0.85, 1.15))
  6. frequency_mask(n_masks=1, max_width=5)
  7. cutout(max_len=16)

Training: CorrMSE loss, AdamW, 200 epochs (more epochs since augmentation
slows convergence), early stopping patience 40.

Confidence: 70% -- augmentation suite is comprehensive but risk of too much
distortion even with p=0.5 gating.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.augmentations import (
    amplitude_scale,
    channel_dropout,
    cutout,
    frequency_mask,
    gaussian_noise,
    mixup,
    temporal_shift,
)
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Model architecture (identical to iter043 TinyDeep)
# ---------------------------------------------------------------------------

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
    """Tiny deep model (55K params) with Flash Attention via SDPA."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
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


# ---------------------------------------------------------------------------
# Loss and validation (same as iter043)
# ---------------------------------------------------------------------------

class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Stochastic augmentation: each transform applied with probability p_apply
# ---------------------------------------------------------------------------

def apply_augmentations(x, y, device, p_apply=0.5):
    """Apply the full augmentation suite, each gated by independent coin flip."""

    # 1. Mixup (needs special handling: operates on shuffled pairs)
    if torch.rand(1).item() < p_apply:
        idx = torch.randperm(x.shape[0], device=device)
        x, y, _ = mixup(x, y, x[idx], y[idx], alpha=0.4)

    # 2. Channel dropout
    if torch.rand(1).item() < p_apply:
        x = channel_dropout(x, p=0.15)

    # 3. Temporal shift (shifts both x and y together)
    if torch.rand(1).item() < p_apply:
        x, y = temporal_shift(x, y, max_shift=10)

    # 4. Gaussian noise (input only, low std)
    if torch.rand(1).item() < p_apply:
        x = gaussian_noise(x, std=0.05)

    # 5. Amplitude scale (both x and y)
    if torch.rand(1).item() < p_apply:
        x, y = amplitude_scale(x, y, range=(0.85, 1.15))

    # 6. Frequency mask (input only)
    if torch.rand(1).item() < p_apply:
        x = frequency_mask(x, n_masks=1, max_width=5)

    # 7. Cutout (input only)
    if torch.rand(1).item() < p_apply:
        x = cutout(x, max_len=16)

    return x, y


# ---------------------------------------------------------------------------
# build_and_train
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit CF for skip-connection initialization
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Build TinyDeep with CF skip init
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeep (heavy augment) params: {n_params:,}")

    # Step 3: Training with full augmentation suite
    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    patience = 40  # More patience since augmentation slows convergence
    max_epochs = 200

    for ep in range(1, max_epochs + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)

            # Apply full augmentation suite (each with p=0.5)
            x, y = apply_augmentations(x, y, device, p_apply=0.5)

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
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f}, no_imp={no_imp})")
        if no_imp >= patience:
            print(f"  Early stopping at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")
    return model
