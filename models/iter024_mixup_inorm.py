"""Iteration 024: Mixup augmentation + InstanceNorm + FIR.

Hypothesis: Cross-subject variability is the bottleneck. Mixup interpolates
between training windows (from potentially different subjects), creating
virtual intermediate subjects. Since the scalp-to-inear mapping is
approximately linear (y ≈ W*x), mixup is theoretically well-motivated:
λ*y_i + (1-λ)*y_j ≈ W*(λ*x_i + (1-λ)*x_j).

Combined with InstanceNorm (iter020) to handle amplitude differences,
and the proven training recipe (combined loss + corr validation + channel dropout).

Expected: +0.003-0.010 over iter017 if inter-subject interpolation helps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear, SpatioTemporalFIR


class InstanceNormFIR(nn.Module):
    """FIR filter with InstanceNorm at input."""

    def __init__(self, C_in, C_out, filter_length=7):
        super().__init__()
        self.inorm = nn.InstanceNorm1d(C_in, affine=True)
        self.fir = SpatioTemporalFIR(C_in, C_out, filter_length, mode="acausal")

    def forward(self, x):
        return self.fir(self.inorm(x))


class CorrMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_m * target_m).sum(dim=-1)
        pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
        target_std = (target_m ** 2).sum(dim=-1).sqrt()
        r = cov / (pred_std * target_std + 1e-8)
        return self.alpha * mse + (1 - self.alpha) * (1.0 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for scalp, inear in loader:
            scalp, inear = scalp.to(device), inear.to(device)
            pred = model(scalp)
            pred_m = pred - pred.mean(dim=-1, keepdim=True)
            target_m = inear - inear.mean(dim=-1, keepdim=True)
            cov = (pred_m * target_m).sum(dim=-1)
            r = cov / ((pred_m**2).sum(dim=-1).sqrt() * (target_m**2).sum(dim=-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def train_one_epoch_mixup(model, loader, loss_fn, optimizer, device,
                          grad_clip=1.0, channel_drop_prob=0.15,
                          mixup_alpha=0.4):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for scalp, inear in loader:
        scalp, inear = scalp.to(device), inear.to(device)

        # Mixup: interpolate random pairs within the batch
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(scalp.shape[0], device=device)
            scalp = lam * scalp + (1 - lam) * scalp[idx]
            inear = lam * inear + (1 - lam) * inear[idx]

        # Channel dropout
        if channel_drop_prob > 0:
            mask = (torch.rand(scalp.shape[0], scalp.shape[1], 1, device=device)
                    > channel_drop_prob).float()
            scalp = scalp * mask / (1 - channel_drop_prob)

        optimizer.zero_grad()
        pred = model(scalp)
        loss = loss_fn(pred, inear)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return {"train_loss": total_loss / max(n_batches, 1)}


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    # CF init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = InstanceNormFIR(C_scalp, C_inear, filter_length=7).to(device)

    with torch.no_grad():
        model.fir.conv.weight.zero_()
        center = model.fir.filter_length // 2
        model.fir.conv.weight[:, :, center] = cf.W.float()

    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, 151):
        train_one_epoch_mixup(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15, mixup_alpha=0.4,
        )
        val_r = validate_correlation(model, val_loader, device)
        scheduler.step()
        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model
