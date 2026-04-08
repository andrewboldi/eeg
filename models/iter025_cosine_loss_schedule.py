"""Iteration 025: Cosine loss schedule — MSE-to-correlation annealing.

Hypothesis: iter017 uses a fixed 50/50 MSE+correlation loss. But early training
needs MSE for stable gradients (large initial errors), while late training
should emphasize correlation to optimize the actual evaluation metric.

A cosine schedule anneals alpha from 0.8 (MSE-heavy) to 0.2 (corr-heavy)
over 150 epochs. This gives the model a stable start while maximizing
correlation in the final phase.

Combined with InstanceNorm + channel dropout from the best recipes.

Expected: +0.002-0.005 over iter017 from better loss scheduling.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear, SpatioTemporalFIR


class InstanceNormFIR(nn.Module):
    def __init__(self, C_in, C_out, filter_length=7):
        super().__init__()
        self.inorm = nn.InstanceNorm1d(C_in, affine=True)
        self.fir = SpatioTemporalFIR(C_in, C_out, filter_length, mode="acausal")

    def forward(self, x):
        return self.fir(self.inorm(x))


class ScheduledCorrMSELoss(nn.Module):
    """Combined loss with time-varying alpha."""

    def __init__(self, alpha_start=0.8, alpha_end=0.2, total_epochs=150):
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_epochs = total_epochs
        self.current_alpha = alpha_start

    def set_epoch(self, epoch):
        # Cosine anneal from alpha_start to alpha_end
        progress = epoch / self.total_epochs
        self.current_alpha = self.alpha_end + 0.5 * (self.alpha_start - self.alpha_end) * (
            1 + math.cos(math.pi * progress)
        )

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_m * target_m).sum(dim=-1)
        pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
        target_std = (target_m ** 2).sum(dim=-1).sqrt()
        r = cov / (pred_std * target_std + 1e-8)
        corr_loss = 1.0 - r.mean()
        return self.current_alpha * mse + (1 - self.current_alpha) * corr_loss


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


def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    grad_clip=1.0, channel_drop_prob=0.15):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for scalp, inear in loader:
        scalp, inear = scalp.to(device), inear.to(device)
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
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = InstanceNormFIR(C_scalp, C_inear, filter_length=7).to(device)

    with torch.no_grad():
        model.fir.conv.weight.zero_()
        center = model.fir.filter_length // 2
        model.fir.conv.weight[:, :, center] = cf.W.float()

    loss_fn = ScheduledCorrMSELoss(alpha_start=0.8, alpha_end=0.2, total_epochs=150)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, 151):
        loss_fn.set_epoch(epoch)
        train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15,
        )
        val_r = validate_correlation(model, val_loader, device)
        scheduler.step()
        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model
