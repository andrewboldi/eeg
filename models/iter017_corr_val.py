"""Iteration 017: FIR + channel dropout with combined loss & correlation validation.

Hypothesis: iter015 trained on correlation loss but validated on MSE,
so early stopping selected MSE-optimal models from the warm-start phase.
Fix: validate on correlation too. Also use a combined loss (MSE + corr)
throughout training for more stable gradients.

Loss = 0.5 * MSE + 0.5 * (1 - pearson_r)
Validation: use correlation-based metric for early stopping.

Expected: +0.003-0.008 over iter011 (to ~0.379-0.384).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.losses import TimeDomainMSE
from src.models import ClosedFormLinear, SpatioTemporalFIR


class CorrMSELoss(nn.Module):
    """Combined MSE + negative correlation loss."""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # MSE component
        mse = torch.mean((pred - target) ** 2)

        # Pearson correlation component
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_m * target_m).sum(dim=-1)
        pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
        target_std = (target_m ** 2).sum(dim=-1).sqrt()
        r = cov / (pred_std * target_std + 1e-8)
        corr_loss = 1.0 - r.mean()

        return self.alpha * mse + (1 - self.alpha) * corr_loss


def validate_correlation(model, loader, device):
    """Validate using mean Pearson r (higher is better)."""
    model.eval()
    all_r = []
    with torch.no_grad():
        for scalp, inear in loader:
            scalp = scalp.to(device)
            inear = inear.to(device)
            pred = model(scalp)

            # Per-channel, per-sample Pearson r
            pred_m = pred - pred.mean(dim=-1, keepdim=True)
            target_m = inear - inear.mean(dim=-1, keepdim=True)
            cov = (pred_m * target_m).sum(dim=-1)
            pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
            target_std = (target_m ** 2).sum(dim=-1).sqrt()
            r = cov / (pred_std * target_std + 1e-8)
            all_r.append(r.cpu())

    return torch.cat(all_r).mean().item()


def train_one_epoch_with_dropout(model, loader, loss_fn, optimizer, device,
                                  grad_clip=1.0, channel_drop_prob=0.15):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for scalp, inear in loader:
        scalp = scalp.to(device)
        inear = inear.to(device)
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
    """FIR + channel dropout with combined loss & correlation validation."""

    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = SpatioTemporalFIR(
        C_in=C_scalp, C_out=C_inear, filter_length=7, mode="acausal"
    ).to(device)

    with torch.no_grad():
        model.conv.weight.zero_()
        center = model.filter_length // 2
        model.conv.weight[:, :, center] = cf.W.float()

    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, 151):
        train_one_epoch_with_dropout(
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
