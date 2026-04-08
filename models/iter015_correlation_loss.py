"""Iteration 015: FIR + channel dropout trained with Pearson correlation loss.

Hypothesis: The benchmark evaluates Pearson r, but all prior models train
on MSE. MSE penalizes scale/offset errors that don't affect correlation.
A Pearson r loss directly optimizes the evaluation metric, focusing model
capacity on matching the signal shape rather than amplitude.

Strategy:
1. Loss = 1 - mean(pearson_r per channel per sample)
2. Same architecture as iter011 (FIR 7-tap + CF init + channel dropout)
3. MSE warm-start for 20 epochs, then switch to correlation loss
   (pure correlation loss can be unstable early in training)

Expected: +0.005-0.015 in mean r (to ~0.381-0.391).
Risk: Correlation loss gradients may be noisy for short windows (T=40).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.losses import TimeDomainMSE
from src.models import ClosedFormLinear, SpatioTemporalFIR
from src.train import validate


class PearsonCorrLoss(nn.Module):
    """Negative Pearson correlation loss: 1 - mean(r)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, T)
            target: (B, C, T)
        Returns:
            Scalar loss in [0, 2] (0 = perfect correlation).
        """
        # Center along time axis
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)

        # Covariance and standard deviations
        cov = (pred_m * target_m).sum(dim=-1)  # (B, C)
        pred_std = (pred_m ** 2).sum(dim=-1).sqrt()  # (B, C)
        target_std = (target_m ** 2).sum(dim=-1).sqrt()  # (B, C)

        # Pearson r per channel per sample
        r = cov / (pred_std * target_std + 1e-8)  # (B, C)

        return 1.0 - r.mean()


def train_one_epoch_with_dropout(model, loader, loss_fn, optimizer, device,
                                  grad_clip=1.0, channel_drop_prob=0.15):
    """Train one epoch with channel dropout."""
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
    """FIR + channel dropout with correlation loss."""

    # CF init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = SpatioTemporalFIR(
        C_in=C_scalp, C_out=C_inear, filter_length=7, mode="acausal"
    ).to(device)

    with torch.no_grad():
        model.conv.weight.zero_()
        center = model.filter_length // 2
        model.conv.weight[:, :, center] = cf.W.float()

    mse_loss = TimeDomainMSE()
    corr_loss = PearsonCorrLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=130)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 131):
        # MSE warm-start for first 20 epochs, then correlation loss
        loss_fn = mse_loss if epoch <= 20 else corr_loss

        train_one_epoch_with_dropout(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15
        )

        # Always validate on MSE for comparable early stopping
        val_metrics = validate(model, val_loader, mse_loss, device)
        scheduler.step()

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model
