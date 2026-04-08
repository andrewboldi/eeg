"""Iteration 019: SWA + combined loss + channel dropout.

Hypothesis: Building on iter017 (r=0.378, new best), add Stochastic
Weight Averaging to find a flatter minimum with better cross-subject
generalization. SWA averages model weights over the last portion of
training, which has been shown to improve generalization in many settings.

Combined improvements:
1. Combined MSE + correlation loss (from iter017)
2. Correlation-based validation (from iter017)
3. Channel dropout at 20% (slightly higher than 15% for more regularization)
4. SWA over last 50 epochs
5. 200 total epochs with cosine schedule

Expected: +0.003-0.008 over iter017 (to ~0.381-0.386).
"""

from __future__ import annotations

import copy

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
        mse = torch.mean((pred - target) ** 2)
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_m * target_m).sum(dim=-1)
        pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
        target_std = (target_m ** 2).sum(dim=-1).sqrt()
        r = cov / (pred_std * target_std + 1e-8)
        corr_loss = 1.0 - r.mean()
        return self.alpha * mse + (1 - self.alpha) * corr_loss


def validate_correlation(model, loader, device):
    """Validate using mean Pearson r."""
    model.eval()
    all_r = []
    with torch.no_grad():
        for scalp, inear in loader:
            scalp = scalp.to(device)
            inear = inear.to(device)
            pred = model(scalp)
            pred_m = pred - pred.mean(dim=-1, keepdim=True)
            target_m = inear - inear.mean(dim=-1, keepdim=True)
            cov = (pred_m * target_m).sum(dim=-1)
            pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
            target_std = (target_m ** 2).sum(dim=-1).sqrt()
            r = cov / (pred_std * target_std + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def train_one_epoch_with_dropout(model, loader, loss_fn, optimizer, device,
                                  grad_clip=1.0, channel_drop_prob=0.20):
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
    """SWA + combined loss + channel dropout."""

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Track best model and SWA model
    best_val_r = -1.0
    best_state = None
    swa_state = None
    swa_count = 0
    swa_start = 150  # Start SWA at epoch 150

    for epoch in range(1, 201):
        train_one_epoch_with_dropout(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.20,
        )
        val_r = validate_correlation(model, val_loader, device)
        scheduler.step()

        # Track best single model
        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # SWA: accumulate weights after swa_start
        if epoch >= swa_start:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k]
                swa_count += 1

    # Average SWA weights
    if swa_state is not None and swa_count > 0:
        for k in swa_state:
            swa_state[k] /= swa_count

    # Evaluate both best single and SWA, pick the better one
    candidates = []
    if best_state:
        model.load_state_dict(best_state)
        val_r_best = validate_correlation(model, val_loader, device)
        candidates.append((val_r_best, best_state, "best_single"))

    if swa_state:
        model.load_state_dict(swa_state)
        val_r_swa = validate_correlation(model, val_loader, device)
        candidates.append((val_r_swa, swa_state, "swa"))

    # Pick the one with highest validation correlation
    candidates.sort(key=lambda x: x[0], reverse=True)
    model.load_state_dict(candidates[0][1])

    return model
