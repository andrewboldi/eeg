"""Iteration 027: PLS Regression spatial filter.

Hypothesis: The closed-form solution W* = R_YX @ inv(R_XX) minimizes MSE.
But our metric is Pearson r, which measures covariance alignment.
Partial Least Squares (PLS) regression finds components that maximize
covariance between input and output, which is more aligned with Pearson r.

PLS finds latent components in X with maximum covariance with Y, then
regresses Y on these components. With enough components (n_components=12),
PLS captures all relevant structure while biasing toward high-covariance
directions.

This is a closed-form solution (no training loop), so it's very fast.
We also add a FIR refinement phase using the PLS solution as initialization.

Expected: +0.002-0.008 if covariance-optimal differs from MSE-optimal.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import SpatioTemporalFIR


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
    # Flatten windows to (N*T, C) for PLS
    scalp_np = train_ds.scalp.numpy()  # (N, C_in, T)
    inear_np = train_ds.inear.numpy()  # (N, C_out, T)
    N, C_in, T = scalp_np.shape
    C_out = inear_np.shape[1]

    X_flat = scalp_np.transpose(0, 2, 1).reshape(-1, C_in)   # (N*T, C_in)
    Y_flat = inear_np.transpose(0, 2, 1).reshape(-1, C_out)  # (N*T, C_out)

    # Fit PLS regression
    n_components = min(C_in, C_out, 12)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_flat, Y_flat)

    # Extract spatial filter W: Y = X @ coef_ + intercept_
    # coef_ is (C_in, C_out), we need (C_out, C_in) for conv1d
    # sklearn PLSRegression.coef_ is (n_targets, n_features) = (C_out, C_in)
    W_pls = torch.tensor(pls.coef_, dtype=torch.float32)  # (C_out, C_in)

    # Initialize FIR model with PLS solution at center tap
    model = SpatioTemporalFIR(C_in, C_out, filter_length=7, mode="acausal").to(device)

    with torch.no_grad():
        model.conv.weight.zero_()
        center = model.filter_length // 2
        model.conv.weight[:, :, center] = W_pls.to(device)

    # Fine-tune with combined loss + corr validation
    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, 151):
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
