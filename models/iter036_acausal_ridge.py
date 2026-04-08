"""Iteration 036: Acausal multi-lag Ledoit-Wolf ridge regression.

Fix for iter034/035: they used causal-only lags (past only), but our
FIR model is acausal (sees both past and future). Since this is offline
prediction, acausal lags are valid and capture bidirectional temporal
structure.

Uses lags [-3, ..., 0, ..., +3] = 7 total offsets (350ms window centered
on current sample), matching the 7-tap acausal FIR. Ledoit-Wolf shrinkage
for optimal regularization. Then fine-tune with combined loss.

Expected: Significant improvement over iter034's causal-only approach.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
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
    scalp_np = train_ds.scalp.numpy()   # (N, C_in, T)
    inear_np = train_ds.inear.numpy()   # (N, C_out, T)
    N, C_in, T = scalp_np.shape
    C_out = inear_np.shape[1]

    # Acausal lags: [-3, -2, -1, 0, +1, +2, +3] = filter_length 7
    half_lags = 3
    filter_length = 2 * half_lags + 1  # 7

    # Build acausal lagged features
    def make_acausal_lagged(data, half_lags):
        """(N, C, T) -> (N*T_valid, C*filter_length) using acausal lags."""
        N, C, T = data.shape
        T_valid = T - 2 * half_lags
        features = np.zeros((N, T_valid, C * (2 * half_lags + 1)), dtype=np.float32)
        for i, lag in enumerate(range(-half_lags, half_lags + 1)):
            start = half_lags + lag
            end = start + T_valid
            features[:, :, i * C:(i + 1) * C] = data[:, :, start:end].transpose(0, 2, 1)
        return features.reshape(-1, C * (2 * half_lags + 1))

    X_flat = make_acausal_lagged(scalp_np, half_lags)
    # Trim target to match valid region
    Y_flat = inear_np[:, :, half_lags:T - half_lags].transpose(0, 2, 1).reshape(-1, C_out)

    print(f"Acausal features: X={X_flat.shape}, Y={Y_flat.shape}")

    # Ledoit-Wolf shrinkage
    lw = LedoitWolf()
    lw.fit(X_flat)
    R_XX = lw.covariance_
    shrinkage = lw.shrinkage_
    print(f"Ledoit-Wolf shrinkage: {shrinkage:.6f}")

    # Cross-covariance and solve
    X_mean = X_flat.mean(axis=0)
    Y_mean = Y_flat.mean(axis=0)
    Xc = X_flat - X_mean
    Yc = Y_flat - Y_mean
    R_YX = (Yc.T @ Xc) / (len(X_flat) - 1)

    W = np.linalg.solve(R_XX.T, R_YX.T).T  # (C_out, C_in * filter_length)
    print(f"LW solution: W shape {W.shape}")

    # Initialize SpatioTemporalFIR with LW solution
    model = SpatioTemporalFIR(C_in, C_out, filter_length=filter_length, mode="acausal").to(device)

    with torch.no_grad():
        # W layout: [lag_{-3}*C_in, lag_{-2}*C_in, ..., lag_{+3}*C_in]
        # Conv1d weight: (C_out, C_in, K) where K[0] is applied to earliest time
        # For acausal conv with padding, kernel index 0 = lag -half_lags
        for i in range(filter_length):
            model.conv.weight[:, :, i] = torch.tensor(
                W[:, i * C_in:(i + 1) * C_in], dtype=torch.float32
            )
        if model.conv.bias is not None:
            model.conv.bias.zero_()

    # Evaluate closed-form solution
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    lw_r = validate_correlation(model, val_loader, device)
    print(f"Ledoit-Wolf acausal closed-form val r: {lw_r:.6f}")

    # Fine-tune with combined loss
    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_r = lw_r
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(1, 101):
        train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15,
        )
        val_r = validate_correlation(model, val_loader, device)
        scheduler.step()
        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Best val r after fine-tuning: {best_val_r:.6f}")
    model.load_state_dict(best_state)
    return model
