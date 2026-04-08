"""Iteration 035: Ledoit-Wolf shrinkage + multi-lag temporal filter.

Hypothesis: The Ear-SAAD paper (Geirnaert 2025) uses Ledoit-Wolf analytical
shrinkage for L2 regularization, which outperforms cross-validation.
Combined with a proper Hankel matrix (multi-lag temporal features),
this is the optimal linear approach for this narrowband data.

Key changes from baseline:
1. Multi-lag features: lags 0 to 8 (400ms at 20Hz), matching the paper
2. Ledoit-Wolf shrinkage: analytically optimal regularization
3. Pure closed-form: no SGD, no training loop
4. Also try with FIR fine-tuning after initialization

Expected: +0.01-0.03 from proper regularization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import SpatioTemporalFIR


class MultiLagPredictor(nn.Module):
    """Multi-lag linear predictor as a Conv1d for benchmark compatibility."""

    def __init__(self, C_in, C_out, n_lags):
        super().__init__()
        self.n_lags = n_lags
        # Conv1d with kernel_size = n_lags+1, causal padding
        self.conv = nn.Conv1d(C_in, C_out, kernel_size=n_lags + 1,
                              padding=0, bias=True)

    def forward(self, x):
        # Causal: pad left side
        x_padded = torch.nn.functional.pad(x, (self.n_lags, 0))
        return self.conv(x_padded)


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
    n_lags = 8  # 0 to 8 = 9 taps (450ms at 20Hz)

    # --- Step 1: Build Hankel matrix (multi-lag features) ---
    def make_lagged(data, n_lags):
        """(N, C, T) -> (N, C*(n_lags+1), T) with causal lags"""
        parts = [data]
        for lag in range(1, n_lags + 1):
            shifted = np.zeros_like(data)
            shifted[:, :, lag:] = data[:, :, :T - lag]
            parts.append(shifted)
        return np.concatenate(parts, axis=1)

    X_lagged = make_lagged(scalp_np, n_lags)  # (N, C_in*(n_lags+1), T)
    n_features = C_in * (n_lags + 1)

    # Flatten to (N*T, features) and (N*T, C_out)
    X_flat = X_lagged.transpose(0, 2, 1).reshape(-1, n_features)
    Y_flat = inear_np.transpose(0, 2, 1).reshape(-1, C_out)

    # --- Step 2: Ledoit-Wolf shrinkage covariance ---
    # Compute R_XX with shrinkage
    lw = LedoitWolf()
    lw.fit(X_flat)
    R_XX_shrunk = lw.covariance_  # (features, features)
    shrinkage = lw.shrinkage_
    print(f"Ledoit-Wolf shrinkage coefficient: {shrinkage:.6f}")

    # Cross-covariance R_YX
    X_mean = X_flat.mean(axis=0)
    Y_mean = Y_flat.mean(axis=0)
    X_centered = X_flat - X_mean
    Y_centered = Y_flat - Y_mean
    R_YX = (Y_centered.T @ X_centered) / (len(X_flat) - 1)  # (C_out, features)

    # --- Step 3: Solve W = R_YX @ inv(R_XX_shrunk) ---
    W = np.linalg.solve(R_XX_shrunk.T, R_YX.T).T  # (C_out, features)
    bias = Y_mean - W @ X_mean  # (C_out,)

    print(f"LW solution: W shape {W.shape}, max |W| = {np.abs(W).max():.6f}")

    # --- Step 4: Initialize Conv1d model with LW solution ---
    model = MultiLagPredictor(C_in, C_out, n_lags).to(device)

    with torch.no_grad():
        # Conv1d weight: (C_out, C_in, kernel_size)
        # Our W: (C_out, C_in*(n_lags+1)) = [W_lag0 | W_lag1 | ... | W_lagL]
        W_reshaped = W.reshape(C_out, n_lags + 1, C_in).transpose(0, 2, 1)
        # Conv1d expects (C_out, C_in, K) where K[0] is oldest lag
        # Our lag ordering: lag0 is most recent, lagL is oldest
        # Conv1d applies kernel as: sum over k of w[k] * x[t - (K-1) + k]
        # So kernel index 0 = oldest lag, index K-1 = no lag
        # We need to reverse: put lag0 at position K-1, lagL at position 0
        W_conv = np.zeros((C_out, C_in, n_lags + 1), dtype=np.float32)
        for lag in range(n_lags + 1):
            W_conv[:, :, n_lags - lag] = W[:, lag * C_in:(lag + 1) * C_in]
        model.conv.weight.copy_(torch.tensor(W_conv))
        model.conv.bias.copy_(torch.tensor(bias, dtype=torch.float32))

    # --- Step 5: Fine-tune with combined loss + corr validation ---
    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    # Evaluate LW solution first
    lw_r = validate_correlation(model, val_loader, device)
    print(f"Ledoit-Wolf closed-form val r: {lw_r:.6f}")
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

    if best_state:
        model.load_state_dict(best_state)
    return model
