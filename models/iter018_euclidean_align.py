"""Iteration 018: Euclidean alignment + FIR + channel dropout.

Hypothesis: Cross-subject variability in scalp covariance is the main
bottleneck. Euclidean alignment (He & Wu, 2020) whitens each subject's
data to identity covariance, making the spatial filter covariance-independent.
This is a standard BCI technique that significantly improves cross-subject
transfer by removing subject-specific volume conduction effects.

Strategy:
1. Training: whiten all training data using the pooled covariance
2. Train FIR + dropout on whitened data
3. Test: whiten each test batch using its own batch covariance
4. This makes the model invariant to subject-specific covariance

At test time, whitening is done per-batch (32 windows × 40 samples
= 1280 timepoints per channel — enough for a stable 27×27 estimate).

Expected: +0.010-0.020 in mean r (to ~0.386-0.396).
Risk: Per-batch covariance may be noisy for small batches.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.losses import TimeDomainMSE
from src.models import ClosedFormLinear, SpatioTemporalFIR
from src.train import validate


def compute_whitening_matrix(X: np.ndarray, reg=1e-5) -> np.ndarray:
    """Compute R^{-1/2} for whitening.

    X: (N, C, T) array.
    Returns: (C, C) whitening matrix.
    """
    X = X.astype(np.float64)
    N, C, T = X.shape

    # Pool covariance across all windows
    R = np.zeros((C, C))
    for i in range(N):
        R += X[i] @ X[i].T
    R /= (N * T)
    R += reg * np.eye(C)

    # R^{-1/2} via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return R_inv_sqrt


class WhitenedFIR(nn.Module):
    """FIR model with per-batch Euclidean alignment."""

    def __init__(self, fir: SpatioTemporalFIR, C: int, reg=1e-5):
        super().__init__()
        self.fir = fir
        self.C = C
        self.reg = reg

    def _whiten_batch(self, x):
        """Whiten a batch using its own covariance."""
        B, C, T = x.shape
        # Flatten to compute covariance: (B*T, C)
        samples = x.permute(0, 2, 1).reshape(-1, C)
        mu = samples.mean(0, keepdim=True)
        centered = samples - mu
        R = (centered.T @ centered) / centered.shape[0]
        R = R + self.reg * torch.eye(C, device=x.device)

        # R^{-1/2} via eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(R)
        eigvals = eigvals.clamp(min=1e-10)
        R_inv_sqrt = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.T

        # Whiten: (C, C) @ (B, C, T)
        return torch.einsum('ij,bjt->bit', R_inv_sqrt, x)

    def forward(self, x):
        x_white = self._whiten_batch(x)
        return self.fir(x_white)


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
    """Euclidean alignment + FIR + channel dropout."""

    # Pre-whiten training and validation data
    scalp_np = train_ds.scalp.numpy()
    W_white = compute_whitening_matrix(scalp_np)
    W_white_t = torch.tensor(W_white, dtype=torch.float32)

    # Whiten training data
    scalp_white = np.einsum('ij,njt->nit', W_white, scalp_np)
    train_ds_white = EEGDataset(
        torch.tensor(scalp_white, dtype=torch.float32),
        train_ds.inear
    )

    # Whiten validation data using training whitening matrix
    val_scalp_np = val_ds.scalp.numpy()
    val_scalp_white = np.einsum('ij,njt->nit', W_white, val_scalp_np)
    val_ds_white = EEGDataset(
        torch.tensor(val_scalp_white, dtype=torch.float32),
        val_ds.inear
    )

    # Compute CF on whitened data for initialization
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(scalp_white, train_ds.inear.numpy())

    # Build FIR on whitened space
    fir = SpatioTemporalFIR(
        C_in=C_scalp, C_out=C_inear, filter_length=7, mode="acausal"
    ).to(device)

    with torch.no_grad():
        fir.conv.weight.zero_()
        center = fir.filter_length // 2
        fir.conv.weight[:, :, center] = cf.W.float()

    # Train on pre-whitened data (no per-batch whitening needed during training)
    loss_fn = TimeDomainMSE()
    optimizer = torch.optim.AdamW(fir.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_loader = DataLoader(train_ds_white, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds_white, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 101):
        train_one_epoch_with_dropout(
            fir, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15,
        )
        val_metrics = validate(fir, val_loader, loss_fn, device)
        scheduler.step()
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in fir.state_dict().items()}

    if best_state:
        fir.load_state_dict(best_state)

    # Wrap in WhitenedFIR for test-time per-batch whitening
    model = WhitenedFIR(fir, C_scalp)
    return model.to(device)
