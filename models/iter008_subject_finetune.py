"""Iteration 008: Subject-specific fine-tuning.

Hypothesis: The closed-form baseline (r=0.366) is limited by inter-subject
anatomical variability. By using a small amount of target-subject data to
adapt the spatial filter, we can significantly improve prediction.

Strategy:
1. Compute pooled closed-form weights W_pool from subjects 1-12
2. For each test subject, use N calibration trials to compute W_subj
3. Final weights = (1-alpha) * W_pool + alpha * W_subj (interpolation)
4. Optimize alpha on validation data

This is equivalent to Tikhonov-regularized subject-specific adaptation,
where the pooled model acts as a prior.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.losses import TimeDomainMSE
from src.models import ClosedFormLinear


class AdaptiveSpatialFilter(nn.Module):
    """Spatial filter that interpolates between pooled and subject-specific weights."""

    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.register_buffer("W", torch.zeros(C_out, C_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oc,bct->bot", self.W, x)


def compute_closed_form_weights(scalp: np.ndarray, inear: np.ndarray, reg: float = 1e-4):
    """Compute W* = R_YX @ inv(R_XX) from data arrays."""
    scalp = scalp.astype(np.float64)
    inear = inear.astype(np.float64)
    N, C_in, T = scalp.shape
    _, C_out, _ = inear.shape

    R_XX = np.zeros((C_in, C_in))
    R_YX = np.zeros((C_out, C_in))
    for i in range(N):
        R_XX += scalp[i] @ scalp[i].T
        R_YX += inear[i] @ scalp[i].T
    R_XX /= N
    R_YX /= N
    R_XX += reg * np.eye(C_in)

    return R_YX @ np.linalg.inv(R_XX)


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build adaptive spatial filter with subject-specific fine-tuning.

    Since benchmark.py calls this once per test subject, we use the
    validation set as a proxy for "calibration data from the target subject."

    In practice, the benchmark trains on subjects 1-12 and tests on 13/14/15.
    The val_ds here is 10% of subjects 1-12 data — not subject-specific.

    To properly test subject-specific adaptation, we need to modify the
    benchmark protocol. For now, we test a regularized closed-form that
    uses both train and val data with different regularization strengths.
    """
    # Pooled weights from training data
    W_pool = compute_closed_form_weights(
        train_ds.scalp.numpy(), train_ds.inear.numpy(), reg=1e-4
    )

    # Also compute from val data (simulates having some extra data)
    W_val = compute_closed_form_weights(
        val_ds.scalp.numpy(), val_ds.inear.numpy(), reg=1e-4
    )

    # Try different regularization strengths and pick best on val
    best_r = -1.0
    best_W = W_pool

    for reg in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        W = compute_closed_form_weights(
            np.concatenate([train_ds.scalp.numpy(), val_ds.scalp.numpy()]),
            np.concatenate([train_ds.inear.numpy(), val_ds.inear.numpy()]),
            reg=reg,
        )

        # Evaluate on val set
        model = AdaptiveSpatialFilter(C_scalp, C_inear)
        model.W = torch.tensor(W, dtype=torch.float32)
        model = model.to(device)
        model.eval()

        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        preds, targets = [], []
        with torch.no_grad():
            for s, i in val_loader:
                preds.append(model(s.to(device)).cpu().numpy())
                targets.append(i.numpy())
        pred = np.concatenate(preds).flatten()
        target = np.concatenate(targets).flatten()
        r = np.corrcoef(pred, target)[0, 1]

        if r > best_r:
            best_r = r
            best_W = W

    model = AdaptiveSpatialFilter(C_scalp, C_inear)
    model.W = torch.tensor(best_W, dtype=torch.float32)
    return model.to(device)
