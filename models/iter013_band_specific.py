"""Iteration 013: Frequency-band-specific spatial filters.

Hypothesis: Delta (1-4 Hz) and theta (4-9 Hz) EEG have different
spatial distributions due to different neural generators. Separate
spatial filters per band should capture band-specific volume conduction
patterns better than a single broadband filter.

Strategy:
1. Bandpass split each trial into delta (1-4 Hz) and theta (4-9 Hz)
2. Compute closed-form spatial filter for each band
3. Apply each filter to its band, sum the predictions
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset


class BandSpecificFilter(nn.Module):
    """Applies separate spatial filters per frequency band."""

    def __init__(self, C_in, C_out, W_delta, W_theta):
        super().__init__()
        self.register_buffer("W_delta", torch.tensor(W_delta, dtype=torch.float32))
        self.register_buffer("W_theta", torch.tensor(W_theta, dtype=torch.float32))
        self.fs = 20.0

    def _bandpass(self, x_np, low, high, fs=20.0, order=2):
        """Apply bandpass filter. x_np: (N, C, T)"""
        sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
        # Apply along time axis
        filtered = np.zeros_like(x_np)
        for i in range(x_np.shape[0]):
            for c in range(x_np.shape[1]):
                filtered[i, c] = sosfiltfilt(sos, x_np[i, c])
        return filtered

    def forward(self, x):
        # Split into bands on CPU (scipy needed)
        x_np = x.cpu().numpy()
        delta = self._bandpass(x_np, 1.0, 4.0)
        theta = self._bandpass(x_np, 4.0, 9.0)

        delta_t = torch.tensor(delta, dtype=torch.float32, device=x.device)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=x.device)

        pred_delta = torch.einsum("oc,bct->bot", self.W_delta, delta_t)
        pred_theta = torch.einsum("oc,bct->bot", self.W_theta, theta_t)

        return pred_delta + pred_theta


def compute_cf_weights(scalp, inear, reg=1e-4):
    """Compute closed-form W* = R_YX @ inv(R_XX)."""
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


def bandpass_dataset(scalp_np, low, high, fs=20.0, order=2):
    """Bandpass filter entire dataset."""
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    filtered = np.zeros_like(scalp_np)
    for i in range(scalp_np.shape[0]):
        for c in range(scalp_np.shape[1]):
            filtered[i, c] = sosfiltfilt(sos, scalp_np[i, c])
    return filtered


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build band-specific spatial filters."""
    scalp_np = train_ds.scalp.numpy()
    inear_np = train_ds.inear.numpy()

    # Filter training data into bands
    scalp_delta = bandpass_dataset(scalp_np, 1.0, 4.0)
    scalp_theta = bandpass_dataset(scalp_np, 4.0, 9.0)
    inear_delta = bandpass_dataset(inear_np, 1.0, 4.0)
    inear_theta = bandpass_dataset(inear_np, 4.0, 9.0)

    # Compute per-band CF weights
    W_delta = compute_cf_weights(scalp_delta, inear_delta)
    W_theta = compute_cf_weights(scalp_theta, inear_theta)

    model = BandSpecificFilter(C_scalp, C_inear, W_delta, W_theta)
    return model.to(device)
