"""Iteration 010: Deep temporal convolution with residual connections.

Hypothesis: The linear FIR (r=0.373) captures single-lag dependencies.
Stacking nonlinear temporal convolutions with residual connections and
batch normalization can learn multi-scale temporal features across
different EEG frequency components within the 1-9 Hz band.

Architecture:
1. Spatial mixing layer (1x1 conv, initialized from closed-form)
2. N temporal blocks: depthwise conv -> BN -> GELU -> residual
3. Output projection (1x1 conv to C_inear channels)

This is a lightweight version of the Conv Encoder adapted for 20 Hz data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.losses import TimeDomainMSE
from src.models import ClosedFormLinear
from src.train import train_one_epoch, validate


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=kernel_size // 2, groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # Pointwise mixing
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.drop(self.act(self.bn(self.conv(x))))
        x = self.drop(self.act(self.bn2(self.pw(x))))
        return x + residual


class DeepTemporalModel(nn.Module):
    def __init__(self, C_in: int, C_out: int, H: int = 32,
                 n_blocks: int = 3, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        # Spatial mixing (like 1x1 conv)
        self.spatial = nn.Conv1d(C_in, H, 1, bias=False)
        self.spatial_bn = nn.BatchNorm1d(H)

        # Temporal blocks
        self.blocks = nn.Sequential(*[
            TemporalBlock(H, kernel_size, dropout) for _ in range(n_blocks)
        ])

        # Output projection
        self.out_proj = nn.Conv1d(H, C_out, 1)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.spatial_bn(self.spatial(x)))
        x = self.blocks(x)
        return self.out_proj(x)


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Train deep temporal model with architecture search over H and n_blocks."""

    # Closed-form for reference
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Single config, fast iteration (100 epochs)
    model = DeepTemporalModel(
        C_in=C_scalp, C_out=C_inear, H=32, n_blocks=3, kernel_size=5, dropout=0.1
    ).to(device)

    loss_fn = TimeDomainMSE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 101):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=1.0)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model
