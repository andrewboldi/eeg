"""Template for submitting a model to the benchmark.

Copy this file, rename it (e.g., models/iter008_temporal_attention.py),
and implement build_and_train(). Then run:

    uv run python scripts/benchmark.py --model-fn models/your_model.py --name your_model

The benchmark will call build_and_train() with the training data and
evaluate the returned model on the fixed test set (subjects 13-15).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.losses import TimeDomainMSE
from src.train import train_one_epoch, validate


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build and train your model.

    Args:
        train_ds: Training dataset (subjects 1-12, ~90%)
        val_ds: Validation dataset (subjects 1-12, ~10%)
        C_scalp: Number of scalp channels (27)
        C_inear: Number of in-ear channels (12)
        device: torch device (cuda or cpu)

    Returns:
        Trained nn.Module that maps (batch, C_scalp, T) -> (batch, C_inear, T)
    """
    # === YOUR MODEL HERE ===
    # Example: simple linear baseline
    from src.models import ClosedFormLinear
    model = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    model.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    return model.to(device)
