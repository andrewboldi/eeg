"""Iteration 011: FIR filter with channel dropout augmentation.

Hypothesis: The FIR model (r=0.373) overfits to channel correlations
present in training subjects. Randomly zeroing out scalp channels
during training forces the model to use redundant spatial information,
improving cross-subject generalization.

This combines the best architecture (FIR with CF init) with a simple
but effective regularization strategy.
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


def train_one_epoch_with_dropout(model, loader, loss_fn, optimizer, device,
                                  grad_clip=1.0, channel_drop_prob=0.2):
    """Train one epoch with random channel dropout."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for scalp, inear in loader:
        scalp = scalp.to(device)
        inear = inear.to(device)

        # Random channel dropout: zero out each channel with probability p
        if channel_drop_prob > 0:
            mask = (torch.rand(scalp.shape[0], scalp.shape[1], 1, device=device)
                    > channel_drop_prob).float()
            # Scale up to maintain expected value
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
    """FIR filter with channel dropout augmentation."""
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = SpatioTemporalFIR(
        C_in=C_scalp, C_out=C_inear, filter_length=7, mode="acausal"
    ).to(device)

    # CF center-tap init
    with torch.no_grad():
        model.conv.weight.zero_()
        center = model.filter_length // 2
        model.conv.weight[:, :, center] = cf.W.float()

    loss_fn = TimeDomainMSE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 101):
        train_one_epoch_with_dropout(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15
        )
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model
