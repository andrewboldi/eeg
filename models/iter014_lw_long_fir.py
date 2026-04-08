"""Iteration 014: Longer FIR + channel dropout + improved training.

Hypothesis: iter009 tested longer filters (up to 11 taps) but WITHOUT
channel dropout. iter011 showed that channel dropout helps the 7-tap FIR.
Combining longer filters (11, 15 taps) with channel dropout should allow
the model to capture more temporal structure without overfitting.

Additional improvements:
- Warm restart cosine schedule (2 cycles of 75 epochs)
- Slightly higher weight decay (2e-3 vs 1e-3)
- Standard CF init (proven reliable)

Expected: +0.003-0.008 in mean r (to ~0.379-0.384).
Risk: Longer filters may still not help — the data is very narrowband.
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
                                  grad_clip=1.0, channel_drop_prob=0.15):
    """Train one epoch with random channel dropout."""
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
    """Longer FIR + channel dropout + warm restarts."""

    # Standard CF init (proven reliable)
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    best_model = None
    best_val_loss = float("inf")

    for filter_length in [7, 11, 15]:
        model = SpatioTemporalFIR(
            C_in=C_scalp, C_out=C_inear,
            filter_length=filter_length, mode="acausal",
        ).to(device)

        # CF center-tap init
        with torch.no_grad():
            model.conv.weight.zero_()
            center = filter_length // 2
            model.conv.weight[:, :, center] = cf.W.float()

        loss_fn = TimeDomainMSE()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
        # Cosine annealing with warm restarts: 2 cycles of 75 epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=75, T_mult=1
        )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        this_best_loss = float("inf")
        best_state = None

        for epoch in range(1, 151):
            train_one_epoch_with_dropout(
                model, train_loader, loss_fn, optimizer, device,
                grad_clip=1.0, channel_drop_prob=0.15,
            )
            val_metrics = validate(model, val_loader, loss_fn, device)
            scheduler.step()

            if val_metrics["val_loss"] < this_best_loss:
                this_best_loss = val_metrics["val_loss"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        if this_best_loss < best_val_loss:
            best_val_loss = this_best_loss
            best_model = model

    return best_model
