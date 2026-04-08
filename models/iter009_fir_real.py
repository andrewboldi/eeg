"""Iteration 009: Spatio-temporal FIR filter on real EEG data.

Hypothesis: The closed-form spatial filter (r=0.366) only captures
instantaneous (zero-lag) relationships. Real volume conduction has
propagation delays, and the 1-9 Hz EEG has temporal autocorrelation.
A short FIR filter (few taps at 20 Hz) can exploit these lags.

At 20 Hz, each tap = 50ms. A 5-tap filter spans 250ms of context.

Strategy:
1. Initialize center tap from closed-form weights (proven critical in iter004)
2. Train with MSE loss, gradient clipping, AdamW
3. Try multiple filter lengths: 3, 5, 7, 11 taps
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.losses import TimeDomainMSE
from src.models import ClosedFormLinear, SpatioTemporalFIR
from src.train import train_one_epoch, validate


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Train FIR filter with closed-form center-tap initialization."""

    # First compute closed-form weights for initialization
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    best_model = None
    best_val_r = -1.0

    for filter_length in [3, 5, 7, 11]:
        model = SpatioTemporalFIR(
            C_in=C_scalp, C_out=C_inear,
            filter_length=filter_length, mode="acausal",
        ).to(device)

        # Initialize center tap from closed-form
        with torch.no_grad():
            model.conv.weight.zero_()
            center = filter_length // 2
            model.conv.weight[:, :, center] = cf.W.float()

        loss_fn = TimeDomainMSE()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, 151):
            train_one_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=1.0)
            val_metrics = validate(model, val_loader, loss_fn, device)
            scheduler.step()

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        # Evaluate on val set to pick best filter length
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for s, i in val_loader:
                preds.append(model(s.to(device)).cpu().numpy())
                targets.append(i.numpy())
        pred = np.concatenate(preds).flatten()
        target = np.concatenate(targets).flatten()
        val_r = float(np.corrcoef(pred, target)[0, 1])

        if val_r > best_val_r:
            best_val_r = val_r
            best_model = model

    return best_model
