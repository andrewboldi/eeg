"""Iteration 016: Residual FIR learning.

Hypothesis: The CF solution captures the dominant linear spatial mapping.
A FIR model initialized from CF effectively re-learns the CF solution
plus small corrections. Instead, explicitly decompose:
    y_hat = W_cf * x + FIR_residual(x)
where FIR_residual is trained to predict only the residual y - W_cf*x.

This forces the model to focus learning capacity on temporal dynamics
and cross-subject patterns that the CF filter misses, rather than
re-learning the already-optimal spatial weights.

Strategy:
1. Compute CF prediction on all training data
2. Create residual targets: r = y - y_hat_cf
3. Train FIR on residual with channel dropout
4. At test time: predict = CF(x) + FIR(x)

Expected: +0.003-0.008 in mean r.
Risk: Residuals may be mostly noise → FIR overfits to noise.
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


class ResidualModel(nn.Module):
    """CF + learned FIR residual."""

    def __init__(self, cf_model: ClosedFormLinear, fir_model: SpatioTemporalFIR):
        super().__init__()
        self.cf = cf_model
        self.fir = fir_model
        # Freeze CF weights
        for p in self.cf.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            cf_pred = self.cf(x)
        residual_pred = self.fir(x)
        return cf_pred + residual_pred


def train_one_epoch_with_dropout(model, loader, loss_fn, optimizer, device,
                                  grad_clip=1.0, channel_drop_prob=0.15):
    """Train one epoch with channel dropout (only FIR params update)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for scalp, inear in loader:
        scalp = scalp.to(device)
        inear = inear.to(device)

        if channel_drop_prob > 0:
            mask = (torch.rand(scalp.shape[0], scalp.shape[1], 1, device=device)
                    > channel_drop_prob).float()
            scalp_aug = scalp * mask / (1 - channel_drop_prob)
        else:
            scalp_aug = scalp

        optimizer.zero_grad()
        pred = model(scalp_aug)
        loss = loss_fn(pred, inear)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.fir.parameters(), grad_clip)
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
    """Residual FIR: CF baseline + learned residual."""

    # Fit CF model
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)

    # FIR for residual (initialize to zero — residual starts at 0)
    fir = SpatioTemporalFIR(
        C_in=C_scalp, C_out=C_inear, filter_length=7, mode="acausal"
    ).to(device)
    with torch.no_grad():
        fir.conv.weight.zero_()

    model = ResidualModel(cf, fir).to(device)

    loss_fn = TimeDomainMSE()
    # Only optimize FIR parameters
    optimizer = torch.optim.AdamW(fir.parameters(), lr=5e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 121):
        train_one_epoch_with_dropout(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15,
        )
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model
