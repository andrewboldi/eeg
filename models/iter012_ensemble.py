"""Iteration 012: Ensemble of closed-form + FIR with channel dropout.

Hypothesis: Closed-form and FIR make complementary errors. The CF
captures the optimal instantaneous spatial mix, while FIR captures
temporal lags. A learned weighted average should outperform either alone.

Strategy: Train FIR with channel dropout (best so far), then combine
predictions with CF using optimized weights on validation set.
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
from models.iter011_channel_dropout import train_one_epoch_with_dropout


class EnsembleModel(nn.Module):
    """Weighted ensemble of two models."""

    def __init__(self, model_a: nn.Module, model_b: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.alpha = alpha  # weight on model_a

    def forward(self, x):
        with torch.no_grad():
            pred_a = self.model_a(x)
            pred_b = self.model_b(x)
        return self.alpha * pred_a + (1 - self.alpha) * pred_b


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build ensemble of CF + FIR with channel dropout."""

    # 1. Closed-form model
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)

    # 2. FIR with channel dropout (same as iter011)
    fir = SpatioTemporalFIR(
        C_in=C_scalp, C_out=C_inear, filter_length=7, mode="acausal"
    ).to(device)

    with torch.no_grad():
        fir.conv.weight.zero_()
        center = fir.filter_length // 2
        fir.conv.weight[:, :, center] = cf.W.float()

    loss_fn = TimeDomainMSE()
    optimizer = torch.optim.AdamW(fir.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 101):
        train_one_epoch_with_dropout(
            fir, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15
        )
        val_metrics = validate(fir, val_loader, loss_fn, device)
        scheduler.step()
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in fir.state_dict().items()}

    if best_state:
        fir.load_state_dict(best_state)

    # 3. Find optimal ensemble weight on val set
    cf.eval()
    fir.eval()

    best_alpha = 0.5
    best_r = -1.0

    for alpha in np.arange(0.0, 1.05, 0.05):
        preds, targets = [], []
        with torch.no_grad():
            for s, i in val_loader:
                s = s.to(device)
                p = alpha * cf(s) + (1 - alpha) * fir(s)
                preds.append(p.cpu().numpy())
                targets.append(i.numpy())
        r = float(np.corrcoef(
            np.concatenate(preds).flatten(),
            np.concatenate(targets).flatten()
        )[0, 1])
        if r > best_r:
            best_r = r
            best_alpha = alpha

    return EnsembleModel(cf, fir, alpha=best_alpha)
