"""Iteration 038: FIR + combined loss on broadband data (1-45 Hz, 128 Hz).

Same proven recipe as iter017/030 but on broadband data:
- FIR filter with CF init
- Combined MSE+corr loss
- Corr validation
- 15% channel dropout
- Mixup augmentation

At 128 Hz, a 7-tap filter = 55ms. We should use more taps to cover
the same temporal range. 33 taps = 256ms, 65 taps = 508ms.
Try filter_length=33 to cover ~250ms (matching the paper's decoder).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear, SpatioTemporalFIR


class CorrMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_m * target_m).sum(dim=-1)
        pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
        target_std = (target_m ** 2).sum(dim=-1).sqrt()
        r = cov / (pred_std * target_std + 1e-8)
        return self.alpha * mse + (1 - self.alpha) * (1.0 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for scalp, inear in loader:
            scalp, inear = scalp.to(device), inear.to(device)
            pred = model(scalp)
            pred_m = pred - pred.mean(dim=-1, keepdim=True)
            target_m = inear - inear.mean(dim=-1, keepdim=True)
            cov = (pred_m * target_m).sum(dim=-1)
            r = cov / ((pred_m**2).sum(dim=-1).sqrt() * (target_m**2).sum(dim=-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # 33 taps at 128 Hz = 258ms temporal context (acausal)
    filter_length = 33
    model = SpatioTemporalFIR(C_scalp, C_inear, filter_length=filter_length,
                               mode="acausal").to(device)

    with torch.no_grad():
        model.conv.weight.zero_()
        center = filter_length // 2
        model.conv.weight[:, :, center] = cf.W.float()

    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, 151):
        model.train()
        for scalp, inear in train_loader:
            scalp, inear = scalp.to(device), inear.to(device)

            # Mixup
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(scalp.shape[0], device=device)
            scalp = lam * scalp + (1 - lam) * scalp[idx]
            inear = lam * inear + (1 - lam) * inear[idx]

            # Channel dropout
            mask = (torch.rand(scalp.shape[0], scalp.shape[1], 1, device=device) > 0.15).float()
            scalp = scalp * mask / 0.85

            optimizer.zero_grad()
            pred = model(scalp)
            loss = loss_fn(pred, inear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_r = validate_correlation(model, val_loader, device)
        scheduler.step()

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: val_r={val_r:.4f} (best={best_val_r:.4f})")

    print(f"Final best val_r: {best_val_r:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model
