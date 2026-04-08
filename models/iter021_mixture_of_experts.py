"""Iteration 021: Mixture of FIR Experts with input-dependent gating.

Hypothesis: Different test subjects need different spatial filters.
A single FIR filter is optimal for the average training subject but
suboptimal for outlier subjects (like Subject 14, r=0.27).

A Mixture of Experts (MoE) learns K different FIR filters and selects
them based on the input's channel statistics. This allows the model
to adapt to different head geometries at test time without needing
subject labels.

Architecture:
1. InstanceNorm input
2. K=4 parallel FIR expert filters (each: 27→12, filter_length=7)
3. Gating network: channel_means → softmax weights
4. Weighted combination of expert outputs

Each expert is CF-initialized but free to specialize during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class MixtureOfFIRExperts(nn.Module):
    """Multiple FIR filters with input-dependent gating."""

    def __init__(self, C_in, C_out, filter_length=7, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.inorm = nn.InstanceNorm1d(C_in, affine=True)

        # Expert FIR filters
        self.experts = nn.ModuleList([
            nn.Conv1d(C_in, C_out, filter_length, padding=filter_length // 2, bias=False)
            for _ in range(n_experts)
        ])

        # Gating network: input statistics → expert weights
        self.gate = nn.Sequential(
            nn.Linear(C_in, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts),
        )

    def forward(self, x):
        # x: (B, C_in, T)
        x = self.inorm(x)

        # Compute gating weights from channel statistics
        stats = x.mean(dim=-1)  # (B, C_in)
        weights = F.softmax(self.gate(stats), dim=-1)  # (B, n_experts)

        # Apply experts and combine
        out = torch.zeros(x.shape[0], self.experts[0].out_channels, x.shape[2],
                          device=x.device)
        for i, expert in enumerate(self.experts):
            w = weights[:, i:i+1, None]  # (B, 1, 1)
            out = out + w * expert(x)

        return out


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


def train_one_epoch_with_dropout(model, loader, loss_fn, optimizer, device,
                                  grad_clip=1.0, channel_drop_prob=0.15):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for scalp, inear in loader:
        scalp, inear = scalp.to(device), inear.to(device)
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
    # CF init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = MixtureOfFIRExperts(
        C_scalp, C_inear, filter_length=7, n_experts=4
    ).to(device)

    # Initialize all experts from CF
    with torch.no_grad():
        for expert in model.experts:
            expert.weight.zero_()
            center = expert.kernel_size[0] // 2
            expert.weight[:, :, center] = cf.W.float()
            # Add small perturbation so experts can diverge
            expert.weight += 0.01 * torch.randn_like(expert.weight)

    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_r = -1.0
    best_state = None

    for epoch in range(1, 151):
        train_one_epoch_with_dropout(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=1.0, channel_drop_prob=0.15,
        )
        val_r = validate_correlation(model, val_loader, device)
        scheduler.step()
        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model
