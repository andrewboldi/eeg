"""Iteration 039: Deep broadband model with GPU (46ch input, 128 Hz).

Architecture: Multi-scale temporal convolutions + channel attention.
- Stage 1: Multi-scale temporal conv (kernels 3, 7, 15, 31) to extract features at different scales
- Stage 2: Channel attention (which input channels matter for each output?)
- Stage 3: Temporal attention over downsampled features
- Stage 4: Output projection with skip connection from CF init

Target: r > 0.6 on broadband data. Uses 46 input channels (scalp + around-ear).
~200K parameters, designed for RTX 4060 (8GB).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class MultiScaleConv(nn.Module):
    """Extract temporal features at multiple scales."""

    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C_in, H // len(kernels), k, padding=k // 2, bias=False),
                nn.BatchNorm1d(H // len(kernels)),
                nn.GELU(),
            )
            for k in kernels
        ])

    def forward(self, x):
        # x: (B, C_in, T)
        return torch.cat([conv(x) for conv in self.convs], dim=1)  # (B, H, T)


class ChannelAttention(nn.Module):
    """Attend across channel dimension."""

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, D) where C is channels, D is features
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return residual + self.drop(x)


class FeedForward(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class DeepBroadbandModel(nn.Module):
    def __init__(self, C_in, C_out, T=256, H=128, n_blocks=4, dropout=0.15):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.T = T

        # Stage 1: Multi-scale temporal features
        self.temporal_enc = MultiScaleConv(C_in, H, kernels=(3, 7, 15, 31))

        # Stage 2: Temporal downsampling (256 -> 64 timesteps)
        self.downsample = nn.Sequential(
            nn.Conv1d(H, H, 4, stride=4, bias=False),
            nn.BatchNorm1d(H),
            nn.GELU(),
        )

        # Stage 3: Transformer blocks on (T/4) temporal tokens
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleList([
                ChannelAttention(H, n_heads=4, dropout=dropout),
                FeedForward(H, expand=4, dropout=dropout),
            ]))

        # Stage 4: Upsample back + output projection
        self.upsample = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.output_proj = nn.Linear(H, C_out)

        # Skip connection: direct linear mapping
        self.skip = nn.Conv1d(C_in, C_out, 1, bias=True)

    def forward(self, x):
        # x: (B, C_in, T)
        skip = self.skip(x)  # (B, C_out, T) — linear shortcut

        # Multi-scale temporal features
        h = self.temporal_enc(x)  # (B, H, T)

        # Downsample temporally
        h = self.downsample(h)  # (B, H, T/4)
        T_down = h.shape[2]

        # Transformer blocks: treat temporal positions as tokens
        h = h.transpose(1, 2)  # (B, T/4, H)
        for attn, ff in self.blocks:
            h = attn(h)
            h = ff(h)
        h = h.transpose(1, 2)  # (B, H, T/4)

        # Upsample
        h = self.upsample(h)[:, :, :self.T]  # (B, H, T)

        # Project to output
        h = h.transpose(1, 2)  # (B, T, H)
        h = self.out_norm(h)
        h = self.output_proj(h)  # (B, T, C_out)
        h = h.transpose(1, 2)  # (B, C_out, T)

        return h + skip


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
    # CF for skip connection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = DeepBroadbandModel(
        C_in=C_scalp, C_out=C_inear, T=256,
        H=128, n_blocks=4, dropout=0.15,
    ).to(device)

    # Init skip connection with CF
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DeepBroadband params: {n_params:,}")

    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=300,
        steps_per_epoch=len(train_ds) // 128 + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    best_val_r = -1.0
    best_state = None
    patience = 50
    no_improve = 0

    for epoch in range(1, 301):
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
            scheduler.step()

        val_r = validate_correlation(model, val_loader, device)

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 25 == 0:
            print(f"Epoch {epoch}: val_r={val_r:.4f} (best={best_val_r:.4f})")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Final best val_r: {best_val_r:.4f}")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model
