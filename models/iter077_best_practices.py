"""Iteration 077: Kitchen Sink of Proven Best Practices.

Combines ALL small improvements that individually add +0.001-0.003:
1. TinyDeep (H=64, 2 blocks) — proven architecture
2. CF skip connection — proven initialization
3. Mixup beta=0.4 — proven augmentation
4. Channel dropout 15% — proven augmentation
5. EMA (decay=0.999) — free stabilization
6. Target noise (sigma=0.05) — free regularization
7. Cosine LR schedule — slightly better than flat
8. Gradient clipping at 1.0 — proven
9. L1+corr loss instead of MSE+corr — REVE uses L1
10. 200 epochs with patience=40 — give it more time

Hypothesis: stacking 5-6 small gains could yield +0.005-0.015 over baseline.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Architecture: TinyDeep (H=64, 2 transformer blocks) with CF skip
# ---------------------------------------------------------------------------

class MultiScaleConv(nn.Module):
    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(h),
                nn.GELU(),
            )
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


class TinyDeep(nn.Module):
    """TinyDeep with CF-initialized skip connection."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.15):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(
            nn.Conv1d(H, H, 4, stride=4, bias=False),
            nn.BatchNorm1d(H),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=4,
            dim_feedforward=H * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.temporal(x)
        h = self.down(h).transpose(1, 2)
        h = self.transformer(h)
        h = h.transpose(1, 2)
        h = self.up(h)[:, :, : self.T]
        h = self.out_norm(h.transpose(1, 2))
        h = self.out_proj(h).transpose(1, 2)
        return h + skip


# ---------------------------------------------------------------------------
# Loss: L1 + Correlation (REVE-style)
# ---------------------------------------------------------------------------

class L1CorrLoss(nn.Module):
    """Weighted combination of L1 and negative-correlation losses."""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        pm = pred - pred.mean(-1, keepdim=True)
        tm = target - target.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
        corr_loss = 1.0 - r.mean()
        return self.alpha * l1 + (1.0 - self.alpha) * corr_loss


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict_to(self, model: nn.Module, device):
        model.load_state_dict(
            {k: v.to(device) for k, v in self.shadow.state_dict().items()}
        )


# ---------------------------------------------------------------------------
# Validation (correlation)
# ---------------------------------------------------------------------------

def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # --- Step 1: Fit closed-form baseline for skip-connection init ---
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # --- Step 2: Build TinyDeep with CF skip init ---
    model = TinyDeep(
        C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.15,
    ).to(device)

    # Initialize skip connection from CF weights
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[iter077] TinyDeep params: {n_params:,}")

    # --- Step 3: Training setup ---
    loss_fn = L1CorrLoss(alpha=0.5)
    max_epochs = 200
    patience = 40

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    # Cosine annealing LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6,
    )

    # EMA for free stabilization
    ema = EMAModel(model, decay=0.999)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True,
    )

    # --- Step 4: Training loop ---
    best_r = -1.0
    best_ema_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Mixup augmentation (beta=0.4)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]

            # Channel dropout (15%)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85

            # Target noise regularization (sigma=0.05)
            y_noisy = y + 0.05 * torch.randn_like(y)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y_noisy)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate using EMA shadow model directly (no weight swapping)
        val_r = validate_correlation(ema.shadow, val_loader, device)

        if val_r > best_r:
            best_r = val_r
            best_ema_state = {
                k: v.cpu().clone() for k, v in ema.shadow.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 25 == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"  [iter077] Epoch {epoch}: loss={epoch_loss/n_batches:.4f} "
                f"val_r={val_r:.4f} best={best_r:.4f} lr={lr:.2e}"
            )

        if no_improve >= patience:
            print(f"  [iter077] Early stopping at epoch {epoch}")
            break

    # Load best EMA checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_ema_state.items()})
    print(f"[iter077] Best val_r: {best_r:.4f}")
    return model
