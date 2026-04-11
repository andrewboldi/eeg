"""Iteration 045: Long Context (4s windows at 128Hz).

Adapts the proven TinyDeep architecture for 4-second windows (512 samples at 128Hz).
Longer temporal context should capture slow EEG dynamics (delta/theta rhythms).
Adds kernel size 63 to multi-scale conv for slow oscillations.
Uses stride-8 downsampling (512->64 tokens) to keep transformer sequence length manageable.

Confidence: 70% — longer context helps if slow dynamics matter, but may dilute signal.
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
    def __init__(self, C_in, H, kernels=(3, 7, 15, 31, 63)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(h), nn.GELU())
            for k in kernels
        ])
        # If H not evenly divisible, pad last conv to make up the difference
        self.pad_h = H - h * len(kernels)
        if self.pad_h > 0:
            self.extra = nn.Sequential(
                nn.Conv1d(C_in, self.pad_h, 3, padding=1, bias=False),
                nn.BatchNorm1d(self.pad_h), nn.GELU())

    def forward(self, x):
        outs = [c(x) for c in self.convs]
        if self.pad_h > 0:
            outs.append(self.extra(x))
        return torch.cat(outs, dim=1)


class TinyDeepLongContext(nn.Module):
    """Tiny deep model adapted for 4s windows (512 samples at 128Hz).

    Uses stride-8 downsampling (512->64 tokens) and kernel 63 for slow dynamics.
    ~60K params with Flash Attention via SDPA.
    """

    def __init__(self, C_in, C_out, T=512, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H, kernels=(3, 7, 15, 31, 63))
        # Stride 8 downsampling: 512 -> 64 tokens
        self.down = nn.Sequential(nn.Conv1d(H, H, 8, stride=8, bias=False),
                                  nn.BatchNorm1d(H), nn.GELU())
        # Transformer with SDPA (Flash Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        # Stride 8 upsampling: 64 -> 512
        self.up = nn.ConvTranspose1d(H, H, 8, stride=8, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)
        # CF skip connection
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.temporal(x)
        h = self.down(h).transpose(1, 2)  # (B, 64, H)
        h = self.transformer(h)
        h = h.transpose(1, 2)  # (B, H, 64)
        h = self.up(h)[:, :, :self.T]  # (B, H, 512)
        h = self.out_norm(h.transpose(1, 2))  # (B, 512, H)
        h = self.out_proj(h).transpose(1, 2)  # (B, C_out, 512)
        return h + skip


class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm, tm = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm, tm = p - p.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    print("NOTE: iter045_long_context requires broadband_46ch_4s.h5 data (4s windows, 512 samples at 128Hz).")
    print("      The default benchmark loads T=256 (2s windows). Run with a modified script pointing to 4s data.")

    # Detect T from data
    T = train_ds.scalp.shape[-1]
    print(f"  Detected T={T} from dataset (expected 512 for 4s at 128Hz)")

    # Step 1: Fit CF baseline for skip connection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)

    # Step 2: Build and train deep model
    model = TinyDeepLongContext(
        C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1
    ).to(device)

    # Init skip connection from CF spatial filter
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeepLongContext params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            # Mixup augmentation (beta=0.4)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout (15%)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vr = validate_correlation(model, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stopping at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")

    return model
