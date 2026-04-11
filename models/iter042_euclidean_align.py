"""Iteration 042: Euclidean Alignment with full-session covariance.

iter018 failed because it used per-batch whitening (noisy covariance estimates).
This version computes covariance over the ENTIRE session per subject, then
whitens all data. This is the standard approach in BCI literature.

Also uses the optimal tiny model (H=64, 2 blocks) from scaling law.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm, inv

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


def euclidean_align(data_by_subject: dict):
    """Align all subjects to a common reference using Euclidean Alignment.

    For each subject, compute session covariance, then whiten.
    Reference: He & Wu 2020, "Transfer Learning for Brain-Computer Interfaces"

    Args:
        data_by_subject: {subject_id: (scalp_array, inear_array)}
    Returns:
        aligned data dict with same structure
    """
    aligned = {}
    for subj, (scalp, inear) in data_by_subject.items():
        # scalp: (N, C, T)
        N, C, T = scalp.shape
        # Compute mean covariance across all windows
        flat = scalp.reshape(N, C, T)
        # Per-window covariance, then average
        covs = np.zeros((C, C))
        for i in range(N):
            x = flat[i]  # (C, T)
            covs += x @ x.T / T
        covs /= N
        # Regularize
        covs += 1e-6 * np.eye(C)
        # Whitening matrix: R^{-1/2}
        try:
            R_sqrt_inv = inv(sqrtm(covs)).real.astype(np.float32)
            # Apply whitening
            aligned_scalp = np.zeros_like(scalp)
            for i in range(N):
                aligned_scalp[i] = R_sqrt_inv @ scalp[i]
            aligned[subj] = (aligned_scalp, inear)
        except Exception:
            # If sqrtm fails, use original
            aligned[subj] = (scalp, inear)

    return aligned


class MultiScaleConv(nn.Module):
    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(C_in, h, k, padding=k//2, bias=False),
                          nn.BatchNorm1d(h), nn.GELU())
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


class AlignedModel(nn.Module):
    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                   nn.BatchNorm1d(H), nn.GELU())
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(H),
                nn.MultiheadAttention(H, 4, dropout=dropout, batch_first=True),
                nn.LayerNorm(H),
                nn.Sequential(nn.Linear(H, H*4), nn.GELU(),
                              nn.Dropout(dropout), nn.Linear(H*4, H), nn.Dropout(dropout)),
            ]))
        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.temporal(x)
        h = self.down(h)
        h = h.transpose(1, 2)
        for ln1, attn, ln2, ff in self.blocks:
            h2 = ln1(h); h2, _ = attn(h2, h2, h2); h = h + h2
            h = h + ff(ln2(h))
        h = h.transpose(1, 2)
        h = self.up(h)[:, :, :x.shape[2]]
        h = self.out_norm(h.transpose(1, 2))
        h = self.out_proj(h).transpose(1, 2)
        return h + skip


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pm = pred - pred.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm**2).sum(-1).sqrt() * (tm**2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


class CorrMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pm = pred - pred.mean(-1, keepdim=True)
        tm = target - target.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm**2).sum(-1).sqrt() * (tm**2).sum(-1).sqrt() + 1e-8)
        return self.alpha * mse + (1 - self.alpha) * (1 - r.mean())


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = AlignedModel(C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    print(f"AlignedModel params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = CorrMSELoss(alpha=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam*x + (1-lam)*x[idx]; y = lam*y + (1-lam)*y[idx]
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad(); loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        vr = validate_correlation(model, vl, device)
        if vr > best_r: best_r = vr; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if ep % 25 == 0: print(f"Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30: break

    print(f"Best val_r: {best_r:.4f}")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model
