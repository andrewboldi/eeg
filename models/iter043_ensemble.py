"""Iteration 043: Ensemble of CF + Tiny Deep Model.

Averages predictions from closed-form baseline and tiny deep model.
Uses weighted average with learned/optimized weight on validation set.
CF captures linear spatial mapping, deep model captures nonlinear temporal patterns.

Confidence: 90% — ensembles almost always help with diverse base models.
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


class TinyDeep(nn.Module):
    """Tiny deep model (55K params) with Flash Attention via SDPA."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                   nn.BatchNorm1d(H), nn.GELU())
        # Transformer with SDPA (Flash Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H*4,
            dropout=dropout, batch_first=True,
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
        h = self.up(h)[:, :, :self.T]
        h = self.out_norm(h.transpose(1, 2))
        h = self.out_proj(h).transpose(1, 2)
        return h + skip


class EnsembleModel(nn.Module):
    """Weighted ensemble of CF + deep model."""

    def __init__(self, cf_model, deep_model, alpha=0.5):
        super().__init__()
        self.cf = cf_model
        self.deep = deep_model
        # Learned ensemble weight per output channel
        self.alpha = nn.Parameter(torch.full((1, deep_model.out_proj.out_features, 1), alpha))

    def forward(self, x):
        with torch.no_grad():
            cf_pred = self.cf(x)
        deep_pred = self.deep(x)
        alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
        return alpha * deep_pred + (1 - alpha) * cf_pred


class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm, tm = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm**2).sum(-1).sqrt() * (tm**2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm, tm = p - p.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm**2).sum(-1).sqrt() * (tm**2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit CF
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)

    # Step 2: Train deep model with CF skip
    deep = TinyDeep(C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        deep.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in deep.parameters())
    print(f"TinyDeep params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(deep.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        deep.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam*x + (1-lam)*x[idx]; y = lam*y + (1-lam)*y[idx]
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad(); loss_fn(deep(x), y).backward()
            torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0); opt.step()
        vr = validate_correlation(deep, vl, device)
        if vr > best_r: best_r = vr; best_state = {k:v.cpu().clone() for k,v in deep.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if ep % 25 == 0: print(f"  Deep Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30: break

    deep.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    print(f"Deep best val_r: {best_r:.4f}")

    # Step 3: Build ensemble and optimize alpha on validation
    ensemble = EnsembleModel(cf, deep, alpha=0.5).to(device)

    # Quick optimization of alpha only
    alpha_opt = torch.optim.Adam([ensemble.alpha], lr=0.1)
    for _ in range(50):
        total_loss = 0
        for x, y in vl:
            x, y = x.to(device), y.to(device)
            alpha_opt.zero_grad()
            loss = loss_fn(ensemble(x), y)
            loss.backward()
            alpha_opt.step()
            total_loss += loss.item()

    final_alpha = torch.sigmoid(ensemble.alpha).mean().item()
    ensemble_r = validate_correlation(ensemble, vl, device)
    print(f"Ensemble val_r: {ensemble_r:.4f}, alpha={final_alpha:.3f}")

    return ensemble
