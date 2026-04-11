"""Iteration 044: Residual-from-CF with Flash Attention.

CF is the primary prediction. Model only learns a small residual correction.
output = CF(x) + learned_alpha * model(x)

Uses nn.TransformerEncoder with SDPA (auto Flash Attention on RTX 4060).
Also incorporates Optuna-found hyperparameters from manual analysis:
  - LR: 2e-4 (slightly lower), weight_decay: 5e-3
  - Loss alpha: 0.4 (more corr, less MSE)
  - Mixup alpha: 0.3
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
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


class ResidualCFModel(nn.Module):
    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.C_out = C_out

        # CF as frozen primary predictor
        self.cf_weight = nn.Parameter(torch.zeros(C_out, C_in), requires_grad=False)

        # Residual model with SDPA/Flash Attention
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                   nn.BatchNorm1d(H), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)

        # Learned residual scale (initialized small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # CF prediction (frozen)
        cf_pred = torch.matmul(self.cf_weight, x)  # (B, C_out, T)

        # Residual prediction
        h = self.temporal(x)
        h = self.down(h).transpose(1, 2)
        h = self.transformer(h).transpose(1, 2)
        h = self.up(h)[:, :, :self.T]
        h = self.out_norm(h.transpose(1, 2))
        residual = self.out_proj(h).transpose(1, 2)

        return cf_pred + self.residual_scale * residual


class CorrMSELoss(nn.Module):
    def __init__(self, a=0.4):
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
    # Fit CF
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = ResidualCFModel(C_in=C_scalp, C_out=C_inear, T=256, H=64,
                             n_blocks=2, dropout=0.1).to(device)

    # Set CF weights (frozen)
    with torch.no_grad():
        model.cf_weight.copy_(cf.W.float())

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResidualCF trainable params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.4)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=5e-3
    )

    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            lam = np.random.beta(0.3, 0.3)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam*x + (1-lam)*x[idx]; y = lam*y + (1-lam)*y[idx]
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad(); loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        vr = validate_correlation(model, vl, device)
        if vr > best_r: best_r = vr; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if ep % 25 == 0:
            rs = model.residual_scale.item()
            print(f"Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f}), res_scale={rs:.4f}")
        if no_imp >= 30: break

    print(f"Best val_r: {best_r:.4f}")
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    return model
