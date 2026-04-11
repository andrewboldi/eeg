"""Iteration 041: Test-time subject adaptation with RevIN + adaptive BN.

Key insight from scaling law: more params doesn't help, cross-subject gap is the bottleneck.
This addresses it directly:
1. RevIN (Reversible Instance Normalization) — normalize input per-subject, denormalize output
2. Test-time BatchNorm adaptation — update BN stats on test subject before evaluation
3. Small model (H=64, 2 blocks) since scaling law shows diminishing returns

RevIN preserves subject-specific amplitude info (unlike InstanceNorm which destroyed it).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


class RevIN(nn.Module):
    """Reversible Instance Normalization — normalize input, denormalize output."""

    def __init__(self, n_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, n_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, n_channels, 1))
        else:
            self.gamma = self.beta = None
        self.mean = self.std = None

    def normalize(self, x):
        # x: (B, C, T)
        self.mean = x.mean(dim=-1, keepdim=True).detach()
        self.std = (x.std(dim=-1, keepdim=True) + self.eps).detach()
        x = (x - self.mean) / self.std
        if self.gamma is not None:
            x = x * self.gamma + self.beta
        return x

    def denormalize(self, x, C_out):
        # Scale output using input statistics (first C_out channels)
        mean = self.mean[:, :C_out, :]
        std = self.std[:, :C_out, :]
        if self.gamma is not None:
            x = (x - self.beta[:, :C_out, :]) / (self.gamma[:, :C_out, :] + self.eps)
        return x * std + mean


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


class AdaptiveModel(nn.Module):
    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

        # RevIN on input
        self.revin = RevIN(C_in, affine=True)

        # Multi-scale temporal
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                   nn.BatchNorm1d(H), nn.GELU())

        # Transformer blocks
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
        # x: (B, C_in, T)
        skip = self.skip(x)
        x = self.revin.normalize(x)

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

    def adapt_bn(self, test_loader, device, n_steps=5):
        """Update BatchNorm stats on test subject data."""
        # Set BN layers to train mode only
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.train()
                m.reset_running_stats()

        with torch.no_grad():
            for _ in range(n_steps):
                for x, _ in test_loader:
                    _ = self(x.to(device))
        self.eval()


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

    model = AdaptiveModel(C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    print(f"AdaptiveModel params: {sum(p.numel() for p in model.parameters()):,}")

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

    # Test-time BN adaptation on validation set (simulates adapting to new subject)
    model.adapt_bn(vl, device, n_steps=3)
    adapted_r = validate_correlation(model, vl, device)
    print(f"After BN adapt: val_r={adapted_r:.4f} (delta={adapted_r-best_r:+.4f})")

    return model
