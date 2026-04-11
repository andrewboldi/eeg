"""Iteration 072: Multi-Seed Ensemble of TinyDeep Models.

Train 3 TinyDeep models with different random seeds, then average predictions
at test time. Same architecture, different initialization and mixup randomness.

This reduces variance without changing bias. The earlier ensemble (iter043)
failed because it mixed CF + deep model (different biases). Same-architecture
ensemble should work better since all models have similar bias but decorrelated errors.

Confidence: 70% — simple ensembles usually help, but 3 models may not be enough
diversity for a meaningful gain on this small dataset.
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


class TinyDeep(nn.Module):
    """Tiny deep model (55K params) with Flash Attention via SDPA."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                   nn.BatchNorm1d(H), nn.GELU())
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


class EnsembleModule(nn.Module):
    """Ensemble of N models — averages predictions at test time."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        preds = [m(x) for m in self.models]
        return torch.stack(preds).mean(dim=0)


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


def _train_one_model(seed, train_ds, val_ds, C_scalp, C_inear, device, cf_weight):
    """Train a single TinyDeep model with the given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.1).to(device)

    # Initialize skip connection from CF
    with torch.no_grad():
        model.skip.weight.copy_(cf_weight.float().unsqueeze(-1))

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            # Mixup with seed-dependent randomness
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout
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
            print(f"    Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r


def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit CF for skip-connection initialization
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf_weight = cf.W.clone()

    # Step 2: Train 3 TinyDeep models with different seeds
    seeds = [42, 123, 456]
    models = []
    for i, seed in enumerate(seeds):
        print(f"  Training model {i+1}/3 (seed={seed})")
        model, best_r = _train_one_model(
            seed, train_ds, val_ds, C_scalp, C_inear, device, cf_weight
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model {i+1} done: val_r={best_r:.4f}, params={n_params:,}")
        models.append(model)

    # Step 3: Build ensemble
    ensemble = EnsembleModule(models).to(device)
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"Ensemble total params: {total_params:,}")

    # Validate ensemble
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    ensemble_r = validate_correlation(ensemble, vl, device)
    print(f"Ensemble val_r: {ensemble_r:.4f}")

    return ensemble
