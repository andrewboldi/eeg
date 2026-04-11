"""Iteration 073: Data-Centric Window Quality Pruning.

Hypothesis: Removing the worst 15% of training windows (by per-window Pearson r
from a CF baseline) will improve both CF and deep model performance. Bad windows
(artifacts, noise, NaN-heavy segments) hurt more than they help — the model wastes
capacity fitting noise instead of learning the true scalp-to-inear mapping.

This is data cleaning, not model architecture innovation.

Protocol:
1. Fit CF on full training data
2. Score each training window by Pearson r between CF prediction and target
3. Remove bottom 15% (worst quality)
4. Refit CF on pruned data
5. Train TinyDeep (H=64, 2 blocks) on pruned data
6. Validation set is unchanged for fair comparison
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ── Architecture (from iter043) ──────────────────────────────────────────────

class MultiScaleConv(nn.Module):
    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(h), nn.GELU())
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


class TinyDeep(nn.Module):
    """Tiny deep model (H=64, 2 blocks) with Flash Attention via SDPA."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                  nn.BatchNorm1d(H), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True, norm_first=True,
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


# ── Loss & validation ────────────────────────────────────────────────────────

class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


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


# ── Window quality scoring ───────────────────────────────────────────────────

def score_windows(cf_model, dataset, device, batch_size=256):
    """Compute per-window mean Pearson r using the CF model."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    window_scores = []
    cf_model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = cf_model(x)
            # Per-window, per-channel Pearson r, then mean across channels
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
            # Mean r across channels for each window
            window_scores.append(r.mean(dim=1).cpu())
    return torch.cat(window_scores).numpy()


# ── Main entry point ─────────────────────────────────────────────────────────

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    prune_fraction = 0.15

    # ── Step 1: Fit CF on full training data ──
    cf_full = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf_full.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf_full = cf_full.to(device)

    # ── Step 2: Score each training window ──
    scores = score_windows(cf_full, train_ds, device)
    n_total = len(scores)
    print(f"Window quality stats (full): mean_r={scores.mean():.4f}, "
          f"std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")

    # ── Step 3: Identify and remove bottom 15% ──
    threshold_idx = int(n_total * prune_fraction)
    sorted_indices = np.argsort(scores)
    bad_indices = set(sorted_indices[:threshold_idx].tolist())
    good_indices = [i for i in range(n_total) if i not in bad_indices]

    n_pruned = n_total - len(good_indices)
    cutoff_r = scores[sorted_indices[threshold_idx]]
    print(f"Pruning {n_pruned}/{n_total} windows ({100*prune_fraction:.0f}%) "
          f"with r < {cutoff_r:.4f}")
    print(f"Kept {len(good_indices)} windows, "
          f"mean_r={scores[good_indices].mean():.4f}")

    # ── Step 4: Build pruned dataset ──
    pruned_scalp = train_ds.scalp[good_indices]
    pruned_inear = train_ds.inear[good_indices]
    pruned_ds = EEGDataset(pruned_scalp, pruned_inear)

    # ── Step 5: Refit CF on pruned data ──
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(pruned_ds.scalp.numpy(), pruned_ds.inear.numpy())
    cf = cf.to(device)

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    cf_r = validate_correlation(cf, val_loader, device)
    print(f"CF (pruned) val_r: {cf_r:.4f}")

    # ── Step 6: Train TinyDeep on pruned data ──
    T = train_ds.scalp.shape[-1]
    deep = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1).to(device)

    # Init skip connection from pruned CF
    with torch.no_grad():
        deep.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in deep.parameters())
    print(f"TinyDeep params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(deep.parameters(), lr=3e-4, weight_decay=1e-2)
    train_loader = DataLoader(pruned_ds, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        deep.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad()
            loss = loss_fn(deep(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
            opt.step()

        vr = validate_correlation(deep, val_loader, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in deep.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    deep.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"TinyDeep best val_r: {best_r:.4f}")

    return deep
