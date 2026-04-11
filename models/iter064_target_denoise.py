"""Iteration 064: Target-side denoising for cleaner training labels.

Hypothesis: In-ear EEG channels suffer from poor electrode contact, especially
on hard subjects (S14). Denoising only the TARGET (in-ear) channels before
training gives the model cleaner labels to learn from, without corrupting input.

Approach:
  1. Moving median filter (window=5) on each in-ear channel to remove spikes
  2. PCA denoising keeping components explaining 95% variance
  3. Train TinyDeep (H=64, 2 blocks) on denoised targets with original inputs
  4. Evaluate against original (noisy) test targets for fair comparison

Confidence: 60% — denoising targets is a known trick in robust regression,
but may remove genuine signal in narrowband 1-9 Hz data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Target denoising utilities
# ---------------------------------------------------------------------------

def moving_median_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving median filter along the last axis (time).

    Args:
        x: array of shape (N, C, T)
        window: median filter window size (must be odd)

    Returns:
        Filtered array of same shape.
    """
    from scipy.ndimage import median_filter
    # median_filter operates element-wise; size=(1,1,window) filters along T
    return median_filter(x, size=(1, 1, window), mode="reflect")


def pca_denoise(x: np.ndarray, var_threshold: float = 0.95) -> np.ndarray:
    """PCA-based denoising: keep components explaining var_threshold of variance.

    Operates across the channel dimension for each window independently.

    Args:
        x: array of shape (N, C, T)
        var_threshold: fraction of variance to retain

    Returns:
        Denoised array of same shape.
    """
    N, C, T = x.shape
    # Reshape to (N, C, T) -> work per-window
    out = np.empty_like(x)
    for i in range(N):
        # (C, T) matrix — channels are variables, time points are observations
        window = x[i]  # (C, T)
        mean = window.mean(axis=1, keepdims=True)  # (C, 1)
        centered = window - mean  # (C, T)

        # SVD: centered = U @ diag(s) @ Vt
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        var_explained = np.cumsum(s ** 2) / (np.sum(s ** 2) + 1e-12)
        k = int(np.searchsorted(var_explained, var_threshold) + 1)
        k = max(1, min(k, C))  # at least 1, at most C

        # Reconstruct with top-k components
        reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        out[i] = reconstructed + mean

    return out


def denoise_targets(inear: np.ndarray, median_window: int = 5,
                    pca_var: float = 0.95) -> np.ndarray:
    """Full target denoising pipeline.

    Args:
        inear: (N, C_inear, T) in-ear EEG data
        median_window: window for spike removal
        pca_var: variance threshold for PCA denoising

    Returns:
        Denoised in-ear data of same shape.
    """
    print(f"  Denoising targets: median(w={median_window}) + PCA({pca_var:.0%} var)")
    # Step 1: Remove spikes with moving median
    denoised = moving_median_filter(inear, window=median_window)

    # Step 2: PCA denoising to remove low-variance noise components
    denoised = pca_denoise(denoised, var_threshold=pca_var)

    # Report how much changed
    delta = np.abs(denoised - inear)
    print(f"  Mean abs change: {delta.mean():.6f}, Max: {delta.max():.6f}")
    corr_before_after = np.corrcoef(inear.ravel(), denoised.ravel())[0, 1]
    print(f"  Correlation original vs denoised targets: {corr_before_after:.4f}")

    return denoised.astype(np.float32)


# ---------------------------------------------------------------------------
# TinyDeep architecture (from iter043)
# ---------------------------------------------------------------------------

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
    """Tiny deep model (55K params) with Flash Attention via SDPA."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                  nn.BatchNorm1d(H), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
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


# ---------------------------------------------------------------------------
# Loss and validation
# ---------------------------------------------------------------------------

class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
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
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    """Train TinyDeep on denoised targets, evaluate on original targets."""

    # ------------------------------------------------------------------
    # Step 1: Denoise in-ear targets for training
    # ------------------------------------------------------------------
    train_inear_np = train_ds.inear.numpy()
    denoised_train = denoise_targets(train_inear_np, median_window=5, pca_var=0.95)

    # Build training dataset with denoised targets
    train_scalp_t = train_ds.scalp  # original scalp input
    train_inear_denoised_t = torch.from_numpy(denoised_train)

    # Validation uses ORIGINAL targets (fair evaluation during training)
    val_scalp_t = val_ds.scalp
    val_inear_t = val_ds.inear

    # ------------------------------------------------------------------
    # Step 2: Fit closed-form baseline on denoised targets for CF init
    # ------------------------------------------------------------------
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), denoised_train)
    cf = cf.to(device)

    # Validate CF on original targets
    cf_loader = DataLoader(
        TensorDataset(val_scalp_t, val_inear_t),
        batch_size=128, shuffle=False,
    )
    cf_r = validate_correlation(cf, cf_loader, device)
    print(f"  CF on denoised targets -> val_r (original targets): {cf_r:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Train TinyDeep with CF skip init
    # ------------------------------------------------------------------
    T = train_ds.scalp.shape[-1]
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64,
                     n_blocks=2, dropout=0.1).to(device)

    # Initialize skip connection from CF weights
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TinyDeep params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    # Training loader uses denoised targets
    train_loader = DataLoader(
        TensorDataset(train_scalp_t, train_inear_denoised_t),
        batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
    )
    # Validation loader uses original targets for fair comparison
    val_loader = DataLoader(
        TensorDataset(val_scalp_t, val_inear_t),
        batch_size=128, shuffle=False, num_workers=2, pin_memory=True,
    )

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
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
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validate on original targets
        vr = validate_correlation(model, val_loader, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Best val_r (original targets): {best_r:.4f}")

    return model
