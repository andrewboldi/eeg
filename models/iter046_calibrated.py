"""Iteration 046: Test-Time Output Calibration.

After training a TinyDeep model normally, fit a per-channel affine transform
(scale + offset) on the validation set to calibrate outputs. This is different
from RevIN (input normalization) -- this calibrates OUTPUTS to better match
each subject's amplitude/offset characteristics.

Two variants:
1. Supervised calibration: fits alpha[ch]*pred[ch]+beta[ch] via least-squares
   on validation set (which has ground truth labels).
2. Unsupervised calibration: matches output mean/std to training set target
   statistics (no labels needed at calibration time).

We return the supervised variant by default as it should perform better,
but both are implemented for comparison.

Hypothesis: The tiny deep model's predictions may have systematic per-channel
bias/scale errors that a simple affine correction can fix, especially since
different in-ear channels have very different amplitude ranges.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Architecture (copied from iter043)
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
# Calibration wrappers
# ---------------------------------------------------------------------------

class CalibratedModel(nn.Module):
    """Wraps a base model and applies per-channel affine calibration.

    output_calibrated[ch] = alpha[ch] * output_raw[ch] + beta[ch]

    alpha and beta are fit via least-squares on a calibration set (supervised).
    """

    def __init__(self, base_model: nn.Module, C_out: int):
        super().__init__()
        self.base = base_model
        # Initialize to identity transform
        self.register_buffer("alpha", torch.ones(1, C_out, 1))
        self.register_buffer("beta", torch.zeros(1, C_out, 1))

    def calibrate(self, loader: DataLoader, device: torch.device):
        """Fit per-channel alpha, beta via least-squares on (pred, target) pairs.

        For each channel ch, we solve:
            target[ch] = alpha[ch] * pred[ch] + beta[ch]
        using ordinary least squares over all samples and time steps.
        """
        self.base.eval()
        all_pred, all_target = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                p = self.base(x)
                all_pred.append(p.cpu())
                all_target.append(y.cpu())

        pred = torch.cat(all_pred, dim=0)    # (N, C_out, T)
        target = torch.cat(all_target, dim=0)  # (N, C_out, T)
        C_out = pred.shape[1]

        alpha = torch.ones(C_out)
        beta = torch.zeros(C_out)

        for ch in range(C_out):
            # Flatten all predictions and targets for this channel
            p_flat = pred[:, ch, :].reshape(-1).float()
            t_flat = target[:, ch, :].reshape(-1).float()

            # Remove NaN values
            valid = ~(torch.isnan(p_flat) | torch.isnan(t_flat))
            p_flat = p_flat[valid]
            t_flat = t_flat[valid]

            if len(p_flat) < 10:
                continue  # Keep identity for degenerate channels

            # Least-squares: [alpha, beta] = (A^T A)^{-1} A^T t
            # where A = [p, 1]
            A = torch.stack([p_flat, torch.ones_like(p_flat)], dim=1)
            # Use lstsq for numerical stability
            result = torch.linalg.lstsq(A, t_flat)
            coeff = result.solution
            alpha[ch] = coeff[0]
            beta[ch] = coeff[1]

        self.alpha.copy_(alpha.reshape(1, C_out, 1))
        self.beta.copy_(beta.reshape(1, C_out, 1))

        print(f"  Calibration alpha: mean={alpha.mean():.4f}, std={alpha.std():.4f}")
        print(f"  Calibration beta:  mean={beta.mean():.6f}, std={beta.std():.6f}")

    def forward(self, x):
        raw = self.base(x)
        return self.alpha * raw + self.beta


class UnsupervisedCalibratedModel(nn.Module):
    """Wraps a base model and matches output statistics to training targets.

    No labels needed at calibration time. Simply forces model outputs to have
    the same per-channel mean and std as the training set targets.

    output_calibrated[ch] = (output_raw[ch] - mu_pred[ch]) / sigma_pred[ch]
                             * sigma_train[ch] + mu_train[ch]
    """

    def __init__(self, base_model: nn.Module, C_out: int):
        super().__init__()
        self.base = base_model
        self.register_buffer("alpha", torch.ones(1, C_out, 1))
        self.register_buffer("beta", torch.zeros(1, C_out, 1))

    def calibrate_from_stats(
        self,
        train_target_mean: torch.Tensor,   # (C_out,)
        train_target_std: torch.Tensor,     # (C_out,)
        pred_loader: DataLoader,
        device: torch.device,
    ):
        """Match output distribution to training target distribution.

        1. Compute mean/std of model predictions on the calibration input set.
        2. Set alpha/beta so output has same mean/std as training targets.
        """
        self.base.eval()
        all_pred = []
        with torch.no_grad():
            for x, _ in pred_loader:
                x = x.to(device)
                p = self.base(x)
                all_pred.append(p.cpu())

        pred = torch.cat(all_pred, dim=0)  # (N, C_out, T)
        C_out = pred.shape[1]

        alpha = torch.ones(C_out)
        beta = torch.zeros(C_out)

        for ch in range(C_out):
            p_flat = pred[:, ch, :].reshape(-1).float()
            valid = ~torch.isnan(p_flat)
            p_flat = p_flat[valid]

            if len(p_flat) < 10:
                continue

            mu_pred = p_flat.mean()
            sigma_pred = p_flat.std()

            if sigma_pred < 1e-8:
                continue  # Degenerate channel

            mu_train = train_target_mean[ch]
            sigma_train = train_target_std[ch]

            # output = alpha * raw + beta
            # We want: alpha * mu_pred + beta = mu_train
            #          alpha * sigma_pred = sigma_train
            alpha[ch] = sigma_train / sigma_pred
            beta[ch] = mu_train - alpha[ch] * mu_pred

        self.alpha.copy_(alpha.reshape(1, C_out, 1))
        self.beta.copy_(beta.reshape(1, C_out, 1))

        print(f"  Unsupervised alpha: mean={alpha.mean():.4f}, std={alpha.std():.4f}")
        print(f"  Unsupervised beta:  mean={beta.mean():.6f}, std={beta.std():.6f}")

    def forward(self, x):
        raw = self.base(x)
        return self.alpha * raw + self.beta


# ---------------------------------------------------------------------------
# Loss and validation (same as iter043)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # --- Step 1: Fit closed-form for skip-connection init ---
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)

    # --- Step 2: Train TinyDeep with CF skip init ---
    T = train_ds.scalp.shape[2]
    deep = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1).to(device)
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
            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad()
            loss_fn(deep(x), y).backward()
            torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
            opt.step()
        vr = validate_correlation(deep, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in deep.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 25 == 0:
            print(f"  Deep Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            break

    deep.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Deep best val_r: {best_r:.4f}")

    # --- Step 3: Supervised calibration on validation set ---
    calibrated = CalibratedModel(deep, C_out=C_inear).to(device)
    calibrated.calibrate(vl, device)

    # Evaluate calibrated vs uncalibrated on val set
    raw_r = validate_correlation(deep, vl, device)
    cal_r = validate_correlation(calibrated, vl, device)
    print(f"Val r -- raw: {raw_r:.4f}, supervised calibrated: {cal_r:.4f}")

    # --- Step 4: Also try unsupervised calibration for comparison ---
    # Compute training target statistics
    train_target = train_ds.inear  # (N, C_out, T)
    train_mean = torch.zeros(C_inear)
    train_std = torch.ones(C_inear)
    for ch in range(C_inear):
        ch_data = train_target[:, ch, :].reshape(-1).float()
        valid = ~torch.isnan(ch_data)
        ch_data = ch_data[valid]
        if len(ch_data) > 10:
            train_mean[ch] = ch_data.mean()
            train_std[ch] = ch_data.std()

    unsup = UnsupervisedCalibratedModel(deep, C_out=C_inear).to(device)
    unsup.calibrate_from_stats(train_mean, train_std, vl, device)
    unsup_r = validate_correlation(unsup, vl, device)
    print(f"Val r -- unsupervised calibrated: {unsup_r:.4f}")

    # --- Step 5: Return the best variant ---
    # Pick whichever calibration (or none) gives best val correlation
    candidates = {
        "raw": (deep, raw_r),
        "supervised": (calibrated, cal_r),
        "unsupervised": (unsup, unsup_r),
    }
    best_name = max(candidates, key=lambda k: candidates[k][1])
    best_model = candidates[best_name][0]
    print(f"Selected variant: {best_name} (val_r={candidates[best_name][1]:.4f})")

    return best_model
