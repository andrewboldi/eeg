"""Iteration 065: High-pass 3 Hz filtering on input scalp EEG.

Hypothesis: Higher high-pass cutoff (3 Hz instead of 1 Hz) removes slow drifts
that are subject-specific noise, improving cross-subject generalization.
The broadband data is 1-45 Hz at 128 Hz. We re-filter scalp inputs to 3-45 Hz
before training, leaving in-ear targets unchanged.

This is a preprocessing change only — architecture is standard TinyDeep (H=64, 2 blocks)
with CF skip connection.

Confidence: 70% — denoising literature suggests 2-3 Hz high-pass helps decoding.
Risk: may remove useful low-frequency signal.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------- Filter utility ----------

def apply_highpass_to_scalp(ds: EEGDataset, fs: float = 128.0, lo: float = 3.0, hi: float = 45.0) -> EEGDataset:
    """Apply 3-45 Hz bandpass to scalp channels of an EEGDataset.

    Operates on the numpy level, returns a new EEGDataset with filtered scalp
    and the original in-ear targets.
    """
    scalp_np = ds.scalp.numpy().copy()  # (N, C, T)
    N, C, T = scalp_np.shape

    # Design 4th-order Butterworth bandpass
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")

    # Filter each channel of each window
    # Reshape to (N*C, T) for vectorized filtering, then reshape back
    flat = scalp_np.reshape(N * C, T)
    # filtfilt needs axis=-1 (last axis) which is default
    # Use padlen to handle short segments gracefully
    padlen = min(3 * max(len(b), len(a)), T - 1)
    filtered = filtfilt(b, a, flat, axis=-1, padlen=padlen).astype(np.float32)
    scalp_filtered = filtered.reshape(N, C, T)

    return EEGDataset(scalp=scalp_filtered, inear=ds.inear.numpy())


# ---------- Architecture (same as iter043 TinyDeep) ----------

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


# ---------- Loss + validation ----------

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


# ---------- Main entry point ----------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    """Build and train TinyDeep on 3-45 Hz filtered scalp data."""

    # Step 0: Apply 3 Hz high-pass to scalp inputs only
    print("Applying 3-45 Hz bandpass filter to scalp inputs...")
    train_ds = apply_highpass_to_scalp(train_ds, fs=128.0, lo=3.0, hi=45.0)
    val_ds = apply_highpass_to_scalp(val_ds, fs=128.0, lo=3.0, hi=45.0)
    print(f"  Filtered train: {train_ds.scalp.shape}, val: {val_ds.scalp.shape}")

    # Step 1: Fit CF on filtered data
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)

    # Step 2: Train TinyDeep with CF skip init
    T = train_ds.scalp.shape[-1]
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeep params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
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
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f} (3-45 Hz high-pass on scalp inputs)")

    return model
