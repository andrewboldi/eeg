"""Iteration 053: Spectral Loss — Combined time + frequency domain loss.

Based on FreDF (ICLR 2025): computing loss in FFT domain reduces estimation bias
and enforces spectral consistency. Uses TinyDeep (H=64, 2 blocks) from iter043.

Loss: CorrMSESpectralLoss = alpha*MSE + beta*(1 - corr) + gamma*spectral_L1
Plus multi-resolution STFT loss variant for magnitude + phase consistency.

Hypothesis: Frequency-domain loss will enforce spectral fidelity that time-domain
losses miss, especially for the narrow 1-9 Hz band where spectral shape matters.
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
# Architecture: TinyDeep from iter043
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
# Loss: CorrMSESpectralLoss + Multi-Resolution STFT
# ---------------------------------------------------------------------------

class CorrMSESpectralLoss(nn.Module):
    """Combined time-domain + frequency-domain loss.

    L = alpha * L_mse + beta * L_corr + gamma * L_spectral

    L_spectral = mean(|rfft(pred) - rfft(target)|)  (complex L1 in freq domain)
    """

    def __init__(self, alpha=0.3, beta=0.3, gamma=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        # Time-domain MSE
        mse = F.mse_loss(pred, target)

        # Pearson correlation loss
        pm = pred - pred.mean(dim=-1, keepdim=True)
        tm = target - target.mean(dim=-1, keepdim=True)
        num = (pm * tm).sum(dim=-1)
        den = pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8
        corr_loss = 1.0 - (num / den).mean()

        # Spectral loss: complex L1 in frequency domain
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        spectral_loss = torch.abs(pred_fft - target_fft).mean()

        return self.alpha * mse + self.beta * corr_loss + self.gamma * spectral_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss over magnitude and phase.

    Computes sum over fft_sizes of:
        mean(|STFT_mag(pred) - STFT_mag(target)|)
      + mean(|STFT_phase(pred) - STFT_phase(target)|)
    """

    def __init__(self, fft_sizes=(64, 128, 256)):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, pred, target):
        # Flatten batch and channel dims: (B, C, T) -> (B*C, T)
        B, C, T = pred.shape
        pred_flat = pred.reshape(B * C, T)
        target_flat = target.reshape(B * C, T)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for n_fft in self.fft_sizes:
            if n_fft > T:
                continue
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)

            pred_stft = torch.stft(pred_flat, n_fft=n_fft, hop_length=hop,
                                   win_length=n_fft, window=window,
                                   return_complex=True)
            target_stft = torch.stft(target_flat, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True)

            # Magnitude loss
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            mag_loss = F.l1_loss(pred_mag, target_mag)

            # Phase loss (angle difference)
            pred_phase = pred_stft.angle()
            target_phase = target_stft.angle()
            phase_loss = F.l1_loss(pred_phase, target_phase)

            loss = loss + mag_loss + phase_loss

        return loss


class CombinedSpectralLoss(nn.Module):
    """Full loss combining CorrMSESpectral + multi-resolution STFT."""

    def __init__(self, alpha=0.3, beta=0.3, gamma=0.4, stft_weight=0.1,
                 fft_sizes=(64, 128, 256)):
        super().__init__()
        self.main_loss = CorrMSESpectralLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.stft_loss = MultiResolutionSTFTLoss(fft_sizes=fft_sizes)
        self.stft_weight = stft_weight

    def forward(self, pred, target):
        return self.main_loss(pred, target) + self.stft_weight * self.stft_loss(pred, target)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# build_and_train
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit closed-form for skip-connection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Build TinyDeep with CF skip init
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=256, H=64,
                     n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeep + SpectralLoss params: {n_params:,}")

    # Loss: CorrMSESpectral + multi-resolution STFT
    loss_fn = CombinedSpectralLoss(
        alpha=0.3, beta=0.3, gamma=0.4,
        stft_weight=0.1,
        fft_sizes=(64, 128, 256),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True,
                    num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False,
                    num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        ep_loss = 0.0
        n_batches = 0
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
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n_batches += 1

        vr = validate_correlation(model, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 25 == 0:
            print(f"  Epoch {ep}: loss={ep_loss/n_batches:.4f} val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")
    return model
