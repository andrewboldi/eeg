"""Iteration 058: Subject Mixup — cross-subject augmentation.

Key ideas:
1. Beta(0.1, 0.1) mixup — U-shaped distribution creates near-pure samples from
   either source, effectively synthesizing diverse subject-like spatial patterns
   rather than blurry blends (Beta(0.4,0.4) creates too many 50/50 mixes).
2. Random channel-pair permutation (p=0.05 per channel) — swaps pairs of input
   channels to break spatial overfitting and force the model to learn more
   robust spatial filters.

Architecture: TinyDeep (H=64, 2 blocks) from iter043.
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
# Architecture (TinyDeep from iter043)
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
# Loss & validation (from iter043)
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
# Augmentation helpers
# ---------------------------------------------------------------------------

def subject_mixup(x, y, alpha=0.1):
    """Mixup with Beta(alpha, alpha) — U-shaped for near-pure subject samples.

    Beta(0.1, 0.1) yields lambda near 0 or 1 most of the time, so the mixed
    sample is dominated by one source. This creates synthetic "subject-like"
    diversity without the blurry averaging of Beta(0.4, 0.4).
    """
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.shape[0], device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix


def random_channel_permutation(x, p=0.05):
    """Randomly swap pairs of input channels to break spatial overfitting.

    For each sample in the batch, with probability p per channel, swap it with
    another randomly chosen channel. Operates in-place on a clone.
    """
    B, C, T = x.shape
    x = x.clone()
    # Generate a mask of which channels to swap (per sample)
    swap_mask = torch.rand(B, C, device=x.device) < p
    # For each flagged channel, pick a random partner
    partners = torch.randint(0, C, (B, C), device=x.device)
    # Apply swaps where mask is True
    for b_idx in range(B):
        channels_to_swap = swap_mask[b_idx].nonzero(as_tuple=True)[0]
        if len(channels_to_swap) == 0:
            continue
        partner_channels = partners[b_idx, channels_to_swap]
        # Perform the swap
        temp = x[b_idx, channels_to_swap].clone()
        x[b_idx, channels_to_swap] = x[b_idx, partner_channels]
        x[b_idx, partner_channels] = temp
    return x


def random_channel_permutation_vectorized(x, p=0.05):
    """Vectorized channel permutation — no Python loops over batch."""
    B, C, T = x.shape
    x = x.clone()
    # For each (batch, channel), decide whether to swap
    swap_mask = torch.rand(B, C, device=x.device) < p  # (B, C)
    partners = torch.randint(0, C, (B, C), device=x.device)  # (B, C)

    # Build gather indices: start with identity, then overwrite swapped channels
    idx = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1).clone()  # (B, C)
    # Where swap_mask is True, redirect to partner
    idx[swap_mask] = partners[swap_mask]
    # Gather along channel dimension
    x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, T))
    return x


# ---------------------------------------------------------------------------
# build_and_train
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit closed-form baseline for skip-connection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)

    # Step 2: Build TinyDeep with CF skip init
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=256, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeep (subject_mixup) params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-5)

    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)

            # --- Subject Mixup: Beta(0.1, 0.1) for U-shaped lambda ---
            x, y = subject_mixup(x, y, alpha=0.1)

            # --- Random channel permutation (p=0.05) ---
            x = random_channel_permutation_vectorized(x, p=0.05)

            # --- Standard channel dropout (15%) ---
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()
        vr = validate_correlation(model, vl, device)

        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f}, lr={sched.get_last_lr()[0]:.2e})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Subject Mixup best val_r: {best_r:.4f}")
    return model
