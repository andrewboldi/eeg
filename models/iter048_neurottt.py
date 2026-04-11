"""Iteration 048: NeuroTTT — Test-Time Training with Self-Supervised Auxiliary Tasks.

Inspired by NeuroTTT (Wang et al., ICLR 2026).

Architecture:
- Shared backbone: MultiScaleConv + TransformerEncoder (H=64, 2 blocks) with BatchNorm
- Main head: Regression head predicting 12 in-ear channels
- SSL auxiliary heads (2-layer MLPs):
  a. Temporal reversal detector: binary classification (was input time-reversed?)
  b. Channel masking predictor: C_scalp-way classification (which channel was masked?)

Training loss: L_regression + 0.1 * L_reversal + 0.1 * L_channel_mask

Test-time adaptation (Tent-style):
- Before evaluation, adapt BatchNorm parameters using ONLY SSL losses on unlabeled val data
- 3-5 adaptation steps over the full validation set
- Then return adapted model for evaluation

Confidence: 70% — SSL tasks should help the backbone learn more transferable features,
and BN-only TTA provides gentle domain adaptation without destabilizing the model.
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
# Building blocks
# ---------------------------------------------------------------------------

class MultiScaleConv(nn.Module):
    """Multi-scale 1D convolution with BatchNorm after each branch."""

    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(h),
                nn.GELU(),
            )
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


class SSLHead(nn.Module):
    """Lightweight 2-layer MLP for SSL auxiliary tasks."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model: NeuroTTT
# ---------------------------------------------------------------------------

class NeuroTTTModel(nn.Module):
    """Backbone + regression head + SSL auxiliary heads for test-time training."""

    def __init__(self, C_in, C_out, T=40, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.T = T
        self.H = H

        # --- Shared backbone ---
        self.temporal = MultiScaleConv(C_in, H)
        # BatchNorm after MultiScaleConv (already inside MultiScaleConv branches)

        self.down = nn.Sequential(
            nn.Conv1d(H, H, 4, stride=4, bias=False),
            nn.BatchNorm1d(H),  # BatchNorm after downsampling
            nn.GELU(),
        )

        # Transformer with norm_first=True for Flash Attention compatibility
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)

        # --- Main regression head ---
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)
        self.skip = nn.Conv1d(C_in, C_out, 1)

        # --- SSL Head 1: Temporal reversal detector (binary) ---
        self.reversal_head = SSLHead(H, H, 2)

        # --- SSL Head 2: Channel masking predictor (C_in-way) ---
        self.channel_mask_head = SSLHead(H, H, C_in)

    def encode(self, x):
        """Shared backbone: input (B, C_in, T) -> features (B, T', H)."""
        h = self.temporal(x)            # (B, H, T)
        h = self.down(h)                # (B, H, T//4)
        h = h.transpose(1, 2)           # (B, T//4, H)
        h = self.transformer(h)         # (B, T//4, H)
        return h

    def decode(self, h, x_skip):
        """Regression head: features (B, T', H) -> prediction (B, C_out, T)."""
        skip = self.skip(x_skip)
        h_up = h.transpose(1, 2)        # (B, H, T//4)
        h_up = self.up(h_up)[:, :, :self.T]  # (B, H, T)
        h_up = self.out_norm(h_up.transpose(1, 2))  # (B, T, H)
        out = self.out_proj(h_up).transpose(1, 2)    # (B, C_out, T)
        return out + skip

    def forward(self, x):
        """Standard forward: (B, C_in, T) -> (B, C_out, T)."""
        h = self.encode(x)
        return self.decode(h, x)

    def forward_with_ssl(self, x, x_reversed, reversed_labels, x_masked, mask_labels):
        """Forward pass computing regression output + SSL losses.

        Args:
            x: (B, C_in, T) original input
            x_reversed: (B, C_in, T) inputs where some are time-reversed
            reversed_labels: (B,) binary labels (1 = reversed, 0 = original)
            x_masked: (B, C_in, T) inputs with one channel masked per sample
            mask_labels: (B,) which channel was masked (0..C_in-1)

        Returns:
            pred: (B, C_out, T) regression prediction
            reversal_logits: (B, 2)
            mask_logits: (B, C_in)
        """
        # Main regression path (on clean input)
        h_main = self.encode(x)
        pred = self.decode(h_main, x)

        # SSL path 1: Temporal reversal detection
        h_rev = self.encode(x_reversed)
        h_rev_pooled = h_rev.mean(dim=1)     # (B, H) — global average pooling over time
        reversal_logits = self.reversal_head(h_rev_pooled)

        # SSL path 2: Channel masking prediction
        h_mask = self.encode(x_masked)
        h_mask_pooled = h_mask.mean(dim=1)   # (B, H)
        mask_logits = self.channel_mask_head(h_mask_pooled)

        return pred, reversal_logits, mask_logits

    def forward_ssl_only(self, x_reversed, reversed_labels, x_masked, mask_labels):
        """SSL-only forward (for test-time adaptation). No regression head used.

        Returns:
            reversal_logits: (B, 2)
            mask_logits: (B, C_in)
        """
        h_rev = self.encode(x_reversed)
        h_rev_pooled = h_rev.mean(dim=1)
        reversal_logits = self.reversal_head(h_rev_pooled)

        h_mask = self.encode(x_masked)
        h_mask_pooled = h_mask.mean(dim=1)
        mask_logits = self.channel_mask_head(h_mask_pooled)

        return reversal_logits, mask_logits


# ---------------------------------------------------------------------------
# SSL data augmentation helpers
# ---------------------------------------------------------------------------

def create_ssl_batch(x, device):
    """Create SSL augmented versions of a batch.

    Args:
        x: (B, C, T) input tensor

    Returns:
        x_reversed: (B, C, T) — 50% of samples are time-reversed
        reversed_labels: (B,) — 1 if reversed, 0 if original
        x_masked: (B, C, T) — one random channel zeroed per sample
        mask_labels: (B,) — index of masked channel
    """
    B, C, T = x.shape

    # Temporal reversal: flip 50% of samples along time axis
    reversed_labels = (torch.rand(B, device=device) > 0.5).long()
    x_reversed = x.clone()
    rev_mask = reversed_labels.bool()
    if rev_mask.any():
        x_reversed[rev_mask] = x_reversed[rev_mask].flip(dims=[-1])

    # Channel masking: zero out one random channel per sample
    mask_labels = torch.randint(0, C, (B,), device=device)
    x_masked = x.clone()
    for i in range(B):
        x_masked[i, mask_labels[i], :] = 0.0

    return x_reversed, reversed_labels, x_masked, mask_labels


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CorrMSELoss(nn.Module):
    """Combined MSE + correlation loss."""

    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_correlation(model, loader, device):
    """Compute mean Pearson r on validation set."""
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
# Test-time adaptation (Tent-style: BN params only)
# ---------------------------------------------------------------------------

def collect_bn_params(model):
    """Collect BatchNorm affine parameters (gamma, beta) for TTA."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.affine:
                params.append(m.weight)  # gamma
                params.append(m.bias)    # beta
    return params


def set_bn_train(model):
    """Set all BatchNorm layers to train mode (updates running stats), rest stays eval."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train()


def test_time_adapt(model, val_ds, device, n_steps=5, lr=1e-3, batch_size=128):
    """Adapt BatchNorm parameters using SSL losses on unlabeled validation data.

    Only updates BN affine parameters (gamma/beta) + running stats.
    SSL heads are also updated to provide better gradients to BN layers.
    """
    print(f"  TTA: adapting BN params for {n_steps} steps on {len(val_ds)} samples...")

    # Collect BN params + SSL head params
    bn_params = collect_bn_params(model)
    ssl_params = list(model.reversal_head.parameters()) + list(model.channel_mask_head.parameters())
    tta_params = bn_params + ssl_params

    if len(bn_params) == 0:
        print("  TTA: WARNING — no BatchNorm parameters found, skipping adaptation")
        return

    print(f"  TTA: {len(bn_params)} BN params + {len(ssl_params)} SSL head params")

    # Set BN layers to train mode, everything else to eval
    set_bn_train(model)

    # Only optimize BN + SSL head params
    optimizer = torch.optim.Adam(tta_params, lr=lr)

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for step in range(n_steps):
        total_loss = 0.0
        n_batches = 0

        for x, _y in loader:
            x = x.to(device)

            # Create SSL augmentations
            x_rev, rev_labels, x_masked, mask_labels = create_ssl_batch(x, device)

            # Forward SSL only
            rev_logits, mask_logits = model.forward_ssl_only(
                x_rev, rev_labels, x_masked, mask_labels
            )

            # SSL losses
            loss_rev = F.cross_entropy(rev_logits, rev_labels)
            loss_mask = F.cross_entropy(mask_logits, mask_labels)
            loss = 0.1 * loss_rev + 0.1 * loss_mask

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tta_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  TTA step {step + 1}/{n_steps}: ssl_loss={avg_loss:.4f}")

    # Return to full eval mode
    model.eval()
    print("  TTA: adaptation complete")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    """Build NeuroTTT model, train with SSL auxiliary tasks, then adapt on val set.

    Args:
        train_ds: Training dataset (subjects 1-12, ~90%)
        val_ds: Validation dataset (subjects 1-12, ~10%)
        C_scalp: Number of scalp input channels
        C_inear: Number of in-ear output channels
        device: torch device

    Returns:
        Adapted nn.Module that maps (batch, C_scalp, T) -> (batch, C_inear, T)
    """
    T = train_ds.scalp.shape[2]  # window length (40 for 2s at 20Hz)

    # --- Step 0: Fit closed-form baseline for skip-connection init ---
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # --- Step 1: Build model ---
    model = NeuroTTTModel(
        C_in=C_scalp, C_out=C_inear, T=T,
        H=64, n_blocks=2, dropout=0.1,
    ).to(device)

    # Initialize skip connection with CF weights
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    n_bn = sum(p.numel() for p in collect_bn_params(model))
    print(f"NeuroTTT params: {n_params:,} total, {n_bn:,} BN params")

    # --- Step 2: Train with regression + SSL losses ---
    reg_loss_fn = CorrMSELoss(a=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True,
    )

    best_r = -1.0
    best_state = None
    no_improvement = 0

    for epoch in range(1, 151):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Mixup augmentation (from iter030 — known to help)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x_mix = lam * x + (1 - lam) * x[idx]
            y_mix = lam * y + (1 - lam) * y[idx]

            # Channel dropout (from iter011 — known to help)
            mask = (torch.rand(x_mix.shape[0], x_mix.shape[1], 1, device=device) > 0.15).float()
            x_aug = x_mix * mask / 0.85

            # Create SSL augmentations from the ORIGINAL (non-augmented) input
            x_rev, rev_labels, x_masked, mask_labels = create_ssl_batch(x, device)

            # Forward with SSL
            pred, rev_logits, mask_logits = model.forward_with_ssl(
                x_aug, x_rev, rev_labels, x_masked, mask_labels,
            )

            # Compute losses
            loss_reg = reg_loss_fn(pred, y_mix)
            loss_rev = F.cross_entropy(rev_logits, rev_labels)
            loss_mask = F.cross_entropy(mask_logits, mask_labels)
            loss_total = loss_reg + 0.1 * loss_rev + 0.1 * loss_mask

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate (regression only)
        val_r = validate_correlation(model, val_loader, device)
        if val_r > best_r:
            best_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1

        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: val_r={val_r:.4f} (best={best_r:.4f})")

        if no_improvement >= 30:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Training complete — best val_r: {best_r:.4f}")

    # --- Step 3: Test-time adaptation on validation set ---
    # This simulates what would happen with real test data:
    # adapt BN params using only the SSL losses (no labels needed)
    test_time_adapt(model, val_ds, device, n_steps=5, lr=1e-3, batch_size=128)

    # Check post-TTA correlation
    post_tta_r = validate_correlation(model, val_loader, device)
    print(f"Post-TTA val_r: {post_tta_r:.4f} (pre-TTA: {best_r:.4f}, delta: {post_tta_r - best_r:+.4f})")

    return model
