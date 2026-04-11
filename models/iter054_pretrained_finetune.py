"""Iteration 054: Fine-tune pretrained HBN temporal encoder for Ear-SAAD.

Loads a TemporalEncoder pretrained via masked channel autoencoding on HBN-EEG
(scripts/pretrain_hbn.py), then fine-tunes it on the scalp-to-in-ear task.

Architecture:
  1. Pretrained TemporalEncoder applied per-channel -> (B, C_scalp, 64)
  2. 2-block TransformerEncoder over channel tokens (dim=64, 4 heads)
  3. Cross-attention decoder: 12 learnable queries attend to C_scalp tokens
  4. Output projection: Linear(64, T) per output channel
  5. CF skip connection for stable gradients

Training strategy:
  - Freeze encoder for first 30 epochs (train transformer + decoder only)
  - Unfreeze encoder at epoch 31 with 10x lower LR to prevent catastrophic forgetting
  - Combined MSE + correlation loss, correlation-based early stopping
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ── Pretrained encoder (exact copy from scripts/pretrain_hbn.py) ──────────

EMBED_DIM = 64
PRETRAINED_PATH = Path("models/pretrained/hbn_temporal_encoder.pt")


class TemporalEncoder(nn.Module):
    """Per-channel temporal encoder: (1, T) -> (EMBED_DIM,).

    Shared across all channels -- learns universal EEG waveform features.
    Small Conv1d stack: 3 layers, ~15K parameters.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, window_size: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        # Conv stack: 256 -> 128 -> 64 -> 32 time steps
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # -> 128
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2),  # -> 64
            nn.GELU(),
            nn.Conv1d(48, embed_dim, kernel_size=5, stride=2, padding=2),  # -> 32
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # -> 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*C, 1, T) -> (B*C, embed_dim)."""
        return self.encoder(x).squeeze(-1)


# ── Cross-attention decoder ───────────────────────────────────────────────


class CrossAttentionDecoder(nn.Module):
    """12 learnable queries attend to C_scalp channel tokens to produce output."""

    def __init__(self, n_queries: int, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, channel_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_tokens: (B, C_scalp, embed_dim) from transformer encoder
        Returns:
            (B, n_queries, embed_dim) -- one token per output channel
        """
        B = channel_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)  # (B, n_queries, D)

        # Cross-attention: queries attend to channel tokens
        attn_out, _ = self.cross_attn(q, channel_tokens, channel_tokens)
        h = self.norm1(q + attn_out)

        # FFN
        h = self.norm2(h + self.ffn(h))
        return h  # (B, n_queries, embed_dim)


# ── Full fine-tuning model ────────────────────────────────────────────────


class PretrainedFinetuneModel(nn.Module):
    """Fine-tune pretrained temporal encoder for scalp-to-in-ear prediction.

    Pipeline:
        input (B, C_scalp, T)
        -> per-channel encode -> (B, C_scalp, 64)
        -> TransformerEncoder (channel interactions) -> (B, C_scalp, 64)
        -> CrossAttentionDecoder (12 queries) -> (B, 12, 64)
        -> time projection -> (B, 12, T)
        + CF skip connection
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        T: int,
        embed_dim: int = EMBED_DIM,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
    ):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.T = T
        self.embed_dim = embed_dim

        # 1. Per-channel temporal encoder (pretrained)
        self.temporal_encoder = TemporalEncoder(embed_dim=embed_dim)

        # 2. Channel-level transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # 3. Cross-attention decoder (12 learnable queries)
        self.decoder = CrossAttentionDecoder(
            n_queries=C_out,
            embed_dim=embed_dim,
            n_heads=n_heads,
        )

        # 4. Time projection: embed_dim -> T for each output channel
        self.time_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, T),
        )

        # 5. CF skip connection (1x1 conv: C_in -> C_out)
        self.skip = nn.Conv1d(C_in, C_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) -> (B, C_out, T)."""
        B, C, T = x.shape

        # Skip connection
        skip = self.skip(x)  # (B, C_out, T)

        # Per-channel encoding
        x_flat = x.reshape(B * C, 1, T)  # (B*C, 1, T)
        embeds = self.temporal_encoder(x_flat)  # (B*C, embed_dim)
        channel_tokens = embeds.reshape(B, C, self.embed_dim)  # (B, C, embed_dim)

        # Channel transformer
        channel_tokens = self.channel_transformer(channel_tokens)  # (B, C, embed_dim)

        # Cross-attention decoder
        out_tokens = self.decoder(channel_tokens)  # (B, C_out, embed_dim)

        # Project to time dimension
        out = self.time_proj(out_tokens)  # (B, C_out, T)

        return out + skip


# ── Loss and validation ───────────────────────────────────────────────────


class CorrMSELoss(nn.Module):
    """Combined MSE + Pearson correlation loss."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        pm = pred - pred.mean(dim=-1, keepdim=True)
        tm = target - target.mean(dim=-1, keepdim=True)
        r = (pm * tm).sum(dim=-1) / (
            pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8
        )
        corr_loss = 1.0 - r.mean()
        return self.alpha * mse + (1.0 - self.alpha) * corr_loss


def validate_correlation(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute mean Pearson r on validation set."""
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(dim=-1, keepdim=True)
            tm = y - y.mean(dim=-1, keepdim=True)
            r = (pm * tm).sum(dim=-1) / (
                pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8
            )
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ── Main entry point ──────────────────────────────────────────────────────


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build and train pretrained-encoder fine-tuning model."""

    T = train_ds.scalp.shape[2]  # window length (40 or 256)

    # ── Step 1: Fit CF baseline for skip-connection init ──
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # ── Step 2: Build model ──
    model = PretrainedFinetuneModel(
        C_in=C_scalp,
        C_out=C_inear,
        T=T,
        embed_dim=EMBED_DIM,
        n_heads=4,
        n_transformer_layers=2,
    ).to(device)

    # Initialize skip connection from CF weights
    with torch.no_grad():
        # CF weight is (C_out, C_in), skip conv weight is (C_out, C_in, 1)
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))
        if model.skip.bias is not None:
            model.skip.bias.zero_()

    # ── Step 3: Load pretrained encoder weights ──
    pretrained_loaded = False
    if PRETRAINED_PATH.exists():
        try:
            ckpt = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
            encoder_state = ckpt["encoder_state_dict"]
            # The pretrained encoder has the same architecture, load weights
            model.temporal_encoder.load_state_dict(encoder_state, strict=False)
            pretrained_loaded = True
            val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", "?"))
            print(f"Loaded pretrained encoder from {PRETRAINED_PATH} (val_loss={val_loss})")
        except Exception as e:
            warnings.warn(
                f"Failed to load pretrained weights from {PRETRAINED_PATH}: {e}. "
                "Falling back to random initialization."
            )
    else:
        warnings.warn(
            f"Pretrained weights not found at {PRETRAINED_PATH}. "
            "Using random initialization. Run scripts/pretrain_hbn.py first."
        )

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.temporal_encoder.parameters())
    print(f"Total params: {n_params:,} (encoder: {n_encoder:,})")
    print(f"Pretrained encoder loaded: {pretrained_loaded}")

    # ── Step 4: Training setup ──
    FREEZE_EPOCHS = 30
    TOTAL_EPOCHS = 150
    PATIENCE = 30
    BASE_LR = 3e-4
    ENCODER_LR = 3e-5  # 10x lower for pretrained encoder

    loss_fn = CorrMSELoss(alpha=0.5)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )

    # Phase 1 optimizer: freeze encoder, train transformer + decoder only
    for p in model.temporal_encoder.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=BASE_LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

    best_r = -1.0
    best_state = None
    no_improve = 0

    # ── Step 5: Training loop ──
    for epoch in range(1, TOTAL_EPOCHS + 1):

        # Phase transition: unfreeze encoder at epoch FREEZE_EPOCHS+1
        if epoch == FREEZE_EPOCHS + 1:
            print(f"  Unfreezing encoder at epoch {epoch} with LR={ENCODER_LR:.1e}")
            for p in model.temporal_encoder.parameters():
                p.requires_grad = True

            # Rebuild optimizer with two param groups
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if not n.startswith("temporal_encoder.") and p.requires_grad
                        ],
                        "lr": BASE_LR,
                    },
                    {
                        "params": list(model.temporal_encoder.parameters()),
                        "lr": ENCODER_LR,
                    },
                ],
                weight_decay=1e-2,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=TOTAL_EPOCHS - FREEZE_EPOCHS
            )
            no_improve = 0  # Reset patience after unfreeze

        # ── Train epoch ──
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]

            # Channel dropout (15%)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # ── Validate ──
        val_r = validate_correlation(model, val_loader, device)

        if val_r > best_r:
            best_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == FREEZE_EPOCHS + 1:
            phase = "frozen" if epoch <= FREEZE_EPOCHS else "finetuning"
            print(
                f"  Epoch {epoch:3d}/{TOTAL_EPOCHS} [{phase}] | "
                f"val_r={val_r:.4f} (best={best_r:.4f}) | "
                f"no_imp={no_improve}"
            )

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # ── Load best and return ──
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")

    return model
