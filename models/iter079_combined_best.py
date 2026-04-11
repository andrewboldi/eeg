"""Iteration 079: Combined best findings — pruned subjects + pretrained encoder + around-ear focus.

Combines THREE proven improvements:
1. Pruned subjects: Train on {3,5,6,8,9,10,11,12}, dropping harmful subjects 1,2,4,7
2. Pretrained encoder: Load unified TemporalEncoder (128-dim) from pretrained checkpoint
3. Around-ear focus: 2x channel weights for around-ear channels (idx 27-45) in attention

Architecture:
  - Pretrained TemporalEncoder (128-dim) applied per-channel -> (B, 46, 128)
  - Learnable channel weights: around-ear init 1.0, scalp init 0.5
  - 2-block TransformerEncoder over weighted channel tokens
  - Cross-attention decoder: 12 output queries
  - CF skip connection (fitted on pruned data only)
  - Freeze encoder 30 epochs, unfreeze with 10x lower LR
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAD_SUBJECTS = {1, 2, 4, 7}
TRAIN_SUBJECTS = list(range(1, 13))
WINDOW_SIZE = 40
STRIDE = 20

EMBED_DIM = 128
PRETRAINED_PATH = Path("models/pretrained/unified_temporal_encoder.pt")

# Around-ear channel indices (27-45 in 0-indexed = channels 28-46 in the 46ch layout)
AROUND_EAR_START = 27
AROUND_EAR_END = 46  # exclusive


# ---------------------------------------------------------------------------
# Pretrained TemporalEncoder (exact copy from pretrain_unified.py)
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Per-channel temporal encoder: (1, T) -> (EMBED_DIM,).

    Shared across all channels -- learns universal EEG waveform features.
    Conv1d stack: 1 -> 32 -> 64 -> 128, ~50K parameters.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, window_size: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*C, 1, T) -> (B*C, embed_dim)."""
        return self.encoder(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Cross-attention decoder
# ---------------------------------------------------------------------------

class CrossAttentionDecoder(nn.Module):
    """Learnable queries attend to channel tokens to produce output."""

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
            channel_tokens: (B, C_in, embed_dim) from transformer encoder
        Returns:
            (B, n_queries, embed_dim)
        """
        B = channel_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)

        attn_out, _ = self.cross_attn(q, channel_tokens, channel_tokens)
        h = self.norm1(q + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h


# ---------------------------------------------------------------------------
# Combined model: pretrained encoder + channel weights + cross-attention
# ---------------------------------------------------------------------------

class CombinedBestModel(nn.Module):
    """Combines pretrained encoder, around-ear channel weighting, and CF skip.

    Pipeline:
        input (B, C_in, T)
        -> per-channel TemporalEncoder -> (B, C_in, 128)
        -> learnable channel weights (around-ear 2x) -> weighted tokens
        -> TransformerEncoder (2 blocks, channel interactions)
        -> CrossAttentionDecoder (12 queries) -> (B, 12, 128)
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

        # 1. Per-channel temporal encoder (pretrained, 128-dim)
        self.temporal_encoder = TemporalEncoder(embed_dim=embed_dim)

        # 2. Learnable channel weights: around-ear init 1.0, scalp init 0.5
        channel_weights = torch.full((C_in,), 0.5)
        if C_in > AROUND_EAR_START:
            around_ear_end = min(AROUND_EAR_END, C_in)
            channel_weights[AROUND_EAR_START:around_ear_end] = 1.0
        self.channel_weights = nn.Parameter(channel_weights)

        # 3. Channel-level transformer encoder (2 blocks, 4 heads, dim=128)
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

        # 4. Cross-attention decoder (12 learnable queries)
        self.decoder = CrossAttentionDecoder(
            n_queries=C_out,
            embed_dim=embed_dim,
            n_heads=n_heads,
        )

        # 5. Time projection: embed_dim -> T
        self.time_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, T),
        )

        # 6. CF skip connection (1x1 conv: C_in -> C_out)
        self.skip = nn.Conv1d(C_in, C_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) -> (B, C_out, T)."""
        B, C, T = x.shape

        # Skip connection
        skip = self.skip(x)  # (B, C_out, T)

        # Per-channel encoding
        x_flat = x.reshape(B * C, 1, T)  # (B*C, 1, T)
        embeds = self.temporal_encoder(x_flat)  # (B*C, embed_dim)
        channel_tokens = embeds.reshape(B, C, self.embed_dim)  # (B, C_in, 128)

        # Apply learnable channel weights (around-ear channels get 2x attention)
        weights = F.softplus(self.channel_weights)  # ensure positive
        channel_tokens = channel_tokens * weights.unsqueeze(0).unsqueeze(-1)

        # Channel transformer (2 blocks over weighted channel tokens)
        channel_tokens = self.channel_transformer(channel_tokens)

        # Cross-attention decoder (12 output queries)
        out_tokens = self.decoder(channel_tokens)  # (B, C_out, 128)

        # Project to time dimension
        out = self.time_proj(out_tokens)  # (B, C_out, T)

        return out + skip


# ---------------------------------------------------------------------------
# Loss and validation
# ---------------------------------------------------------------------------

class CorrMSELoss(nn.Module):
    """Combined MSE + Pearson correlation loss."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        pm = pred - pred.mean(dim=-1, keepdim=True)
        tm = target - target.mean(dim=-1, keepdim=True)
        r = (pm * tm).sum(dim=-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
        corr_loss = 1.0 - r.mean()
        return self.alpha * mse + (1.0 - self.alpha) * corr_loss


def validate_correlation(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """Compute mean Pearson r on validation set."""
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(dim=-1, keepdim=True)
            tm = y - y.mean(dim=-1, keepdim=True)
            r = (pm * tm).sum(dim=-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Subject boundary computation (from iter078)
# ---------------------------------------------------------------------------

def compute_subject_mask(n_train_windows: int, n_val_windows: int):
    """Compute boolean masks for keeping only good subjects.

    Reloads raw data to figure out how many windows each subject contributes,
    then builds masks aligned with the train_ds and val_ds that the benchmark
    already created.
    """
    from scripts.real_data_experiment import load_subject, window_trials

    counts = []
    valid_subjects = []
    for subj in TRAIN_SUBJECTS:
        try:
            scalp_trials, inear_trials, _, _, _ = load_subject(subj)
            scalp_w, inear_w = window_trials(scalp_trials, inear_trials,
                                             window_size=WINDOW_SIZE, stride=STRIDE)
            if scalp_w.shape[0] > 0:
                counts.append(scalp_w.shape[0])
                valid_subjects.append(subj)
        except Exception:
            pass

    subject_ids = np.concatenate([
        np.full(c, s, dtype=np.int32)
        for s, c in zip(valid_subjects, counts)
    ])

    n_total = subject_ids.shape[0]
    n_train_expected = int(0.9 * n_total)

    assert n_train_expected == n_train_windows, (
        f"Expected {n_train_expected} train windows but got {n_train_windows}. "
        f"Data loading mismatch."
    )
    assert n_total - n_train_expected == n_val_windows, (
        f"Expected {n_total - n_train_expected} val windows but got {n_val_windows}. "
        f"Data loading mismatch."
    )

    train_ids = subject_ids[:n_train_expected]
    val_ids = subject_ids[n_train_expected:]

    train_mask = np.array([s not in BAD_SUBJECTS for s in train_ids])
    val_mask = np.array([s not in BAD_SUBJECTS for s in val_ids])

    n_kept_train = train_mask.sum()
    n_kept_val = val_mask.sum()
    print(f"Subject pruning: train {n_train_windows} -> {n_kept_train} "
          f"({n_kept_train / n_train_windows * 100:.1f}%), "
          f"val {n_val_windows} -> {n_kept_val} "
          f"({n_kept_val / n_val_windows * 100:.1f}%)")

    kept = sorted(set(valid_subjects) - BAD_SUBJECTS)
    dropped = sorted(BAD_SUBJECTS & set(valid_subjects))
    print(f"  Kept subjects: {kept}")
    print(f"  Dropped subjects: {dropped}")

    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build and train combined best model with pruned subjects."""

    T = train_ds.scalp.shape[2]  # window length

    # -- Step 1: Prune subjects -----------------------------------------------
    print("Computing subject boundaries for pruned training...")
    train_mask, val_mask = compute_subject_mask(len(train_ds), len(val_ds))

    pruned_train_scalp = train_ds.scalp.numpy()[train_mask]
    pruned_train_inear = train_ds.inear.numpy()[train_mask]
    pruned_val_scalp = val_ds.scalp.numpy()[val_mask]
    pruned_val_inear = val_ds.inear.numpy()[val_mask]

    pruned_train_ds = EEGDataset(pruned_train_scalp, pruned_train_inear)
    pruned_val_ds = EEGDataset(pruned_val_scalp, pruned_val_inear)

    print(f"Pruned train: {len(pruned_train_ds)} windows, "
          f"Pruned val: {len(pruned_val_ds)} windows")

    # -- Step 2: Fit CF on pruned data only -----------------------------------
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(pruned_train_ds.scalp.numpy(), pruned_train_ds.inear.numpy())

    # -- Step 3: Build model --------------------------------------------------
    model = CombinedBestModel(
        C_in=C_scalp,
        C_out=C_inear,
        T=T,
        embed_dim=EMBED_DIM,
        n_heads=4,
        n_transformer_layers=2,
    ).to(device)

    # Initialize skip connection from CF weights
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))
        if model.skip.bias is not None:
            model.skip.bias.zero_()

    # -- Step 4: Load pretrained encoder weights ------------------------------
    pretrained_loaded = False
    if PRETRAINED_PATH.exists():
        try:
            ckpt = torch.load(
                PRETRAINED_PATH, map_location=device, weights_only=False
            )
            encoder_state = ckpt["encoder_state_dict"]
            model.temporal_encoder.load_state_dict(encoder_state, strict=False)
            pretrained_loaded = True
            val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", "?"))
            n_datasets = ckpt.get("n_datasets", "?")
            n_windows = ckpt.get("n_windows_total", "?")
            print(
                f"Loaded unified pretrained encoder from {PRETRAINED_PATH} "
                f"(val_loss={val_loss}, datasets={n_datasets}, windows={n_windows})"
            )
        except Exception as e:
            warnings.warn(
                f"Failed to load pretrained weights from {PRETRAINED_PATH}: {e}. "
                "Falling back to random initialization."
            )
    else:
        warnings.warn(
            f"Pretrained weights not found at {PRETRAINED_PATH}. "
            "Using random initialization. Run scripts/pretrain_unified.py first."
        )

    # Report channel weight initialization
    w = model.channel_weights.data
    n_scalp = min(AROUND_EAR_START, C_scalp)
    n_around = max(0, min(AROUND_EAR_END, C_scalp) - AROUND_EAR_START)
    print(f"Channel weights: {n_scalp} scalp @ {w[:n_scalp].mean():.2f}, "
          f"{n_around} around-ear @ {w[AROUND_EAR_START:AROUND_EAR_START+n_around].mean():.2f}")

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.temporal_encoder.parameters())
    print(f"Total params: {n_params:,} (encoder: {n_encoder:,})")
    print(f"Pretrained encoder loaded: {pretrained_loaded}")

    # -- Step 5: Training setup -----------------------------------------------
    FREEZE_EPOCHS = 30
    TOTAL_EPOCHS = 150
    PATIENCE = 30
    BASE_LR = 3e-4
    ENCODER_LR = 3e-5  # 10x lower for pretrained encoder

    loss_fn = CorrMSELoss(alpha=0.5)

    train_loader = DataLoader(
        pruned_train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        pruned_val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )

    # Phase 1: Freeze encoder, train transformer + decoder + channel weights
    for p in model.temporal_encoder.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=BASE_LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS
    )

    best_r = -1.0
    best_state = None
    no_improve = 0

    # -- Step 6: Training loop ------------------------------------------------
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
                            if not n.startswith("temporal_encoder.")
                            and p.requires_grad
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

        # -- Train epoch --
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]

            # Channel dropout (15%)
            mask = (
                torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15
            ).float()
            x = x * mask / 0.85

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # -- Validate --
        val_r = validate_correlation(model, val_loader, device)

        if val_r > best_r:
            best_r = val_r
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == FREEZE_EPOCHS + 1:
            phase = "frozen" if epoch <= FREEZE_EPOCHS else "finetuning"
            w = model.channel_weights.data
            print(
                f"  Epoch {epoch:3d}/{TOTAL_EPOCHS} [{phase}] | "
                f"val_r={val_r:.4f} (best={best_r:.4f}) | "
                f"no_imp={no_improve} | "
                f"ch_w: scalp={F.softplus(w[:AROUND_EAR_START]).mean():.3f} "
                f"ear={F.softplus(w[AROUND_EAR_START:]).mean():.3f}"
            )

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # -- Load best and return -------------------------------------------------
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Report final channel weights
    w = model.channel_weights.data
    print(f"Final channel weights (softplus): "
          f"scalp={F.softplus(w[:AROUND_EAR_START]).mean():.3f}, "
          f"around-ear={F.softplus(w[AROUND_EAR_START:]).mean():.3f}")
    print(f"Best val_r: {best_r:.4f}")

    return model
