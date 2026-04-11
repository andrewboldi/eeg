"""SimCLR-style contrastive pretraining on multi-dataset EEG data.

Learns subject-invariant temporal representations by contrasting augmented views
of EEG channels. Positive pairs = two augmented views of the same channel from
the same window. Negative pairs = channels from different windows in the batch.

Datasets loaded (same as pretrain_unified.py, skip MOABB):
1. HBN-EEG: ~20 subjects, 111ch, 500Hz
2. Mobile BCI: ~8 subjects, 36ch, 500Hz
3. EESM19/23: ~12 subjects, BIDS .set, 500Hz
4. Ear-SAAD: 15 subjects, 46+12ch, 128Hz

Architecture:
- Encoder: TemporalEncoder (Conv1d 1->32->64->128) -- same as pretrain_unified.py
- Projection head: Linear(128, 64) for contrastive loss
- NT-Xent loss (normalized temperature-scaled cross entropy)

Augmentations (per channel):
- Temporal shift: circular shift by random offset
- Gaussian noise: additive jitter
- Amplitude scaling: multiply by random scalar

Usage:
    uv run python scripts/pretrain_contrastive.py
    uv run python scripts/pretrain_contrastive.py --epochs 50 --batch-size 256 --lr 1e-3
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Reuse preprocessing from pretrain_unified ─────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretrain_unified import (
    BP_HIGH,
    BP_LOW,
    EMBED_DIM,
    FS_TARGET,
    WINDOW_SIZE,
    load_ear_saad_data,
    load_eesm_data,
    load_hbn_data,
    load_mobile_bci_data,
)

# ── Contrastive constants ─────────────────────────────────────────────────
TEMPERATURE = 0.1
PROJ_DIM = 64
SAVE_PATH = Path("models/pretrained/contrastive_temporal_encoder.pt")


# ═══════════════════════════════════════════════════════════════════════════
# Augmentations
# ═══════════════════════════════════════════════════════════════════════════


def augment_temporal_shift(x: torch.Tensor, max_shift: int = 20) -> torch.Tensor:
    """Circular temporal shift by a random offset. x: (B, T)."""
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), device=x.device)
    out = torch.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = torch.roll(x[i], shifts[i].item(), dims=0)
    return out


def augment_noise(x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """Additive Gaussian noise. x: (B, T)."""
    return x + torch.randn_like(x) * noise_std


def augment_amplitude_scale(
    x: torch.Tensor, low: float = 0.8, high: float = 1.2
) -> torch.Tensor:
    """Random per-sample amplitude scaling. x: (B, T)."""
    scales = torch.empty(x.shape[0], 1, device=x.device).uniform_(low, high)
    return x * scales


def create_augmented_view(x: torch.Tensor) -> torch.Tensor:
    """Apply all three augmentations to create one view. x: (B, T)."""
    x = augment_temporal_shift(x)
    x = augment_noise(x)
    x = augment_amplitude_scale(x)
    return x


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════


class TemporalEncoder(nn.Module):
    """Per-channel temporal encoder: (1, T) -> (EMBED_DIM,).

    Identical to pretrain_unified.py TemporalEncoder.
    Conv1d stack: 1 -> 32 -> 64 -> 128, ~50K parameters.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, window_size: int = WINDOW_SIZE):
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
        """x: (B, 1, T) -> (B, embed_dim)."""
        return self.encoder(x).squeeze(-1)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive loss.

    Maps encoder output to a lower-dim space where NT-Xent is computed.
    Following SimCLR: Linear -> ReLU -> Linear.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, embed_dim) -> (B, proj_dim)."""
        return self.head(x)


class ContrastiveModel(nn.Module):
    """Encoder + projection head for SimCLR-style contrastive learning."""

    def __init__(self, embed_dim: int = EMBED_DIM, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.encoder = TemporalEncoder(embed_dim)
        self.projector = ProjectionHead(embed_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 1, T) -> (embeddings, projections).

        Returns:
            h: (B, embed_dim) -- encoder representations (for transfer)
            z: (B, proj_dim)  -- projected representations (for NT-Xent)
        """
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

    def get_temporal_encoder(self) -> TemporalEncoder:
        """Return just the temporal encoder for transfer."""
        return self.encoder


# ═══════════════════════════════════════════════════════════════════════════
# NT-Xent Loss
# ═══════════════════════════════════════════════════════════════════════════


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE
) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss (NT-Xent).

    Given N samples, z1[i] and z2[i] are positive pairs (two views of same channel).
    All other 2(N-1) samples in the batch are negatives.

    Args:
        z1: (N, D) L2-normalized projections from view 1
        z2: (N, D) L2-normalized projections from view 2
        temperature: scaling temperature (lower = sharper distribution)

    Returns:
        Scalar loss.
    """
    N = z1.shape[0]
    device = z1.device

    # L2 normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: [z1_0, ..., z1_N-1, z2_0, ..., z2_N-1]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)

    # Cosine similarity matrix: (2N, 2N)
    sim = torch.mm(z, z.t()) / temperature  # (2N, 2N)

    # Mask out self-similarity (diagonal)
    mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim.masked_fill_(mask_self, -1e9)

    # For each z1[i], the positive is z2[i] (at index N+i)
    # For each z2[i], the positive is z1[i] (at index i)
    # Labels: for row i (i < N), positive is at column N+i
    #         for row N+i, positive is at column i
    labels = torch.cat(
        [torch.arange(N, 2 * N, device=device), torch.arange(N, device=device)]
    )

    loss = F.cross_entropy(sim, labels)
    return loss


# ═══════════════════════════════════════════════════════════════════════════
# Dataset -- flattens all channels into individual samples
# ═══════════════════════════════════════════════════════════════════════════


class ContrastiveEEGDataset(Dataset):
    """Dataset that stores individual channel waveforms (T,) from all datasets.

    Each item is a single channel's waveform from one window.
    Augmentation is applied on-the-fly during training.
    """

    def __init__(self, subject_windows: list[np.ndarray]):
        """subject_windows: list of arrays, each (N_windows, C_i, T)."""
        self.channels: list[torch.Tensor] = []
        for windows in subject_windows:
            n_win, n_ch, T = windows.shape
            for w in range(n_win):
                for c in range(n_ch):
                    self.channels.append(torch.from_numpy(windows[w, c]))  # (T,)

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        return self.channels[idx]  # (T,)


def contrastive_collate(batch: list[torch.Tensor]):
    """Collate channel waveforms into a batch. Returns (B, T)."""
    return torch.stack(batch, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════


def train(
    model: ContrastiveModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    temperature: float,
    device: torch.device,
    save_path: Path,
):
    """Train with NT-Xent loss, cosine LR schedule, and gradient clipping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Training for {epochs} epochs on {device}")
    logger.info(
        f"Architecture: embed_dim={EMBED_DIM}, proj_dim={PROJ_DIM}, "
        f"temperature={temperature}"
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)  # (B, T)

            # Create two augmented views
            view1 = create_augmented_view(x)  # (B, T)
            view2 = create_augmented_view(x)  # (B, T)

            # Add channel dim for Conv1d: (B, T) -> (B, 1, T)
            _, z1 = model(view1.unsqueeze(1))
            _, z2 = model(view2.unsqueeze(1))

            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                view1 = create_augmented_view(x)
                view2 = create_augmented_view(x)
                _, z1 = model(view1.unsqueeze(1))
                _, z2 = model(view2.unsqueeze(1))
                loss = nt_xent_loss(z1, z2, temperature)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
            patience_counter = 0
        else:
            marker = ""
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or marker:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_ntxent={train_loss:.4f} | val_ntxent={val_loss:.4f} | "
                f"best={best_val_loss:.4f} | lr={lr_now:.2e} | "
                f"{elapsed:.1f}s{marker}"
            )

        if patience_counter >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # Save best model
    if best_state is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract encoder state dict (without projector)
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in best_state.items()
            if k.startswith("encoder.")
        }

        checkpoint = {
            "model_state_dict": best_state,
            "encoder_state_dict": encoder_state,
            "embed_dim": EMBED_DIM,
            "proj_dim": PROJ_DIM,
            "temperature": temperature,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch,
            "window_size": WINDOW_SIZE,
            "fs": FS_TARGET,
            "bp_low": BP_LOW,
            "bp_high": BP_HIGH,
            "method": "simclr_ntxent",
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved best model to {save_path} (val_ntxent={best_val_loss:.4f})")

    return best_val_loss


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR-style contrastive pretraining on EEG data"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default=str(SAVE_PATH))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Load datasets (skip MOABB per spec) ───────────────────────────────
    all_windows: list[np.ndarray] = []
    dataset_stats: dict[str, dict] = {}

    loaders = [
        ("HBN-EEG", load_hbn_data),
        ("Mobile BCI Ear", load_mobile_bci_data),
        ("EESM19/23", load_eesm_data),
        ("Ear-SAAD", load_ear_saad_data),
    ]

    for name, loader_fn in loaders:
        logger.info(f"Loading {name}...")
        t0 = time.time()
        data = loader_fn()
        dt = time.time() - t0

        if data:
            n_subjects = len(data)
            n_windows = sum(d.shape[0] for d in data)
            ch_counts = [d.shape[1] for d in data]
            dataset_stats[name] = {
                "subjects": n_subjects,
                "windows": n_windows,
                "channels": f"{min(ch_counts)}-{max(ch_counts)}",
            }
            all_windows.extend(data)
            logger.info(
                f"  {name}: {n_subjects} subjects, {n_windows} windows, "
                f"channels {min(ch_counts)}-{max(ch_counts)}, {dt:.1f}s"
            )
        else:
            logger.info(f"  {name}: no data loaded ({dt:.1f}s)")

    if not all_windows:
        raise RuntimeError(
            "No data loaded from any dataset! Check data directories."
        )

    # ── Summary ───────────────────────────────────────────────────────────
    total_windows = sum(d.shape[0] for d in all_windows)
    total_subjects = len(all_windows)
    total_channels = sum(d.shape[0] * d.shape[1] for d in all_windows)
    logger.info(f"\n{'='*70}")
    logger.info("CONTRASTIVE PRETRAINING DATASET SUMMARY")
    logger.info(f"  Total: {total_subjects} recordings, {total_windows} windows")
    logger.info(f"  Total channel-windows (training samples): {total_channels:,}")
    for name, stats in dataset_stats.items():
        logger.info(
            f"  {name}: {stats['subjects']} subj, "
            f"{stats['windows']} win, {stats['channels']}ch"
        )
    logger.info(f"{'='*70}\n")

    # ── Subject-level train/val split (15% held-out) ──────────────────────
    n_subjects = len(all_windows)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_subjects)
    n_val_subj = max(1, int(n_subjects * 0.15))
    val_indices = set(indices[:n_val_subj].tolist())
    train_windows = [all_windows[i] for i in range(n_subjects) if i not in val_indices]
    val_windows = [all_windows[i] for i in range(n_subjects) if i in val_indices]
    logger.info(
        f"Subject split: {n_subjects - n_val_subj} train, {n_val_subj} val"
    )

    train_ds = ContrastiveEEGDataset(train_windows)
    val_ds = ContrastiveEEGDataset(val_windows)
    logger.info(f"Train: {len(train_ds)} channel-windows, Val: {len(val_ds)} channel-windows")

    del all_windows, train_windows, val_windows

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=contrastive_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,  # NT-Xent needs consistent batch size
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=contrastive_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    # ── Build model ───────────────────────────────────────────────────────
    model = ContrastiveModel(
        embed_dim=EMBED_DIM,
        proj_dim=PROJ_DIM,
    ).to(device)

    # ── Train ─────────────────────────────────────────────────────────────
    save_path = Path(args.save_path)
    best_loss = train(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        args.temperature,
        device,
        save_path,
    )

    logger.info(f"\nContrastive pretraining complete. Best val NT-Xent: {best_loss:.4f}")
    logger.info(f"Encoder saved to: {save_path}")
    logger.info(
        "To load the encoder for fine-tuning:\n"
        f"  ckpt = torch.load('{save_path}')\n"
        f"  encoder = TemporalEncoder(embed_dim=ckpt['embed_dim'])\n"
        "  encoder.load_state_dict(ckpt['encoder_state_dict'])"
    )


if __name__ == "__main__":
    main()
