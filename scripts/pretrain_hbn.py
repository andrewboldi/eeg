"""Self-supervised pretraining on HBN-EEG using masked channel autoencoding.

Pretrains a per-channel temporal encoder on 20 HBN subjects (111 channels,
500 Hz resting-state EEG). The encoder learns universal EEG temporal features
via a masked autoencoding objective: mask 50% of channels, encode visible
channels, use a spatial transformer to predict masked channel embeddings,
then decode back to waveforms.

The pretrained temporal encoder can then be loaded and fine-tuned on the
Ear-SAAD scalp-to-in-ear prediction task.

Usage:
    uv run python scripts/pretrain_hbn.py
    uv run python scripts/pretrain_hbn.py --epochs 50 --batch-size 128
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, resample_poly
from torch.utils.data import DataLoader, Dataset, random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Preprocessing constants ────────────────────────────────────────────────
HBN_ROOT = Path("data/raw/hbn_eeg")
BP_LOW = 1.0
BP_HIGH = 45.0
FS_TARGET = 128
WINDOW_SIZE = 256  # 2s at 128 Hz
WINDOW_STRIDE = 128  # 1s overlap (50%)

# ── Model constants ────────────────────────────────────────────────────────
EMBED_DIM = 64
MASK_RATIO = 0.50
SAVE_PATH = Path("models/pretrained/hbn_temporal_encoder.pt")


# ═══════════════════════════════════════════════════════════════════════════
# Data loading & preprocessing
# ═══════════════════════════════════════════════════════════════════════════


def bandpass_filter(data: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass. data: (C, T)."""
    nyq = fs / 2.0
    b, a = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(data.dtype)


def downsample(data: np.ndarray, fs_orig: float) -> np.ndarray:
    """Polyphase resampling to FS_TARGET. data: (C, T)."""
    if int(fs_orig) == FS_TARGET:
        return data
    up = FS_TARGET
    down = int(fs_orig)
    g = gcd(up, down)
    return resample_poly(data, up // g, down // g, axis=-1).astype(data.dtype)


def load_hbn_subject(mat_path: Path) -> tuple[np.ndarray, float, int] | None:
    """Load one HBN subject's RestingState .mat file.

    Returns (data, srate, n_channels) or None on failure.
    data shape: (C, T) float32
    """
    try:
        mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
    except Exception as e:
        logger.warning(f"Failed to load {mat_path}: {e}")
        return None

    # The EEGLAB struct is stored under 'result' or 'EEG'
    struct = None
    for key in ["result", "EEG"]:
        if key in mat:
            struct = mat[key]
            break

    if struct is None:
        # Try top-level keys that look like structs
        for key in mat:
            if not key.startswith("_"):
                struct = mat[key]
                break

    if struct is None:
        logger.warning(f"No EEGLAB struct found in {mat_path}")
        return None

    # Extract fields from structured array
    try:
        if hasattr(struct, "dtype") and struct.dtype.names:
            data = struct["data"].item() if struct["data"].ndim == 0 else struct["data"]
            srate = float(
                struct["srate"].item() if struct["srate"].ndim == 0 else struct["srate"]
            )
            nbchan = int(
                struct["nbchan"].item()
                if struct["nbchan"].ndim == 0
                else struct["nbchan"]
            )
        else:
            # Might be a plain dict-like
            data = np.array(struct.data, dtype=np.float32)
            srate = float(struct.srate)
            nbchan = int(struct.nbchan)
    except Exception as e:
        logger.warning(f"Failed to extract fields from {mat_path}: {e}")
        return None

    data = np.array(data, dtype=np.float32)

    # Ensure shape is (C, T)
    if data.ndim == 1:
        logger.warning(f"1D data in {mat_path}, skipping")
        return None
    if data.shape[0] > data.shape[1]:
        data = data.T  # Was (T, C), transpose to (C, T)

    return data, srate, data.shape[0]


def preprocess_subject(mat_path: Path) -> np.ndarray | None:
    """Load, filter, downsample, z-score, and window one subject.

    Returns array of shape (N_windows, C, WINDOW_SIZE) or None.
    """
    result = load_hbn_subject(mat_path)
    if result is None:
        return None

    data, srate, n_ch = result
    duration = data.shape[1] / srate
    logger.info(
        f"  Loaded: {n_ch} channels, {srate:.0f} Hz, {duration:.1f}s"
    )

    # Reject channels that are all-zero or all-NaN
    good_mask = np.ones(n_ch, dtype=bool)
    for ch in range(n_ch):
        if np.all(np.isnan(data[ch])) or np.std(data[ch]) < 1e-10:
            good_mask[ch] = False
    data = data[good_mask]
    n_ch = data.shape[0]
    logger.info(f"  After channel rejection: {n_ch} channels")

    if n_ch < 20:
        logger.warning(f"  Too few channels ({n_ch}), skipping")
        return None

    # Interpolate NaNs
    for ch in range(n_ch):
        nans = np.isnan(data[ch])
        if nans.any():
            good = ~nans
            if good.any():
                data[ch, nans] = np.interp(
                    np.flatnonzero(nans), np.flatnonzero(good), data[ch, good]
                )
            else:
                data[ch] = 0.0

    # Bandpass filter
    try:
        data = bandpass_filter(data, srate)
    except Exception as e:
        logger.warning(f"  Filter failed: {e}, skipping")
        return None

    # Downsample
    data = downsample(data, srate)

    # Z-score per channel
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    data = (data - mean) / std

    # Window into 2s segments
    T = data.shape[1]
    windows = []
    for start in range(0, T - WINDOW_SIZE + 1, WINDOW_STRIDE):
        windows.append(data[:, start : start + WINDOW_SIZE])

    if not windows:
        logger.warning(f"  No valid windows")
        return None

    windows = np.stack(windows)  # (N, C, WINDOW_SIZE)
    logger.info(f"  -> {windows.shape[0]} windows, {n_ch} channels")
    return windows.astype(np.float32)


def load_all_hbn_data() -> list[np.ndarray]:
    """Load and preprocess all HBN subjects. Returns list of (N_i, C_i, T) arrays."""
    subjects = sorted(HBN_ROOT.iterdir())
    all_data = []

    for subj_dir in subjects:
        mat_path = subj_dir / "RestingState.mat"
        if not mat_path.exists():
            continue
        logger.info(f"Processing {subj_dir.name}...")
        windows = preprocess_subject(mat_path)
        if windows is not None:
            all_data.append(windows)

    logger.info(f"Loaded {len(all_data)} subjects total")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════


class HBNMaskedDataset(Dataset):
    """Dataset that yields (all_channels, n_channels) per window.

    Each subject may have different channel counts, so we store windows
    per-subject and sample uniformly.
    """

    def __init__(self, subject_windows: list[np.ndarray]):
        """subject_windows: list of arrays, each (N_windows, C_i, T)."""
        self.segments = []
        for windows in subject_windows:
            for i in range(windows.shape[0]):
                self.segments.append(torch.from_numpy(windows[i]))  # (C, T)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]  # (C, T) -- variable C across subjects


def collate_variable_channels(batch: list[torch.Tensor]):
    """Collate windows with variable channel counts by padding.

    Returns:
        data: (B, C_max, T) padded tensor
        channel_counts: (B,) actual channel count per sample
    """
    T = batch[0].shape[-1]
    C_max = max(x.shape[0] for x in batch)
    B = len(batch)
    data = torch.zeros(B, C_max, T)
    counts = torch.zeros(B, dtype=torch.long)
    for i, x in enumerate(batch):
        C = x.shape[0]
        data[i, :C, :] = x
        counts[i] = C
    return data, counts


# ═══════════════════════════════════════════════════════════════════════════
# Model architecture
# ═══════════════════════════════════════════════════════════════════════════


class TemporalEncoder(nn.Module):
    """Per-channel temporal encoder: (1, T) -> (EMBED_DIM,).

    Shared across all channels -- learns universal EEG waveform features.
    Small Conv1d stack: 3 layers, ~15K parameters.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, window_size: int = WINDOW_SIZE):
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


class TemporalDecoder(nn.Module):
    """Per-channel temporal decoder: (EMBED_DIM,) -> (1, T).

    Mirror of encoder using transposed convolutions.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, window_size: int = WINDOW_SIZE):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 32),
            nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                embed_dim, 48, kernel_size=5, stride=2, padding=2, output_padding=1
            ),  # 32 -> 64
            nn.GELU(),
            nn.ConvTranspose1d(
                48, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),  # 64 -> 128
            nn.GELU(),
            nn.ConvTranspose1d(
                32, 1, kernel_size=7, stride=2, padding=3, output_padding=1
            ),  # 128 -> 256
        )
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*C, embed_dim) -> (B*C, 1, T)."""
        h = self.decoder(x)  # (B*C, embed_dim * 32)
        h = h.view(-1, self.embed_dim, 32)  # (B*C, embed_dim, 32)
        out = self.deconv(h)  # (B*C, 1, ~T)
        # Trim or pad to exact window size
        if out.shape[-1] > self.window_size:
            out = out[:, :, : self.window_size]
        elif out.shape[-1] < self.window_size:
            out = F.pad(out, (0, self.window_size - out.shape[-1]))
        return out


class SpatialTransformer(nn.Module):
    """Small transformer that operates on channel embeddings.

    Takes visible channel embeddings + mask tokens, predicts all channel embeddings.
    Uses learned positional encoding (channel index agnostic since channels vary).
    """

    def __init__(self, embed_dim: int = EMBED_DIM, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        visible_embeds: torch.Tensor,
        mask: torch.Tensor,
        n_channels: int,
    ) -> torch.Tensor:
        """
        Args:
            visible_embeds: (B, C_vis, embed_dim)
            mask: (B, C_total) bool -- True = masked (to predict)
            n_channels: total channels to reconstruct

        Returns:
            all_embeds: (B, C_total, embed_dim) with predicted masked channels
        """
        B = visible_embeds.shape[0]
        device = visible_embeds.device

        # Build full sequence: visible embeddings + mask tokens
        all_embeds = self.mask_token.expand(B, n_channels, -1).clone()
        # Place visible embeddings in their positions
        vis_mask = ~mask  # True = visible
        for b in range(B):
            vis_idx = vis_mask[b].nonzero(as_tuple=True)[0]
            all_embeds[b, vis_idx] = visible_embeds[b, : vis_idx.shape[0]]

        # Transformer processes all positions
        out = self.transformer(all_embeds)
        out = self.norm(out)
        return out


class MaskedChannelAutoencoder(nn.Module):
    """Full masked channel autoencoding model.

    1. Encode each channel independently (temporal encoder)
    2. Mask 50% of channels
    3. Spatial transformer predicts masked channel embeddings from visible ones
    4. Decode predicted embeddings back to waveforms
    5. L1 loss on masked channels only
    """

    def __init__(self, embed_dim: int = EMBED_DIM, mask_ratio: float = MASK_RATIO):
        super().__init__()
        self.encoder = TemporalEncoder(embed_dim)
        self.decoder = TemporalDecoder(embed_dim)
        self.spatial = SpatialTransformer(embed_dim)
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

    def forward(
        self, data: torch.Tensor, channel_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            data: (B, C_max, T) padded multi-channel EEG
            channel_counts: (B,) actual channel count per sample

        Returns:
            loss: scalar L1 reconstruction loss on masked channels
            pred: (B, C_max, T) predicted waveforms
            mask: (B, C_max) True = masked
        """
        B, C_max, T = data.shape
        device = data.device

        # 1. Encode all channels (shared encoder)
        x_flat = data.reshape(B * C_max, 1, T)  # (B*C_max, 1, T)
        embeds_flat = self.encoder(x_flat)  # (B*C_max, embed_dim)
        embeds = embeds_flat.reshape(B, C_max, self.embed_dim)  # (B, C_max, embed_dim)

        # 2. Create channel mask (respecting actual channel counts)
        mask = torch.ones(B, C_max, dtype=torch.bool, device=device)  # True = masked
        for b in range(B):
            C = channel_counts[b].item()
            n_mask = max(1, int(C * self.mask_ratio))
            n_vis = C - n_mask
            perm = torch.randperm(C, device=device)
            vis_idx = perm[:n_vis]
            mask[b, vis_idx] = False
            # Padding channels stay masked (won't contribute to loss)

        # 3. Gather visible embeddings
        vis_mask = ~mask
        # Find max visible channels for padding
        max_vis = vis_mask.sum(dim=1).max().item()
        visible_embeds = torch.zeros(B, max_vis, self.embed_dim, device=device)
        for b in range(B):
            vis_idx = vis_mask[b].nonzero(as_tuple=True)[0]
            visible_embeds[b, : vis_idx.shape[0]] = embeds[b, vis_idx]

        # 4. Spatial transformer predicts all embeddings
        pred_embeds = self.spatial(visible_embeds, mask, C_max)

        # 5. Decode masked channels only
        pred_flat = self.decoder(pred_embeds.reshape(B * C_max, self.embed_dim))
        pred = pred_flat.reshape(B, C_max, T)

        # 6. L1 loss on masked real channels only
        # Build loss mask: masked AND within actual channel count
        loss_mask = torch.zeros(B, C_max, dtype=torch.bool, device=device)
        for b in range(B):
            C = channel_counts[b].item()
            loss_mask[b, :C] = mask[b, :C]

        if loss_mask.sum() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            masked_pred = pred[loss_mask]  # (N_masked, T)
            masked_true = data[loss_mask]  # (N_masked, T)
            loss = F.l1_loss(masked_pred, masked_true)

        return loss, pred, mask

    def get_temporal_encoder(self) -> TemporalEncoder:
        """Return just the temporal encoder for transfer."""
        return self.encoder


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════


def train(
    model: MaskedChannelAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: Path,
):
    """Train the masked autoencoder."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Training for {epochs} epochs on {device}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for data, counts in train_loader:
            data, counts = data.to(device), counts.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(data, counts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, counts in val_loader:
                data, counts = data.to(device), counts.to(device)
                loss, _, _ = model(data, counts)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step()
        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_L1={train_loss:.4f} | val_L1={val_loss:.4f} | "
                f"best={best_val_loss:.4f} | {elapsed:.1f}s{marker}"
            )

    # Save best model
    if best_state is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save full model and encoder separately for easy transfer
        checkpoint = {
            "model_state_dict": best_state,
            "encoder_state_dict": {
                k.replace("encoder.", ""): v
                for k, v in best_state.items()
                if k.startswith("encoder.")
            },
            "embed_dim": model.embed_dim,
            "mask_ratio": model.mask_ratio,
            "best_val_loss": best_val_loss,
            "epochs": epochs,
            "window_size": WINDOW_SIZE,
            "fs": FS_TARGET,
            "bp_low": BP_LOW,
            "bp_high": BP_HIGH,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved best model to {save_path} (val_L1={best_val_loss:.4f})")

    return best_val_loss


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Pretrain temporal encoder on HBN-EEG")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default=str(SAVE_PATH))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading HBN-EEG data...")
    subject_data = load_all_hbn_data()
    if not subject_data:
        raise RuntimeError("No HBN data loaded! Check data/raw/hbn_eeg/")

    total_windows = sum(w.shape[0] for w in subject_data)
    channel_counts = [w.shape[1] for w in subject_data]
    logger.info(
        f"Total: {total_windows} windows from {len(subject_data)} subjects, "
        f"channels per subject: {min(channel_counts)}-{max(channel_counts)}"
    )

    # Create dataset
    dataset = HBNMaskedDataset(subject_data)

    # Split 90/10 train/val
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {n_train} windows, Val: {n_val} windows")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_channels,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_channels,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    # Build model
    model = MaskedChannelAutoencoder(embed_dim=EMBED_DIM, mask_ratio=MASK_RATIO).to(
        device
    )

    # Train
    save_path = Path(args.save_path)
    best_loss = train(model, train_loader, val_loader, args.epochs, args.lr, device, save_path)

    logger.info(f"Pretraining complete. Best val L1: {best_loss:.4f}")
    logger.info(f"Encoder saved to: {save_path}")
    logger.info(
        "To load the encoder for fine-tuning:\n"
        "  ckpt = torch.load(save_path)\n"
        "  encoder = TemporalEncoder(embed_dim=ckpt['embed_dim'])\n"
        "  encoder.load_state_dict(ckpt['encoder_state_dict'])"
    )


if __name__ == "__main__":
    main()
