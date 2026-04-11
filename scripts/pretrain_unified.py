"""Unified self-supervised pretraining on ALL downloaded EEG datasets.

Pretrains a channel-agnostic temporal encoder using masked channel autoencoding
across multiple EEG datasets with different montages, sampling rates, and formats.

Datasets loaded:
1. HBN-EEG: ~20 subjects, 111ch, 500Hz (EEGLAB .mat)
2. Mobile BCI: ~8 subjects, 36ch, 500Hz (.mat with raw_x)
3. EESM19: ~12 subjects, BIDS .set, 500Hz (MNE-readable)
4. MOABB: BNCI2014_001 (9 subj, 22ch, 250Hz), BNCI2014_002 (14 subj, 15ch, 512Hz)
5. Ear-SAAD: 15 subjects, 46+12ch, already at 128Hz (from broadband_46ch.h5)

Architecture (scaled up from pretrain_hbn.py):
- TemporalEncoder: Conv1d stack (1->32->64->128) -> 128-dim per channel
- SpatialTransformer: 4-layer transformer, 4 heads over channel tokens
- TemporalDecoder: ConvTranspose1d mirror back to waveforms
- L1 masked channel reconstruction loss

Usage:
    uv run python scripts/pretrain_unified.py
    uv run python scripts/pretrain_unified.py --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, resample_poly
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Preprocessing constants ────────────────────────────────────────────────
BP_LOW = 1.0
BP_HIGH = 45.0
FS_TARGET = 128
WINDOW_SIZE = 256  # 2s at 128 Hz
WINDOW_STRIDE = 128  # 1s overlap (50%)

# ── Model constants ────────────────────────────────────────────────────────
EMBED_DIM = 128
MASK_RATIO = 0.50
N_HEADS = 4
N_LAYERS = 4
SAVE_PATH = Path("models/pretrained/unified_temporal_encoder.pt")

# ── Data roots ─────────────────────────────────────────────────────────────
HBN_ROOT = Path("data/raw/hbn_eeg")
MOBILE_BCI_ROOT = Path("data/raw/mobile_bci_ear")
EESM19_ROOT = Path("data/raw/eesm19")
EESM23_ROOT = Path("data/raw/eesm23")
EAR_SAAD_H5 = Path("data/processed/broadband_46ch.h5")
MNE_DATA_DIR = Path.home() / "mne_data"


# ═══════════════════════════════════════════════════════════════════════════
# Shared preprocessing utilities
# ═══════════════════════════════════════════════════════════════════════════


def bandpass_filter(data: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass. data: (C, T)."""
    nyq = fs / 2.0
    # Clamp high to Nyquist - margin
    high = min(BP_HIGH, nyq - 1.0)
    if high <= BP_LOW:
        return data  # Cannot filter if band is degenerate
    b, a = butter(4, [BP_LOW / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def downsample(data: np.ndarray, fs_orig: float) -> np.ndarray:
    """Polyphase resampling to FS_TARGET. data: (C, T)."""
    fs_orig_int = int(round(fs_orig))
    if fs_orig_int == FS_TARGET:
        return data
    up = FS_TARGET
    down = fs_orig_int
    g = gcd(up, down)
    return resample_poly(data, up // g, down // g, axis=-1).astype(np.float32)


def clean_and_zscore(data: np.ndarray) -> np.ndarray | None:
    """Reject bad channels, interpolate NaNs, z-score. data: (C, T) -> (C', T)."""
    n_ch = data.shape[0]

    # Reject channels that are all-zero, all-NaN, or near-constant
    good_mask = np.ones(n_ch, dtype=bool)
    for ch in range(n_ch):
        if np.all(np.isnan(data[ch])) or np.std(data[ch]) < 1e-10:
            good_mask[ch] = False
    data = data[good_mask]
    if data.shape[0] < 3:
        return None

    # Interpolate NaNs
    for ch in range(data.shape[0]):
        nans = np.isnan(data[ch])
        if nans.any():
            good = ~nans
            if good.any():
                data[ch, nans] = np.interp(
                    np.flatnonzero(nans), np.flatnonzero(good), data[ch, good]
                )
            else:
                data[ch] = 0.0

    # Z-score per channel
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    data = (data - mean) / std
    return data.astype(np.float32)


def make_windows(data: np.ndarray) -> np.ndarray | None:
    """Window (C, T) into (N, C, WINDOW_SIZE) with 50% overlap."""
    T = data.shape[1]
    windows = []
    for start in range(0, T - WINDOW_SIZE + 1, WINDOW_STRIDE):
        windows.append(data[:, start : start + WINDOW_SIZE])
    if not windows:
        return None
    return np.stack(windows).astype(np.float32)


def preprocess_continuous(data: np.ndarray, fs: float) -> np.ndarray | None:
    """Full pipeline: filter -> downsample -> clean -> window. data: (C, T)."""
    if data.shape[1] < fs * 3:  # Skip recordings shorter than 3s
        return None
    try:
        data = bandpass_filter(data, fs)
    except Exception:
        return None
    data = downsample(data, fs)
    data = clean_and_zscore(data)
    if data is None:
        return None
    return make_windows(data)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset-specific loaders (each yields list of (N_windows, C, T) arrays)
# ═══════════════════════════════════════════════════════════════════════════


def load_hbn_data() -> list[np.ndarray]:
    """Load HBN-EEG subjects. Returns list of (N_i, C_i, 256) arrays."""
    if not HBN_ROOT.exists():
        logger.info("HBN-EEG: directory not found, skipping")
        return []

    all_data = []
    subjects = sorted(HBN_ROOT.iterdir())
    for subj_dir in subjects:
        mat_path = subj_dir / "RestingState.mat"
        if not mat_path.exists():
            continue
        logger.info(f"  HBN {subj_dir.name}...")
        try:
            mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
        except Exception as e:
            logger.warning(f"    Failed to load: {e}")
            continue

        # Extract EEGLAB struct
        struct = None
        for key in ["result", "EEG"]:
            if key in mat:
                struct = mat[key]
                break
        if struct is None:
            for key in mat:
                if not key.startswith("_"):
                    struct = mat[key]
                    break
        if struct is None:
            continue

        try:
            if hasattr(struct, "dtype") and struct.dtype.names:
                data = struct["data"].item() if struct["data"].ndim == 0 else struct["data"]
                srate = float(
                    struct["srate"].item() if struct["srate"].ndim == 0 else struct["srate"]
                )
            else:
                data = np.array(struct.data, dtype=np.float32)
                srate = float(struct.srate)
        except Exception:
            continue

        data = np.array(data, dtype=np.float32)
        if data.ndim < 2:
            continue
        if data.shape[0] > data.shape[1]:
            data = data.T  # Ensure (C, T)

        windows = preprocess_continuous(data, srate)
        if windows is not None:
            all_data.append(windows)
            logger.info(f"    -> {windows.shape[0]} windows, {windows.shape[1]}ch")

    logger.info(f"  HBN-EEG: {len(all_data)} subjects loaded")
    return all_data


def load_mobile_bci_data() -> list[np.ndarray]:
    """Load Mobile BCI Ear dataset raw continuous data."""
    if not MOBILE_BCI_ROOT.exists():
        logger.info("Mobile BCI: directory not found, skipping")
        return []

    all_data = []
    # Find unique subjects that have raw continuous data
    mat_files = sorted(MOBILE_BCI_ROOT.glob("s*_*_*_*.mat"))

    # Group by subject - use raw continuous data from any condition
    subjects_seen = set()
    for mat_path in mat_files:
        name = mat_path.stem  # e.g. s01_scalp_ERP_0.0
        subj_id = name.split("_")[0]  # e.g. s01
        if subj_id in subjects_seen:
            continue

        logger.info(f"  MobileBCI {subj_id} ({mat_path.name})...")
        try:
            mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
            if "raw_x" not in mat or "raw_fs" not in mat:
                continue
            raw_x = np.array(mat["raw_x"], dtype=np.float32)  # (T, C)
            raw_fs = int(mat["raw_fs"])
            if raw_x.ndim < 2:
                continue
            # Transpose to (C, T)
            if raw_x.shape[0] > raw_x.shape[1]:
                data = raw_x.T
            else:
                data = raw_x

            windows = preprocess_continuous(data, raw_fs)
            if windows is not None:
                subjects_seen.add(subj_id)
                all_data.append(windows)
                logger.info(f"    -> {windows.shape[0]} windows, {windows.shape[1]}ch")
        except Exception as e:
            logger.warning(f"    Failed: {e}")

    logger.info(f"  Mobile BCI: {len(all_data)} subjects loaded")
    return all_data


def load_eesm_data() -> list[np.ndarray]:
    """Load EESM19/EESM23 BIDS .set files via MNE."""
    all_data = []
    for root, label in [(EESM19_ROOT, "EESM19"), (EESM23_ROOT, "EESM23")]:
        if not root.exists():
            logger.info(f"  {label}: directory not found, skipping")
            continue

        set_files = sorted(root.glob("sub-*/ses-*/eeg/*_eeg.set"))
        if not set_files:
            logger.info(f"  {label}: no .set files found")
            continue

        for set_file in set_files:
            # Check that companion .fdt file exists (needed by EEGLAB format)
            fdt_file = set_file.with_suffix(".fdt")
            if not fdt_file.exists():
                continue

            subj_ses = set_file.parent.parent.parent.name + "/" + set_file.parent.parent.name
            logger.info(f"  {label} {subj_ses}...")
            try:
                import mne
                raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
                # Pick only EEG channels
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
                if len(eeg_picks) < 3:
                    continue
                data = raw.get_data(picks=eeg_picks)  # (C, T) in volts
                data = data.astype(np.float32) * 1e6  # Convert to microvolts
                fs = raw.info["sfreq"]

                windows = preprocess_continuous(data, fs)
                if windows is not None:
                    all_data.append(windows)
                    logger.info(f"    -> {windows.shape[0]} windows, {windows.shape[1]}ch")
            except Exception as e:
                logger.warning(f"    Failed: {e}")

    logger.info(f"  EESM: {len(all_data)} recordings loaded")
    return all_data


def load_moabb_data() -> list[np.ndarray]:
    """Load already-downloaded MOABB datasets via the moabb API."""
    all_data = []

    # Only load datasets we know are downloaded
    MOABB_DATASETS = []
    try:
        import moabb.datasets as ds

        # Check which datasets are available by testing download status
        dataset_classes = [
            ("BNCI2014_001", ds.BNCI2014_001),
            ("BNCI2014_002", ds.BNCI2014_002),
        ]

        # Also try PhysionetMI if files exist
        physionet_dir = MNE_DATA_DIR / "MNE-eegbci-data"
        if physionet_dir.exists():
            dataset_classes.append(("PhysionetMI", ds.PhysionetMI))

        # GigaDB
        gigadb_dir = MNE_DATA_DIR / "MNE-gigadb-data"
        if gigadb_dir.exists():
            dataset_classes.append(("Cho2017", ds.Cho2017))

        MOABB_DATASETS = dataset_classes
    except ImportError:
        logger.info("  MOABB: moabb not installed, skipping")
        return []

    import mne

    for name, cls in MOABB_DATASETS:
        logger.info(f"  MOABB {name}...")
        try:
            dataset = cls()
            subjects = dataset.subject_list
            n_loaded = 0
            for subj in subjects:
                try:
                    data_dict = dataset.get_data(subjects=[subj])
                    for s_id, sessions in data_dict.items():
                        for sess_id, runs in sessions.items():
                            for run_id, raw in runs.items():
                                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
                                if len(eeg_picks) < 3:
                                    continue
                                data = raw.get_data(picks=eeg_picks).astype(np.float32)
                                # MNE data is in volts; convert to uV
                                if np.abs(data).max() < 0.01:
                                    data = data * 1e6
                                fs = raw.info["sfreq"]
                                windows = preprocess_continuous(data, fs)
                                if windows is not None:
                                    all_data.append(windows)
                                    n_loaded += 1
                except Exception as e:
                    logger.debug(f"    Subject {subj} failed: {e}")
                    continue
            logger.info(f"    {name}: {n_loaded} recordings loaded")
        except Exception as e:
            logger.warning(f"    {name} failed: {e}")

    logger.info(f"  MOABB total: {len(all_data)} recordings loaded")
    return all_data


def load_ear_saad_data() -> list[np.ndarray]:
    """Load Ear-SAAD from preprocessed HDF5 (already at 128Hz, 1-45Hz).

    Combines scalp + in-ear channels into a single array per subject.
    """
    if not EAR_SAAD_H5.exists():
        logger.info("  Ear-SAAD: broadband_46ch.h5 not found, skipping")
        return []

    import h5py

    all_data = []
    with h5py.File(str(EAR_SAAD_H5), "r") as f:
        for subj_key in sorted(f.keys()):
            grp = f[subj_key]
            scalp = np.array(grp["scalp"])  # (N, 46, 256)
            inear = np.array(grp["inear"])  # (N, 12, 256)
            # Concatenate all channels for pretraining
            combined = np.concatenate([scalp, inear], axis=1)  # (N, 58, 256)
            all_data.append(combined.astype(np.float32))
            logger.info(f"    {subj_key}: {combined.shape[0]} windows, {combined.shape[1]}ch")

    logger.info(f"  Ear-SAAD: {len(all_data)} subjects loaded")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════
# Dataset & collation
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedEEGDataset(Dataset):
    """Dataset that stores (C, T) windows from all datasets.

    Each window may have a different channel count. We sample uniformly
    across all windows regardless of source dataset.
    """

    def __init__(self, subject_windows: list[np.ndarray], max_channels: int = 0):
        """subject_windows: list of arrays, each (N_windows, C_i, T)."""
        self.segments: list[torch.Tensor] = []
        self.max_channels = max_channels
        for windows in subject_windows:
            for i in range(windows.shape[0]):
                self.segments.append(torch.from_numpy(windows[i]))  # (C, T)
                if windows.shape[1] > self.max_channels:
                    self.max_channels = windows.shape[1]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]  # (C, T) -- variable C


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
# Model architecture (scaled up from pretrain_hbn.py)
# ═══════════════════════════════════════════════════════════════════════════


class TemporalEncoder(nn.Module):
    """Per-channel temporal encoder: (1, T) -> (EMBED_DIM,).

    Shared across all channels -- learns universal EEG waveform features.
    Conv1d stack: 1 -> 32 -> 64 -> 128, ~50K parameters.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, window_size: int = WINDOW_SIZE):
        super().__init__()
        self.embed_dim = embed_dim
        # Conv stack: 256 -> 128 -> 64 -> 32 time steps -> 1
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # -> 128
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # -> 64
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=5, stride=2, padding=2),  # -> 32
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
        self.window_size = window_size
        self.project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 32),
            nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                embed_dim, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),  # 32 -> 64
            nn.GELU(),
            nn.ConvTranspose1d(
                64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),  # 64 -> 128
            nn.GELU(),
            nn.ConvTranspose1d(
                32, 1, kernel_size=7, stride=2, padding=3, output_padding=1
            ),  # 128 -> 256
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*C, embed_dim) -> (B*C, 1, T)."""
        h = self.project(x)  # (B*C, embed_dim * 32)
        h = h.view(-1, self.embed_dim, 32)  # (B*C, embed_dim, 32)
        out = self.deconv(h)  # (B*C, 1, ~T)
        # Trim or pad to exact window size
        if out.shape[-1] > self.window_size:
            out = out[:, :, : self.window_size]
        elif out.shape[-1] < self.window_size:
            out = F.pad(out, (0, self.window_size - out.shape[-1]))
        return out


class SpatialTransformer(nn.Module):
    """Transformer that operates on channel embeddings.

    Takes visible channel embeddings + mask tokens, predicts all channel embeddings.
    Channel-agnostic: no positional encoding since channels vary across datasets.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
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
        vis_mask = ~mask  # True = visible
        for b in range(B):
            vis_idx = vis_mask[b].nonzero(as_tuple=True)[0]
            all_embeds[b, vis_idx] = visible_embeds[b, : vis_idx.shape[0]]

        # Create attention mask for padding (padded positions should not attend)
        # padding_mask: (B, C_total) -- True means IGNORE this position
        # (positions beyond actual channel count are padding)
        padding_mask = mask.clone()
        # Actually, padding channels ARE masked, and we want the transformer
        # to process all real channels (visible + masked). So padding_mask
        # should only mask out actual padding (beyond channel count).
        # This is handled in the forward of MaskedChannelAutoencoder.

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

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        mask_ratio: float = MASK_RATIO,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(embed_dim)
        self.decoder = TemporalDecoder(embed_dim)
        self.spatial = SpatialTransformer(embed_dim, n_heads, n_layers)
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
        x_flat = data.reshape(B * C_max, 1, T)
        embeds_flat = self.encoder(x_flat)  # (B*C_max, embed_dim)
        embeds = embeds_flat.reshape(B, C_max, self.embed_dim)

        # 2. Create channel mask (respecting actual channel counts)
        mask = torch.ones(B, C_max, dtype=torch.bool, device=device)  # True = masked
        for b in range(B):
            C = channel_counts[b].item()
            n_mask = max(1, int(C * self.mask_ratio))
            n_vis = C - n_mask
            perm = torch.randperm(C, device=device)
            vis_idx = perm[:n_vis]
            mask[b, vis_idx] = False

        # 3. Gather visible embeddings
        vis_mask = ~mask
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
    """Train the masked autoencoder with cosine LR and gradient clipping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    patience_counter = 0

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Training for {epochs} epochs on {device}")
    logger.info(f"Architecture: embed_dim={model.embed_dim}, "
                f"spatial={N_LAYERS}L/{N_HEADS}H, mask_ratio={model.mask_ratio}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for batch_idx, (data, counts) in enumerate(train_loader):
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
                f"train_L1={train_loss:.4f} | val_L1={val_loss:.4f} | "
                f"best={best_val_loss:.4f} | lr={lr_now:.2e} | "
                f"{elapsed:.1f}s{marker}"
            )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Save best model
    if best_state is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract encoder state dict
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in best_state.items()
            if k.startswith("encoder.")
        }

        checkpoint = {
            "model_state_dict": best_state,
            "encoder_state_dict": encoder_state,
            "embed_dim": model.embed_dim,
            "mask_ratio": model.mask_ratio,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch,
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
    parser = argparse.ArgumentParser(
        description="Unified pretraining on all EEG datasets"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default=str(SAVE_PATH))
    parser.add_argument(
        "--skip-datasets",
        type=str,
        nargs="*",
        default=[],
        help="Datasets to skip (hbn, mobile_bci, eesm, moabb, ear_saad)",
    )
    parser.add_argument(
        "--max-subjects-per-dataset",
        type=int,
        default=50,
        help="Max subjects to load per dataset (default 50 to fit in RAM)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Load all datasets ──────────────────────────────────────────────────
    all_windows: list[np.ndarray] = []
    dataset_stats: dict[str, dict] = {}
    skip = set(args.skip_datasets)

    loaders = [
        ("hbn", "HBN-EEG", load_hbn_data),
        ("mobile_bci", "Mobile BCI Ear", load_mobile_bci_data),
        ("eesm", "EESM19/23", load_eesm_data),
        ("moabb", "MOABB", load_moabb_data),
        ("ear_saad", "Ear-SAAD", load_ear_saad_data),
    ]

    for key, name, loader_fn in loaders:
        if key in skip:
            logger.info(f"Skipping {name}")
            continue
        logger.info(f"Loading {name}...")
        t0 = time.time()
        data = loader_fn()
        dt = time.time() - t0

        # Limit subjects per dataset to fit in RAM
        max_subj = args.max_subjects_per_dataset
        if data and len(data) > max_subj:
            logger.info(f"  Limiting {name} from {len(data)} to {max_subj} subjects")
            data = data[:max_subj]

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

    # ── Summary ────────────────────────────────────────────────────────────
    total_windows = sum(d.shape[0] for d in all_windows)
    total_subjects = len(all_windows)
    all_ch_counts = [d.shape[1] for d in all_windows]
    logger.info(f"\n{'='*70}")
    logger.info(f"UNIFIED DATASET SUMMARY")
    logger.info(f"  Total: {total_subjects} recordings, {total_windows} windows")
    logger.info(f"  Channel counts: {min(all_ch_counts)}-{max(all_ch_counts)}")
    for name, stats in dataset_stats.items():
        logger.info(f"  {name}: {stats['subjects']} subj, {stats['windows']} win, {stats['channels']}ch")
    logger.info(f"{'='*70}\n")

    # ── Split by SUBJECT (not by window) to measure real generalization ───
    n_subjects = len(all_windows)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_subjects)
    n_val_subj = max(1, int(n_subjects * 0.15))
    val_indices = set(indices[:n_val_subj])
    train_windows = [all_windows[i] for i in range(n_subjects) if i not in val_indices]
    val_windows = [all_windows[i] for i in range(n_subjects) if i in val_indices]
    logger.info(f"Subject split: {n_subjects - n_val_subj} train subjects, {n_val_subj} val subjects")

    train_ds = UnifiedEEGDataset(train_windows)
    val_ds = UnifiedEEGDataset(val_windows)
    logger.info(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    del all_windows, train_windows, val_windows

    num_workers = 0  # Workers fork entire dataset (~10GB), causing OOM with 178K windows
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_channels,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_channels,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    # ── Build model ────────────────────────────────────────────────────────
    model = MaskedChannelAutoencoder(
        embed_dim=EMBED_DIM,
        mask_ratio=MASK_RATIO,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    ).to(device)

    # ── Train ──────────────────────────────────────────────────────────────
    save_path = Path(args.save_path)
    best_loss = train(
        model, train_loader, val_loader, args.epochs, args.lr, device, save_path
    )

    logger.info(f"\nPretraining complete. Best val L1: {best_loss:.4f}")
    logger.info(f"Encoder saved to: {save_path}")
    logger.info(
        "To load the encoder for fine-tuning:\n"
        f"  ckpt = torch.load('{save_path}')\n"
        f"  encoder = TemporalEncoder(embed_dim=ckpt['embed_dim'])\n"
        "  encoder.load_state_dict(ckpt['encoder_state_dict'])"
    )


if __name__ == "__main__":
    main()
