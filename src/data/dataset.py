"""PyTorch Dataset for paired scalp / in-ear EEG windows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """Dataset of (scalp, in-ear) EEG window pairs.

    Each sample is a tuple:
        scalp: (C_in, T)  float32 tensor
        inear: (C_out, T) float32 tensor
    """

    def __init__(
        self,
        scalp: np.ndarray,
        inear: np.ndarray,
    ):
        """
        Args:
            scalp: (N, C_in, T)
            inear: (N, C_out, T)
        """
        assert scalp.shape[0] == inear.shape[0], "Mismatched number of windows"
        assert scalp.shape[2] == inear.shape[2], "Mismatched window lengths"
        self.scalp = torch.as_tensor(scalp, dtype=torch.float32)
        self.inear = torch.as_tensor(inear, dtype=torch.float32)

    def __len__(self) -> int:
        return self.scalp.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scalp[idx], self.inear[idx]

    @classmethod
    def from_hdf5(cls, path: str | Path, split: str = "train") -> EEGDataset:
        """Load dataset from HDF5 file.

        Expected structure:
            /{split}/scalp  -> (N, C_in, T)
            /{split}/inear  -> (N, C_out, T)
        """
        path = Path(path)
        with h5py.File(path, "r") as f:
            grp = f[split]
            scalp = grp["scalp"][:]
            inear = grp["inear"][:]
        return cls(scalp, inear)


def save_to_hdf5(
    path: str | Path,
    split: str,
    scalp: np.ndarray,
    inear: np.ndarray,
    **meta: np.ndarray,
) -> None:
    """Save preprocessed data to HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        grp = f.require_group(split)
        for key in ("scalp", "inear"):
            if key in grp:
                del grp[key]
        grp.create_dataset("scalp", data=scalp, compression="gzip")
        grp.create_dataset("inear", data=inear, compression="gzip")
        for k, v in meta.items():
            if k in grp:
                del grp[k]
            grp.create_dataset(k, data=v)
    logger.info("Saved %s/%s: scalp %s, inear %s", path, split, scalp.shape, inear.shape)


def make_splits(
    scalp: np.ndarray,
    inear: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    *,
    chronological: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split windowed data into train/val/test.

    Uses chronological ordering (no shuffling) to prevent temporal leakage.

    Returns:
        Dict mapping split name to (scalp, inear) arrays.
    """
    n = scalp.shape[0]
    if chronological:
        idx = np.arange(n)
    else:
        idx = np.random.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": (scalp[idx[:n_train]], inear[idx[:n_train]]),
        "val": (scalp[idx[n_train : n_train + n_val]], inear[idx[n_train : n_train + n_val]]),
        "test": (scalp[idx[n_train + n_val :]], inear[idx[n_train + n_val :]]),
    }

    for name, (s, i) in splits.items():
        logger.info("Split %s: %d windows", name, s.shape[0])

    return splits
