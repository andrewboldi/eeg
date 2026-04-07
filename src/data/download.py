"""Download scripts for EEG datasets."""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# 10-20 channel names used throughout the project (21 scalp channels)
SCALP_CHANNELS_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2", "A1",
]

IN_EAR_CHANNELS = ["EarL1", "EarL2", "EarR1", "EarR2"]


def download_file(url: str, dest: Path, *, chunk_size: int = 8192) -> Path:
    """Download a file with progress reporting."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("File already exists: %s", dest)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "eeg-downsampling/0.1"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        shutil.move(str(tmp), str(dest))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def download_dataset_a(raw_dir: str | Path) -> Path:
    """Download Dataset A: KU Leuven auditory attention decoding.

    URL: https://zenodo.org/records/3997352
    Returns the directory containing the downloaded data.
    """
    raw_dir = Path(raw_dir)
    dest_dir = raw_dir / "dataset_a"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Zenodo record 3997352 — the main ZIP archive
    zip_url = "https://zenodo.org/records/3997352/files/data.zip"
    zip_path = dest_dir / "data.zip"

    if not any(dest_dir.glob("*.bdf")) and not any(dest_dir.glob("**/*.bdf")):
        download_file(zip_url, zip_path)
        logger.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        logger.info("Extraction complete")
    else:
        logger.info("Dataset A already extracted in %s", dest_dir)

    return dest_dir


def download_dataset_b(raw_dir: str | Path) -> Path:
    """Download Dataset B: DTU in-ear EEG (Kappel et al. 2019).

    URL: https://zenodo.org/records/2647551
    """
    raw_dir = Path(raw_dir)
    dest_dir = raw_dir / "dataset_b"
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_url = "https://zenodo.org/records/2647551/files/data.zip"
    zip_path = dest_dir / "data.zip"

    if not any(dest_dir.iterdir()) or (dest_dir / "data.zip").exists():
        try:
            download_file(zip_url, zip_path)
            logger.info("Extracting %s", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
        except Exception as e:
            logger.warning("Dataset B download failed: %s. Use synthetic data instead.", e)

    return dest_dir


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download EEG datasets")
    parser.add_argument("--dataset", choices=["a", "b", "all"], default="all")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    args = parser.parse_args()

    if args.dataset in ("a", "all"):
        download_dataset_a(args.raw_dir)
    if args.dataset in ("b", "all"):
        download_dataset_b(args.raw_dir)
