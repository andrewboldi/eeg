"""Download and manage EEG datasets.

Provides ``ensure_dataset(name)`` which checks for local data first and
auto-downloads if missing.  Supports Zenodo, Figshare, and OpenNeuro sources.

Usage::

    from src.data.download import ensure_dataset

    path = ensure_dataset("ear_saad")       # core dataset
    path = ensure_dataset("hbn_eeg")        # pretraining
    path = ensure_dataset("all")            # download everything

CLI::

    uv run python -m src.data.download --dataset ear_saad
    uv run python -m src.data.download --dataset all --raw-dir data/raw
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")

# 10-20 channel names used throughout the project (21 scalp channels)
SCALP_CHANNELS_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2", "A1",
]

IN_EAR_CHANNELS = ["EarL1", "EarL2", "EarR1", "EarR2"]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "ear_saad": {
        "source": "zenodo",
        "url": "https://zenodo.org/records/16536441/files/ear_saad_dataset.zip",
        "check_glob": "bids_dataset/sub-01",
        "description": "Ear-SAAD (Geirnaert 2025): 15 subj, 27 scalp + 19 around-ear + 12 in-ear",
    },
    "mobile_bci_ear": {
        "source": "figshare",
        "url": "https://figshare.com/ndownloader/articles/13604078/versions/1",
        "zip_name": "mobile_bci_ear.zip",
        "check_glob": "s01_ear_ERP_0.0.mat",
        "description": "Mobile BCI (Lee 2021): 24 subj, 32 scalp + 14 around-ear cEEGrid",
    },
    "hbn_eeg": {
        "source": "openneuro",
        "dataset_id": "ds005505",
        "check_glob": "NDAR*",
        "description": "HBN-EEG: large developmental EEG dataset",
    },
    "eesm19": {
        "source": "openneuro",
        "dataset_id": "ds005185",
        "check_glob": "dataset_description.json",
        "description": "EESM19 (Aarhus 2025): 20 subj, ear-EEG sleep",
    },
    "eesm23": {
        "source": "openneuro",
        "dataset_id": "ds005178",
        "check_glob": "sub-001",
        "description": "EESM23 (Aarhus 2025): 10 subj, ear-EEG sleep",
    },
    "dataset_a": {
        "source": "zenodo",
        "url": "https://zenodo.org/records/3997352/files/data.zip",
        "check_glob": "**/*.bdf",
        "description": "KU Leuven AAD (Das 2019): 16 subj, 64 scalp",
    },
    "dataset_b": {
        "source": "zenodo",
        "url": "https://zenodo.org/records/2647551/files/data.zip",
        "check_glob": "**/*.mat",
        "description": "DTU in-ear EEG (Kappel 2019)",
    },
}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


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
            total = resp.headers.get("Content-Length")
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / int(total) * 100
                    if downloaded % (1024 * 1024 * 50) < chunk_size:  # log every ~50MB
                        logger.info("  %.0f%% (%d / %s MB)", pct, downloaded >> 20, int(total) >> 20)
        shutil.move(str(tmp), str(dest))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file, handling nested single-directory archives."""
    logger.info("Extracting %s -> %s", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    # If the zip contained a single top-level directory, move contents up
    children = [p for p in dest_dir.iterdir() if p.name != zip_path.name]
    if len(children) == 1 and children[0].is_dir():
        nested = children[0]
        for item in nested.iterdir():
            target = dest_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        nested.rmdir()
    logger.info("Extraction complete")


def _run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, logging it first."""
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def _has_aws_cli() -> bool:
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ---------------------------------------------------------------------------
# Source-specific downloaders
# ---------------------------------------------------------------------------


def _download_zenodo(dest_dir: Path, info: dict) -> None:
    """Download and extract a Zenodo zip archive."""
    zip_name = info.get("zip_name", info["url"].split("/")[-1])
    zip_path = dest_dir / zip_name
    download_file(info["url"], zip_path)
    _extract_zip(zip_path, dest_dir)
    zip_path.unlink()
    logger.info("Removed zip: %s", zip_path)


def _download_figshare(dest_dir: Path, info: dict) -> None:
    """Download from Figshare (follows redirects)."""
    zip_name = info.get("zip_name", "download.zip")
    zip_path = dest_dir / zip_name
    download_file(info["url"], zip_path)
    _extract_zip(zip_path, dest_dir)
    zip_path.unlink()
    logger.info("Removed zip: %s", zip_path)


def _download_openneuro(dest_dir: Path, info: dict) -> None:
    """Download from OpenNeuro via AWS S3 (no auth) or git clone fallback."""
    dataset_id = info["dataset_id"]
    if _has_aws_cli():
        _run_cmd([
            "aws", "s3", "sync", "--no-sign-request",
            f"s3://openneuro.org/{dataset_id}",
            str(dest_dir),
        ])
    else:
        git_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"
        logger.info("aws CLI not found, falling back to git clone")
        _run_cmd(["git", "clone", "--depth", "1", git_url, str(dest_dir)])


_DOWNLOADERS = {
    "zenodo": _download_zenodo,
    "figshare": _download_figshare,
    "openneuro": _download_openneuro,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_downloaded(name: str, raw_dir: Path = DEFAULT_RAW_DIR) -> bool:
    """Check whether a dataset already exists locally."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name!r}. Known: {list(DATASETS)}")
    info = DATASETS[name]
    dest_dir = raw_dir / name
    if not dest_dir.exists():
        return False
    return bool(list(dest_dir.glob(info["check_glob"])))


def ensure_dataset(
    name: str,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> Path:
    """Ensure a dataset exists locally, downloading if needed.

    Parameters
    ----------
    name : str
        Dataset name (key in DATASETS), or "all" to download everything.
    raw_dir : Path
        Root directory for raw data (default: ``data/raw``).

    Returns
    -------
    Path
        Directory containing the dataset.
    """
    raw_dir = Path(raw_dir)

    if name == "all":
        for ds_name in DATASETS:
            ensure_dataset(ds_name, raw_dir)
        return raw_dir

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name!r}. Known: {list(DATASETS)}")

    info = DATASETS[name]
    dest_dir = raw_dir / name

    if is_downloaded(name, raw_dir):
        logger.info("[%s] Already downloaded at %s", name, dest_dir)
        return dest_dir

    logger.info("[%s] Not found locally — downloading: %s", name, info["description"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloader = _DOWNLOADERS[info["source"]]
    downloader(dest_dir, info)

    if not is_downloaded(name, raw_dir):
        logger.warning("[%s] Download completed but check_glob %r not found in %s",
                       name, info["check_glob"], dest_dir)
    else:
        logger.info("[%s] Download verified at %s", name, dest_dir)

    return dest_dir


# Legacy aliases for backwards compatibility
def download_dataset_a(raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    return ensure_dataset("dataset_a", raw_dir)

def download_dataset_b(raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    return ensure_dataset("dataset_b", raw_dir)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download EEG datasets")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--check-only", action="store_true", help="Only check status, don't download")
    args = parser.parse_args()

    if args.check_only:
        for ds_name, ds_info in DATASETS.items():
            status = "OK" if is_downloaded(ds_name, Path(args.raw_dir)) else "MISSING"
            print(f"  [{status:>7s}] {ds_name:<20s} — {ds_info['description']}")
    elif args.dataset == "all":
        ensure_dataset("all", args.raw_dir)
    else:
        ensure_dataset(args.dataset, args.raw_dir)
