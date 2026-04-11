"""Broadband benchmark for scalp-to-in-ear EEG prediction.

Same LOSO protocol as the narrowband benchmark, but using broadband data
(1-45 Hz at 128 Hz) instead of narrowband (1-9 Hz at 20 Hz).

Test set: Subjects 13, 14, 15
Metric: Mean Pearson r across 3 test subjects × 12 in-ear channels

Usage:
    uv run python scripts/benchmark_broadband.py --baseline
    uv run python scripts/benchmark_broadband.py --model-fn models/iter_XXX.py --name iter_XXX
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.metrics.evaluation import compute_all_metrics
from src.models import ClosedFormLinear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# === TEST CONFIGURATION ===
TEST_SUBJECTS = [13, 14, 15]
TRAIN_SUBJECTS = list(range(1, 13))
DATA_PATH = Path("data/processed/broadband.h5")
LEADERBOARD_PATH = Path("results/benchmark/leaderboard_broadband.jsonl")
# ==========================


def load_all_subjects():
    """Load broadband data from HDF5."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: uv run python scripts/preprocess_broadband.py"
        )

    subject_data = {}
    with h5py.File(DATA_PATH, "r") as f:
        fs = f.attrs["fs"]
        C_scalp = f.attrs["C_scalp"]
        C_inear = f.attrs["C_inear"]
        logger.info(f"Broadband data: {C_scalp} scalp -> {C_inear} in-ear, fs={fs} Hz")

        for subj in range(1, 16):
            key = f"subject_{subj:02d}"
            if key in f:
                grp = f[key]
                scalp = grp["scalp"][:]
                inear = grp["inear"][:]
                subject_data[subj] = (scalp, inear)
                logger.info(f"  Subject {subj}: {scalp.shape[0]} windows")

    return subject_data, int(C_scalp), int(C_inear), float(fs)


def evaluate_model(model, test_ds, device, fs):
    """Evaluate model on test data."""
    model.eval()
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    all_pred, all_target = [], []
    with torch.no_grad():
        for scalp, inear in loader:
            pred = model(scalp.to(device))
            all_pred.append(pred.cpu().numpy())
            all_target.append(inear.numpy())

    pred = np.concatenate(all_pred)
    target = np.concatenate(all_target)
    metrics = compute_all_metrics(pred, target, fs)

    return {
        "pearson_r": float(metrics["pearson_r"].mean()),
        "pearson_r_per_ch": metrics["pearson_r"].tolist(),
        "rmse": float(metrics["rmse"].mean()),
        "snr_db": float(metrics["snr_db"].mean()),
    }


def run_baseline(subject_data, C_scalp, C_inear, fs):
    """Run closed-form baseline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results = []

    for held_out in TEST_SUBJECTS:
        if held_out not in subject_data:
            continue

        train_scalp = np.concatenate([s for k, (s, _) in subject_data.items()
                                       if k != held_out and k in TRAIN_SUBJECTS])
        train_inear = np.concatenate([i for k, (_, i) in subject_data.items()
                                       if k != held_out and k in TRAIN_SUBJECTS])
        test_scalp, test_inear = subject_data[held_out]

        train_ds = EEGDataset(train_scalp, train_inear)
        test_ds = EEGDataset(test_scalp, test_inear)

        cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
        cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
        cf = cf.to(device)

        metrics = evaluate_model(cf, test_ds, device, fs)
        fold_results.append(metrics)
        logger.info(f"  Subject {held_out}: r={metrics['pearson_r']:.4f}")

    mean_r = np.mean([f["pearson_r"] for f in fold_results])
    std_r = np.std([f["pearson_r"] for f in fold_results])
    mean_snr = np.mean([f["snr_db"] for f in fold_results])

    return {
        "model": "closed_form_broadband",
        "mean_r": float(mean_r),
        "std_r": float(std_r),
        "mean_snr": float(mean_snr),
    }


def run_model_fn(model_fn_path, subject_data, C_scalp, C_inear, fs):
    """Run a model function file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = importlib.util.spec_from_file_location("model_fn", model_fn_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fold_results = []
    for held_out in TEST_SUBJECTS:
        if held_out not in subject_data:
            continue

        train_scalp = np.concatenate([s for k, (s, _) in subject_data.items()
                                       if k != held_out and k in TRAIN_SUBJECTS])
        train_inear = np.concatenate([i for k, (_, i) in subject_data.items()
                                       if k != held_out and k in TRAIN_SUBJECTS])
        test_scalp, test_inear = subject_data[held_out]

        # Split train into train/val (85/15)
        n = len(train_scalp)
        n_val = int(0.15 * n)
        val_scalp, val_inear = train_scalp[-n_val:], train_inear[-n_val:]
        train_scalp, train_inear = train_scalp[:-n_val], train_inear[:-n_val]

        train_ds = EEGDataset(train_scalp, train_inear)
        val_ds = EEGDataset(val_scalp, val_inear)
        test_ds = EEGDataset(test_scalp, test_inear)

        logger.info(f"Training model for subject {held_out} "
                     f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})")

        model = mod.build_and_train(train_ds, val_ds, C_scalp, C_inear, device)
        model = model.to(device)
        metrics = evaluate_model(model, test_ds, device, fs)
        fold_results.append(metrics)
        logger.info(f"  Subject {held_out}: r={metrics['pearson_r']:.4f}")

    mean_r = np.mean([f["pearson_r"] for f in fold_results])
    std_r = np.std([f["pearson_r"] for f in fold_results])
    mean_snr = np.mean([f["snr_db"] for f in fold_results])

    return {
        "mean_r": float(mean_r),
        "std_r": float(std_r),
        "mean_snr": float(mean_snr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--model-fn", type=str)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    subject_data, C_scalp, C_inear, fs = load_all_subjects()

    if args.baseline:
        results = run_baseline(subject_data, C_scalp, C_inear, fs)
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    elif args.model_fn:
        logger.info(f"Training model from {args.model_fn}...")
        results = run_model_fn(args.model_fn, subject_data, C_scalp, C_inear, fs)
        results["model"] = args.name or Path(args.model_fn).stem
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        parser.error("Specify --baseline or --model-fn")

    logger.info(f"Result: mean_r={results['mean_r']:.4f} +/- {results['std_r']:.4f}, "
                f"SNR={results['mean_snr']:.2f} dB")

    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_PATH, "a") as f:
        f.write(json.dumps(results) + "\n")
    logger.info(f"Appended to {LEADERBOARD_PATH}")


if __name__ == "__main__":
    main()
