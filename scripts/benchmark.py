"""Fixed benchmark for autoresearch loop.

IMMUTABLE TEST PROTOCOL — DO NOT MODIFY THIS FILE.
Any changes to the test set or evaluation protocol invalidate all prior results.

Test set: Subjects 13, 14, 15 (LOSO — train on 1-12, test on each held-out)
Metric: Mean Pearson r across 3 held-out subjects × 12 in-ear channels
Secondary: SNR (dB), per-channel r, per-subject r

Usage:
    uv run python scripts/benchmark.py --model-fn path/to/model_fn.py
    uv run python scripts/benchmark.py --baseline  # run closed-form baseline
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.metrics.evaluation import compute_all_metrics
from src.models import ClosedFormLinear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# === FIXED TEST CONFIGURATION — DO NOT CHANGE ===
TEST_SUBJECTS = [13, 14, 15]
TRAIN_SUBJECTS = list(range(1, 13))  # 1-12
WINDOW_SIZE = 40   # 2 seconds at 20 Hz
STRIDE = 20        # 1 second overlap
FS = 20.0
# ================================================


def load_all_subjects():
    """Load and preprocess all subjects. Returns dict[subject_id] -> (scalp_windows, inear_windows)."""
    from scripts.real_data_experiment import load_subject, window_trials

    subject_data = {}
    for subj in range(1, 16):
        try:
            scalp_trials, inear_trials, scalp_ch, inear_ch, fs = load_subject(subj)
            scalp_w, inear_w = window_trials(scalp_trials, inear_trials,
                                              window_size=WINDOW_SIZE, stride=STRIDE)
            if scalp_w.shape[0] > 0:
                subject_data[subj] = (scalp_w, inear_w)
        except Exception as e:
            logger.warning("Subject %d failed: %s", subj, e)
    return subject_data


def get_train_test_split(subject_data):
    """Split into train (subjects 1-12) and test folds (subjects 13, 14, 15)."""
    train_scalp = np.concatenate([subject_data[s][0] for s in TRAIN_SUBJECTS if s in subject_data])
    train_inear = np.concatenate([subject_data[s][1] for s in TRAIN_SUBJECTS if s in subject_data])

    # Validation: last 10% of training data
    n_train = int(0.9 * train_scalp.shape[0])
    val_scalp, val_inear = train_scalp[n_train:], train_inear[n_train:]
    train_scalp, train_inear = train_scalp[:n_train], train_inear[:n_train]

    train_ds = EEGDataset(train_scalp, train_inear)
    val_ds = EEGDataset(val_scalp, val_inear)

    test_folds = {}
    for subj in TEST_SUBJECTS:
        if subj in subject_data:
            test_folds[subj] = EEGDataset(subject_data[subj][0], subject_data[subj][1])

    return train_ds, val_ds, test_folds


def evaluate_on_test(model, test_folds, device):
    """Evaluate model on each test subject. Returns benchmark results dict."""
    model.eval()
    fold_results = []

    for subj, test_ds in sorted(test_folds.items()):
        loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        all_pred, all_target = [], []
        with torch.no_grad():
            for scalp, inear in loader:
                pred = model(scalp.to(device))
                all_pred.append(pred.cpu().numpy())
                all_target.append(inear.numpy())

        pred = np.concatenate(all_pred)
        target = np.concatenate(all_target)
        metrics = compute_all_metrics(pred, target, FS)

        fold_results.append({
            "subject": subj,
            "pearson_r": float(metrics["pearson_r"].mean()),
            "pearson_r_per_ch": metrics["pearson_r"].tolist(),
            "snr_db": float(metrics["snr_db"].mean()),
            "rmse": float(metrics["rmse"].mean()),
        })
        logger.info("  Subject %d: r=%.4f, SNR=%.2f dB", subj,
                     fold_results[-1]["pearson_r"], fold_results[-1]["snr_db"])

    # Aggregate
    mean_r = np.mean([f["pearson_r"] for f in fold_results])
    std_r = np.std([f["pearson_r"] for f in fold_results])
    mean_snr = np.mean([f["snr_db"] for f in fold_results])

    results = {
        "benchmark_version": "1.0",
        "test_subjects": TEST_SUBJECTS,
        "train_subjects": TRAIN_SUBJECTS,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "fs": FS,
        "folds": fold_results,
        "mean_r": float(mean_r),
        "std_r": float(std_r),
        "mean_snr": float(mean_snr),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info("=== BENCHMARK: r=%.4f +/- %.4f, SNR=%.2f dB ===", mean_r, std_r, mean_snr)
    return results


def run_baseline():
    """Run closed-form baseline on the fixed test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_data = load_all_subjects()
    train_ds, val_ds, test_folds = get_train_test_split(subject_data)

    C_scalp = train_ds.scalp.shape[1]
    C_inear = train_ds.inear.shape[1]

    model = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    model.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    model = model.to(device)

    results = evaluate_on_test(model, test_folds, device)
    results["model"] = "closed_form_baseline"
    return results


def run_model_fn(model_fn_path: str):
    """Run a custom model function on the fixed test set.

    The model_fn_path should point to a Python file with a function:
        def build_and_train(train_ds, val_ds, C_scalp, C_inear, device) -> nn.Module

    This function receives the training data and must return a trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_data = load_all_subjects()
    train_ds, val_ds, test_folds = get_train_test_split(subject_data)

    C_scalp = train_ds.scalp.shape[1]
    C_inear = train_ds.inear.shape[1]

    # Load model function
    spec = importlib.util.spec_from_file_location("model_fn", model_fn_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    logger.info("Training model from %s...", model_fn_path)
    model = mod.build_and_train(train_ds, val_ds, C_scalp, C_inear, device)
    model = model.to(device)

    results = evaluate_on_test(model, test_folds, device)
    results["model"] = Path(model_fn_path).stem
    return results


def save_results(results: dict, name: str):
    """Save benchmark results."""
    out_dir = Path("results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", path)

    # Append to leaderboard
    leaderboard_path = out_dir / "leaderboard.jsonl"
    with open(leaderboard_path, "a") as f:
        entry = {
            "model": results["model"],
            "mean_r": results["mean_r"],
            "std_r": results["std_r"],
            "mean_snr": results["mean_snr"],
            "timestamp": results["timestamp"],
        }
        f.write(json.dumps(entry) + "\n")
    logger.info("Appended to leaderboard")


def main():
    parser = argparse.ArgumentParser(description="Fixed benchmark evaluation")
    parser.add_argument("--baseline", action="store_true", help="Run closed-form baseline")
    parser.add_argument("--model-fn", type=str, help="Path to model function file")
    parser.add_argument("--name", type=str, default=None, help="Name for results file")
    args = parser.parse_args()

    if args.baseline:
        results = run_baseline()
        save_results(results, args.name or "baseline_closed_form")
    elif args.model_fn:
        results = run_model_fn(args.model_fn)
        save_results(results, args.name or Path(args.model_fn).stem)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
