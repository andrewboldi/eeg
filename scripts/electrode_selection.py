"""Electrode subset selection via greedy backward elimination.

Determines the minimum number of scalp electrodes needed to maintain
in-ear prediction quality above a threshold.

Usage:
    uv run python scripts/electrode_selection.py
    uv run python scripts/electrode_selection.py --threshold 0.80
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.data.download import SCALP_CHANNELS_10_20
from src.data.preprocess import preprocess_raw
from src.data.synthetic import generate_synthetic_data
from src.models import ClosedFormLinear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHANNEL_NAMES = SCALP_CHANNELS_10_20  # 21 channels


def evaluate_subset(train_scalp, train_inear, test_scalp, test_inear, channel_indices):
    """Evaluate closed-form model using only the given channel subset."""
    # Subset channels
    train_sub = train_scalp[:, channel_indices, :]
    test_sub = test_scalp[:, channel_indices, :]

    C_in = len(channel_indices)
    model = ClosedFormLinear(C_in=C_in, C_out=4)
    model.fit(train_sub, train_inear)

    # Predict
    import torch
    with torch.no_grad():
        pred = model(torch.tensor(test_sub, dtype=torch.float32))
    pred = pred.numpy()

    # Pearson correlation per channel, then mean
    n_windows, n_out, T = pred.shape
    correlations = []
    for c in range(n_out):
        p = pred[:, c, :].flatten()
        t = test_inear[:, c, :].flatten()
        r = np.corrcoef(p, t)[0, 1]
        correlations.append(r)

    return float(np.mean(correlations))


def greedy_backward_elimination(train_scalp, train_inear, test_scalp, test_inear,
                                 threshold=0.80):
    """Remove channels one at a time, keeping the least impactful removal."""
    remaining = list(range(21))
    history = []

    # Baseline with all channels
    baseline_r = evaluate_subset(train_scalp, train_inear, test_scalp, test_inear, remaining)
    logger.info("Baseline (all %d channels): r=%.4f", len(remaining), baseline_r)
    history.append({
        "n_channels": len(remaining),
        "channels": [CHANNEL_NAMES[i] for i in remaining],
        "channel_indices": remaining.copy(),
        "r": baseline_r,
        "removed": None,
    })

    while len(remaining) > 1:
        best_r = -1.0
        best_remove = None

        for ch in remaining:
            subset = [c for c in remaining if c != ch]
            r = evaluate_subset(train_scalp, train_inear, test_scalp, test_inear, subset)
            if r > best_r:
                best_r = r
                best_remove = ch

        remaining.remove(best_remove)
        logger.info("Removed %s (idx %d) -> %d channels, r=%.4f",
                     CHANNEL_NAMES[best_remove], best_remove, len(remaining), best_r)

        history.append({
            "n_channels": len(remaining),
            "channels": [CHANNEL_NAMES[i] for i in remaining],
            "channel_indices": remaining.copy(),
            "r": best_r,
            "removed": CHANNEL_NAMES[best_remove],
        })

        if best_r < threshold:
            logger.info("Dropped below threshold r=%.2f at %d channels", threshold, len(remaining))
            break

    return history


def run_electrode_selection(threshold=0.80, n_subjects=20, n_samples=153600):
    """Run the full electrode selection experiment."""
    logger.info("Generating data...")
    subjects = generate_synthetic_data(
        n_subjects=n_subjects, n_samples=n_samples, fs=256.0, snr_db=10.0,
    )

    # Preprocess
    all_scalp, all_inear = [], []
    for subj in subjects:
        result = preprocess_raw(subj["scalp"], subj["inear"],
                                fs=256.0, target_fs=256.0, window_size=256, stride=128)
        if result["scalp"].shape[0] > 0:
            all_scalp.append(result["scalp"])
            all_inear.append(result["inear"])

    scalp = np.concatenate(all_scalp)
    inear = np.concatenate(all_inear)

    # 80/20 split
    n = scalp.shape[0]
    n_train = int(0.8 * n)
    train_scalp, test_scalp = scalp[:n_train], scalp[n_train:]
    train_inear, test_inear = inear[:n_train], inear[n_train:]

    logger.info("Train: %d windows, Test: %d windows", n_train, n - n_train)

    # Run backward elimination
    history = greedy_backward_elimination(
        train_scalp, train_inear, test_scalp, test_inear, threshold=threshold
    )

    # Find minimum channels above threshold
    above_threshold = [h for h in history if h["r"] >= threshold]
    if above_threshold:
        best = min(above_threshold, key=lambda h: h["n_channels"])
        logger.info("=== Minimum channels above r=%.2f: %d channels ===",
                     threshold, best["n_channels"])
        logger.info("  Channels: %s", ", ".join(best["channels"]))
        logger.info("  r = %.4f", best["r"])
    else:
        logger.info("No subset met the threshold r=%.2f", threshold)

    # Save results
    out_dir = Path("results/electrode_selection")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "threshold": threshold,
        "history": history,
        "minimum_channels": best["n_channels"] if above_threshold else None,
        "minimum_channel_names": best["channels"] if above_threshold else None,
        "minimum_r": best["r"] if above_threshold else None,
    }

    with open(out_dir / "elimination_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_dir / "elimination_results.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="Electrode subset selection")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Minimum r to maintain")
    args = parser.parse_args()
    run_electrode_selection(threshold=args.threshold)


if __name__ == "__main__":
    main()
