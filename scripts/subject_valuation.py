"""Subject valuation: measure each training subject's marginal value.

For each training subject i (1-12):
  1. Train CF on subjects {1..12} \ {i}
  2. Evaluate on test subjects 13, 14, 15
  3. Record mean_r without subject i

Marginal value = mean_r_full - mean_r_without_i
Negative value means removing that subject IMPROVES performance.

Uses broadband_46ch.h5 data and CF baseline (~15s per LOO experiment).
"""

from __future__ import annotations

import json, logging, time
from pathlib import Path

import h5py, numpy as np, torch
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import EEGDataset
from src.metrics.evaluation import compute_all_metrics
from src.models import ClosedFormLinear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_SUBJECTS = [13, 14, 15]
TRAIN_SUBJECTS = list(range(1, 13))
DATA_PATH = Path("data/processed/broadband_46ch.h5")
OUTPUT_PATH = Path("results/subject_valuation.json")


def load_all_subjects():
    with h5py.File(DATA_PATH, "r") as f:
        fs = float(f.attrs["fs"])
        C_in = int(f.attrs["C_scalp"])
        C_out = int(f.attrs["C_inear"])
        logger.info(f"Data: {C_in} input -> {C_out} output, fs={fs} Hz")
        data = {}
        for s in range(1, 16):
            k = f"subject_{s:02d}"
            if k in f:
                data[s] = (f[k]["scalp"][:], f[k]["inear"][:])
                logger.info(f"  Subject {s}: {data[s][0].shape[0]} windows")
    return data, C_in, C_out, fs


def evaluate_cf(train_subjects, data, C_in, C_out, fs):
    """Train CF on given train_subjects, evaluate on TEST_SUBJECTS. Return mean r."""
    device = torch.device("cpu")
    fold_rs = []

    for held_out in TEST_SUBJECTS:
        if held_out not in data:
            continue

        # Gather training data from specified subjects (excluding the test subject)
        train_s = np.concatenate([data[k][0] for k in train_subjects if k in data and k != held_out])
        train_i = np.concatenate([data[k][1] for k in train_subjects if k in data and k != held_out])
        test_s, test_i = data[held_out]

        train_ds = EEGDataset(train_s, train_i)
        test_ds = EEGDataset(test_s, test_i)

        cf = ClosedFormLinear(C_in=C_in, C_out=C_out)
        cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
        model = cf.to(device)
        model.eval()

        loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                preds.append(model(x).numpy())
                targets.append(y.numpy())
        p, t = np.concatenate(preds), np.concatenate(targets)
        m = compute_all_metrics(p, t, fs)
        r = float(m["pearson_r"].mean())
        fold_rs.append(r)

    return float(np.mean(fold_rs)), fold_rs


def main():
    t0 = time.time()
    data, C_in, C_out, fs = load_all_subjects()

    # 1. Full baseline: train on all 12 subjects
    logger.info("=" * 60)
    logger.info("Training full model on all 12 subjects...")
    full_r, full_per_test = evaluate_cf(TRAIN_SUBJECTS, data, C_in, C_out, fs)
    logger.info(f"Full model: mean_r = {full_r:.6f}")
    for s, r in zip(TEST_SUBJECTS, full_per_test):
        logger.info(f"  Test subject {s}: r = {r:.6f}")

    # 2. Leave-one-out for each training subject
    loo_results = []
    for drop_subj in TRAIN_SUBJECTS:
        if drop_subj not in data:
            logger.warning(f"Subject {drop_subj} not in data, skipping")
            continue

        remaining = [s for s in TRAIN_SUBJECTS if s != drop_subj]
        logger.info(f"--- LOO: dropping subject {drop_subj}, training on {len(remaining)} subjects ---")
        loo_r, loo_per_test = evaluate_cf(remaining, data, C_in, C_out, fs)
        marginal_value = full_r - loo_r

        n_windows = data[drop_subj][0].shape[0]
        result = {
            "subject": drop_subj,
            "n_windows": int(n_windows),
            "mean_r_without": round(loo_r, 6),
            "per_test_r_without": {str(s): round(r, 6) for s, r in zip(TEST_SUBJECTS, loo_per_test)},
            "marginal_value": round(marginal_value, 6),
        }
        loo_results.append(result)

        sign = "+" if marginal_value > 0 else ""
        logger.info(f"  Without subj {drop_subj}: mean_r = {loo_r:.6f} (marginal value: {sign}{marginal_value:.6f})")

    # 3. Sort by marginal value (most valuable first)
    loo_results.sort(key=lambda x: x["marginal_value"], reverse=True)

    # 4. Summary
    logger.info("=" * 60)
    logger.info("SUBJECT VALUATION SUMMARY (sorted by marginal value)")
    logger.info(f"{'Subj':>5s} {'Windows':>8s} {'r_without':>10s} {'Marginal':>10s} {'Verdict':>10s}")
    logger.info("-" * 50)
    harmful = []
    for r in loo_results:
        verdict = "HARMFUL" if r["marginal_value"] < 0 else ("neutral" if r["marginal_value"] == 0 else "helpful")
        if r["marginal_value"] < 0:
            harmful.append(r["subject"])
        logger.info(
            f"  {r['subject']:>3d} {r['n_windows']:>8d} {r['mean_r_without']:>10.6f} "
            f"{r['marginal_value']:>+10.6f} {verdict:>10s}"
        )

    if harmful:
        logger.info(f"\nRECOMMENDATION: Drop subjects {harmful} to improve performance")
        # Also compute what happens if we drop ALL harmful subjects
        keep = [s for s in TRAIN_SUBJECTS if s not in harmful]
        logger.info(f"Training on subjects {keep} only...")
        drop_all_r, drop_all_per_test = evaluate_cf(keep, data, C_in, C_out, fs)
        improvement = drop_all_r - full_r
        logger.info(f"  Result: mean_r = {drop_all_r:.6f} (change: {improvement:+.6f})")
    else:
        logger.info("\nAll training subjects are helpful or neutral.")
        drop_all_r = None

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    # 5. Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(DATA_PATH),
        "full_model": {
            "train_subjects": TRAIN_SUBJECTS,
            "mean_r": round(full_r, 6),
            "per_test_r": {str(s): round(r, 6) for s, r in zip(TEST_SUBJECTS, full_per_test)},
        },
        "loo_results": loo_results,
        "harmful_subjects": harmful,
        "elapsed_seconds": round(elapsed, 1),
    }
    if drop_all_r is not None:
        output["drop_all_harmful"] = {
            "kept_subjects": keep,
            "mean_r": round(drop_all_r, 6),
            "improvement": round(drop_all_r - full_r, 6),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
