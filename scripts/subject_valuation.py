r"""Subject valuation: measure each training subject's marginal value.

For each training subject i (1-12):
  1. Train CF on subjects {1..12} \ {i}
  2. Evaluate on test subjects 13, 14, 15
  3. Record mean_r without subject i

Marginal value = mean_r_full - mean_r_without_i
Negative value means removing that subject IMPROVES performance.

Uses broadband_46ch.h5 data and CF baseline.
"""

from __future__ import annotations

import json, logging, time
from pathlib import Path

import h5py, numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_SUBJECTS = [13, 14, 15]
TRAIN_SUBJECTS = list(range(1, 13))
DATA_PATH = Path("data/processed/broadband_46ch.h5")
OUTPUT_PATH = Path("results/subject_valuation.json")


def load_all_subjects():
    """Load data and precompute per-subject covariance matrices."""
    with h5py.File(DATA_PATH, "r") as f:
        fs = float(f.attrs["fs"])
        C_in = int(f.attrs["C_scalp"])
        C_out = int(f.attrs["C_inear"])
        logger.info(f"Data: {C_in} input -> {C_out} output, fs={fs} Hz")
        subjects = {}
        for s in range(1, 16):
            k = f"subject_{s:02d}"
            if k in f:
                scalp = f[k]["scalp"][:].astype(np.float32)
                inear = f[k]["inear"][:].astype(np.float32)
                N, _, T = scalp.shape
                # Precompute per-subject covariances for fast LOO
                # R_XX_s = sum_i X_i @ X_i^T (not divided by N yet)
                # R_YX_s = sum_i Y_i @ X_i^T
                X = scalp.reshape(N, C_in, T)
                Y = inear.reshape(N, C_out, T)
                # Batch outer products: (N, C, T) -> sum of (C, C)
                R_XX = np.einsum("nct,ndt->cd", X, X, dtype=np.float64)
                R_YX = np.einsum("nct,ndt->cd", Y, X, dtype=np.float64)
                subjects[s] = {
                    "scalp": scalp,
                    "inear": inear,
                    "n_windows": N,
                    "R_XX": R_XX,
                    "R_YX": R_YX,
                }
                logger.info(f"  Subject {s}: {N} windows")
    return subjects, C_in, C_out, fs


def solve_cf(R_XX_sum, R_YX_sum, N_total, C_in, reg=1e-4):
    """Solve W = R_YX @ inv(R_XX) from precomputed covariance sums."""
    R_XX = R_XX_sum / N_total + reg * np.eye(C_in, dtype=np.float64)
    R_YX = R_YX_sum / N_total
    W = R_YX @ np.linalg.inv(R_XX)
    return W.astype(np.float32)


def evaluate_on_test(W, test_scalp, test_inear):
    """Apply W to test data and return mean Pearson r across channels."""
    # W: (C_out, C_in), test_scalp: (N, C_in, T)
    pred = np.einsum("oc,nct->not", W, test_scalp)
    # Per-channel Pearson r across concatenated windows
    C_out = pred.shape[1]
    p = pred.transpose(1, 0, 2).reshape(C_out, -1)
    t = test_inear.transpose(1, 0, 2).reshape(C_out, -1)
    p_m = p - p.mean(axis=1, keepdims=True)
    t_m = t - t.mean(axis=1, keepdims=True)
    num = (p_m * t_m).sum(axis=1)
    den = np.sqrt((p_m**2).sum(axis=1) * (t_m**2).sum(axis=1))
    r_per_ch = num / np.maximum(den, 1e-12)
    return float(r_per_ch.mean())


def run_evaluation(train_subject_ids, subjects, C_in, C_out):
    """Train CF from precomputed covs for given subjects, eval on test subjects."""
    fold_rs = []
    for held_out in TEST_SUBJECTS:
        if held_out not in subjects:
            continue
        # Sum covariances from training subjects (excluding test subject)
        active = [s for s in train_subject_ids if s in subjects and s != held_out]
        R_XX_sum = sum(subjects[s]["R_XX"] for s in active)
        R_YX_sum = sum(subjects[s]["R_YX"] for s in active)
        N_total = sum(subjects[s]["n_windows"] for s in active)

        W = solve_cf(R_XX_sum, R_YX_sum, N_total, C_in)
        r = evaluate_on_test(W, subjects[held_out]["scalp"], subjects[held_out]["inear"])
        fold_rs.append(r)

    return float(np.mean(fold_rs)), fold_rs


def main():
    t0 = time.time()
    subjects, C_in, C_out, fs = load_all_subjects()

    # 1. Full baseline: train on all 12 subjects
    logger.info("=" * 60)
    logger.info("Training full model on all 12 subjects...")
    full_r, full_per_test = run_evaluation(TRAIN_SUBJECTS, subjects, C_in, C_out)
    logger.info(f"Full model: mean_r = {full_r:.6f}")
    for s, r in zip(TEST_SUBJECTS, full_per_test):
        logger.info(f"  Test subject {s}: r = {r:.6f}")

    # 2. Leave-one-out for each training subject
    loo_results = []
    for drop_subj in TRAIN_SUBJECTS:
        if drop_subj not in subjects:
            logger.warning(f"Subject {drop_subj} not in data, skipping")
            continue

        remaining = [s for s in TRAIN_SUBJECTS if s != drop_subj]
        loo_r, loo_per_test = run_evaluation(remaining, subjects, C_in, C_out)
        marginal_value = full_r - loo_r

        n_windows = subjects[drop_subj]["n_windows"]
        result = {
            "subject": drop_subj,
            "n_windows": int(n_windows),
            "mean_r_without": round(loo_r, 6),
            "per_test_r_without": {str(s): round(r, 6) for s, r in zip(TEST_SUBJECTS, loo_per_test)},
            "marginal_value": round(marginal_value, 6),
        }
        loo_results.append(result)

        sign = "+" if marginal_value > 0 else ""
        logger.info(f"LOO subj {drop_subj:>2d}: r_without = {loo_r:.6f}  marginal = {sign}{marginal_value:.6f}")

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
        # Compute what happens if we drop ALL harmful subjects
        keep = [s for s in TRAIN_SUBJECTS if s not in harmful]
        logger.info(f"Training on subjects {keep} only...")
        drop_all_r, drop_all_per_test = run_evaluation(keep, subjects, C_in, C_out)
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
    if harmful:
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
