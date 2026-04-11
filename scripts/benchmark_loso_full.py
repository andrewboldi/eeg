"""Full 15-subject LOSO benchmark with statistical testing.

Evaluates on ALL 15 subjects (not just 3), giving much more reliable
estimates with confidence intervals and paired t-tests.
"""

from __future__ import annotations

import argparse, importlib.util, json, logging, time
from pathlib import Path

import h5py, numpy as np, torch
from scipy import stats
from torch.utils.data import DataLoader
from src.data.dataset import EEGDataset
from src.metrics.evaluation import compute_all_metrics
from src.models import ClosedFormLinear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LEADERBOARD_PATH = Path("results/benchmark/leaderboard_loso_full.jsonl")


def load_data(data_path):
    with h5py.File(data_path, "r") as f:
        fs = float(f.attrs["fs"])
        C_in = int(f.attrs["C_scalp"])
        C_out = int(f.attrs["C_inear"])
        T = int(f.attrs["window_size"])
        data = {}
        for s in range(1, 16):
            k = f"subject_{s:02d}"
            if k in f:
                data[s] = (f[k]["scalp"][:], f[k]["inear"][:])
        logger.info(f"Loaded {len(data)} subjects, {C_in}->{C_out}, T={T}, fs={fs}")
    return data, C_in, C_out, fs, T


def evaluate_model(model, test_ds, device, fs):
    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())
    p, t = np.concatenate(preds), np.concatenate(targets)
    m = compute_all_metrics(p, t, fs)
    return float(m["pearson_r"].mean()), float(m["snr_db"].mean())


def run_loso(args):
    data, C_in, C_out, fs, T = load_data(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = sorted(data.keys())

    per_subject_r = {}
    per_subject_snr = {}

    for held_out in subjects:
        train_subjects = [s for s in subjects if s != held_out]
        train_s = np.concatenate([data[s][0] for s in train_subjects])
        train_i = np.concatenate([data[s][1] for s in train_subjects])
        test_s, test_i = data[held_out]

        if args.baseline:
            train_ds = EEGDataset(train_s, train_i)
            test_ds = EEGDataset(test_s, test_i)
            cf = ClosedFormLinear(C_in=C_in, C_out=C_out)
            cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
            model = cf.to(device)
        else:
            n = len(train_s); nv = int(0.15 * n)
            train_ds = EEGDataset(train_s[:-nv], train_i[:-nv])
            val_ds = EEGDataset(train_s[-nv:], train_i[-nv:])
            test_ds = EEGDataset(test_s, test_i)
            spec = importlib.util.spec_from_file_location("m", args.model_fn)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            logger.info(f"Training for subject {held_out} ({len(train_ds)} train, {len(val_ds)} val)")
            model = mod.build_and_train(train_ds, val_ds, C_in, C_out, device).to(device)

        r, snr = evaluate_model(model, test_ds, device, fs)
        per_subject_r[held_out] = r
        per_subject_snr[held_out] = snr
        logger.info(f"Subject {held_out:2d}: r={r:.4f}, SNR={snr:.2f} dB")

    # Statistics
    r_vals = list(per_subject_r.values())
    mean_r = np.mean(r_vals)
    std_r = np.std(r_vals)
    sem_r = stats.sem(r_vals)
    ci95 = stats.t.interval(0.95, len(r_vals)-1, loc=mean_r, scale=sem_r)

    logger.info(f"\n{'='*60}")
    logger.info(f"FULL LOSO RESULTS ({len(subjects)} subjects)")
    logger.info(f"  Mean r:     {mean_r:.4f} +/- {std_r:.4f}")
    logger.info(f"  95% CI:     [{ci95[0]:.4f}, {ci95[1]:.4f}]")
    logger.info(f"  Min/Max:    {min(r_vals):.4f} / {max(r_vals):.4f}")
    logger.info(f"  Mean SNR:   {np.mean(list(per_subject_snr.values())):.2f} dB")
    logger.info(f"{'='*60}")

    result = {
        "model": args.name or ("cf_baseline" if args.baseline else Path(args.model_fn).stem),
        "data": str(args.data),
        "mean_r": float(mean_r),
        "std_r": float(std_r),
        "ci95_low": float(ci95[0]),
        "ci95_high": float(ci95[1]),
        "mean_snr": float(np.mean(list(per_subject_snr.values()))),
        "per_subject_r": {str(k): float(v) for k, v in per_subject_r.items()},
        "n_subjects": len(subjects),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/broadband_46ch.h5")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--model-fn", type=str)
    p.add_argument("--name", type=str)
    run_loso(p.parse_args())
