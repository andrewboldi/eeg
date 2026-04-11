"""Broadband 46-channel benchmark (scalp + around-ear → in-ear).

Same as benchmark_broadband.py but uses broadband_46ch.h5 (46 input channels).
"""

from __future__ import annotations

import argparse, importlib.util, json, logging, time
from pathlib import Path

import h5py, numpy as np, torch
from torch.utils.data import DataLoader
from src.data.dataset import EEGDataset
from src.metrics.evaluation import compute_all_metrics
from src.models import ClosedFormLinear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_SUBJECTS = [13, 14, 15]
TRAIN_SUBJECTS = list(range(1, 13))
DATA_PATH = Path("data/processed/broadband_46ch.h5")
LEADERBOARD_PATH = Path("results/benchmark/leaderboard_broadband_46ch.jsonl")


def load_all_subjects():
    with h5py.File(DATA_PATH, "r") as f:
        fs = float(f.attrs["fs"])
        C_in = int(f.attrs["C_scalp"])
        C_out = int(f.attrs["C_inear"])
        logger.info(f"46ch data: {C_in} input -> {C_out} output, fs={fs} Hz")
        data = {}
        for s in range(1, 16):
            k = f"subject_{s:02d}"
            if k in f:
                data[s] = (f[k]["scalp"][:], f[k]["inear"][:])
                logger.info(f"  Subject {s}: {data[s][0].shape[0]} windows")
    return data, C_in, C_out, fs


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
    return {"pearson_r": float(m["pearson_r"].mean()), "snr_db": float(m["snr_db"].mean()),
            "pearson_r_per_ch": m["pearson_r"].tolist()}


def run_benchmark(args):
    data, C_in, C_out, fs = load_all_subjects()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    fold_results = []
    for held_out in TEST_SUBJECTS:
        if held_out not in data: continue
        train_s = np.concatenate([s for k,(s,_) in data.items() if k != held_out and k in TRAIN_SUBJECTS])
        train_i = np.concatenate([i for k,(_,i) in data.items() if k != held_out and k in TRAIN_SUBJECTS])
        test_s, test_i = data[held_out]

        if args.baseline:
            train_ds = EEGDataset(train_s, train_i)
            test_ds = EEGDataset(test_s, test_i)
            cf = ClosedFormLinear(C_in=C_in, C_out=C_out)
            cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
            model = cf.to(device)
        else:
            n = len(train_s); nv = int(0.15*n)
            train_ds = EEGDataset(train_s[:-nv], train_i[:-nv])
            val_ds = EEGDataset(train_s[-nv:], train_i[-nv:])
            test_ds = EEGDataset(test_s, test_i)
            spec = importlib.util.spec_from_file_location("m", args.model_fn)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            logger.info(f"Training for subject {held_out} (train={len(train_ds)}, val={len(val_ds)})")
            model = mod.build_and_train(train_ds, val_ds, C_in, C_out, device).to(device)

        metrics = evaluate_model(model, test_ds, device, fs)
        fold_results.append(metrics)
        logger.info(f"  Subject {held_out}: r={metrics['pearson_r']:.4f}")

    result = {
        "model": args.name or ("closed_form_46ch" if args.baseline else Path(args.model_fn).stem),
        "mean_r": float(np.mean([f["pearson_r"] for f in fold_results])),
        "std_r": float(np.std([f["pearson_r"] for f in fold_results])),
        "mean_snr": float(np.mean([f["snr_db"] for f in fold_results])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    logger.info(f"Result: mean_r={result['mean_r']:.4f} +/- {result['std_r']:.4f}")
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--model-fn", type=str)
    p.add_argument("--name", type=str)
    run_benchmark(p.parse_args())
