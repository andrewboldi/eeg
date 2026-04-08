"""Extension 007: Scalp-to-in-ear prediction on real EEG data.

Uses the Ear-SAAD dataset (Geirnaert et al. 2025):
  15 subjects, 29 scalp + 12 in-ear channels, 20 Hz, 6 x 10-min trials.

Usage:
    uv run python scripts/real_data_experiment.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.losses import TimeDomainMSE
from src.metrics.evaluation import compute_all_metrics
from src.models import ClosedFormLinear, ConvEncoder
from src.train import train_one_epoch, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw/ear_saad/preprocessedData")

# In-ear channel prefixes (6 per ear = 12 total)
INEAR_PREFIXES = {"ELA", "ELB", "ELC", "ELE", "ELI", "ELT",
                  "ERA", "ERB", "ERC", "ERE", "ERI", "ERT"}

# Around-ear channel prefixes (to exclude)
AROUND_EAR_PREFIXES = {"cEL", "cER"}

# Channels to exclude
EXCLUDE = {"M1", "M2", "Fp1-cr"}


def load_subject(subject_id: int):
    """Load one subject's data from .mat file.

    Returns:
        scalp: list of (C_scalp, T) arrays per trial
        inear: list of (C_inear, T) arrays per trial
        scalp_ch: list of scalp channel names
        inear_ch: list of in-ear channel names
        fs: sampling rate
    """
    mat_path = DATA_DIR / f"dataSubject{subject_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing: {mat_path}")

    mat = sio.loadmat(str(mat_path), squeeze_me=True)

    # Channel names
    channels = [str(ch).strip() for ch in mat["channels"]]
    fs = float(mat["fs"])

    # Classify channels
    scalp_idx, inear_idx = [], []
    scalp_ch, inear_ch = [], []

    for i, ch in enumerate(channels):
        if ch in INEAR_PREFIXES or ch in {"ELA", "ELB", "ELC", "ELE", "ELI", "ELT",
                                           "ERA", "ERB", "ERC", "ERE", "ERI", "ERT"}:
            inear_idx.append(i)
            inear_ch.append(ch)
        elif ch in EXCLUDE:
            continue  # skip mastoids, reference
        elif any(ch.startswith(p) for p in AROUND_EAR_PREFIXES):
            continue  # skip around-ear
        else:
            scalp_idx.append(i)
            scalp_ch.append(ch)

    scalp_idx = np.array(scalp_idx)
    inear_idx = np.array(inear_idx)

    # Extract trials
    eeg_trials = mat["eegTrials"]  # object array of shape (6,)
    scalp_trials = []
    inear_trials = []

    for trial_idx in range(len(eeg_trials)):
        trial_data = eeg_trials[trial_idx]  # (T, C) or similar
        if trial_data.ndim == 1:
            continue
        # Ensure (T, C) shape
        if trial_data.shape[0] < trial_data.shape[1]:
            trial_data = trial_data.T  # now (T, C)

        scalp_data = trial_data[:, scalp_idx].T  # (C_scalp, T)
        inear_data = trial_data[:, inear_idx].T  # (C_inear, T)

        # Interpolate NaN values per channel (linear)
        for arr in [scalp_data, inear_data]:
            for ch_i in range(arr.shape[0]):
                row = arr[ch_i]
                nans = np.isnan(row)
                if nans.all():
                    row[:] = 0.0  # fully bad channel
                elif nans.any():
                    good = ~nans
                    row[nans] = np.interp(
                        np.flatnonzero(nans), np.flatnonzero(good), row[good]
                    )

        # Z-score normalize per channel (now NaN-free)
        scalp_data = (scalp_data - scalp_data.mean(axis=1, keepdims=True)) / (scalp_data.std(axis=1, keepdims=True) + 1e-8)
        inear_data = (inear_data - inear_data.mean(axis=1, keepdims=True)) / (inear_data.std(axis=1, keepdims=True) + 1e-8)

        scalp_trials.append(scalp_data.astype(np.float32))
        inear_trials.append(inear_data.astype(np.float32))

    return scalp_trials, inear_trials, scalp_ch, inear_ch, fs


def window_trials(scalp_trials, inear_trials, window_size=40, stride=20):
    """Window continuous trials into fixed-length segments.

    At 20 Hz, window_size=40 = 2 seconds, stride=20 = 1 second.
    """
    all_scalp, all_inear = [], []
    for scalp, inear in zip(scalp_trials, inear_trials):
        T = scalp.shape[1]
        for start in range(0, T - window_size + 1, stride):
            all_scalp.append(scalp[:, start:start + window_size])
            all_inear.append(inear[:, start:start + window_size])

    if not all_scalp:
        return np.array([]), np.array([])
    return np.stack(all_scalp), np.stack(all_inear)


def train_conv_encoder(train_ds, val_ds, C_in, C_out, device, epochs=200, window_size=40):
    """Train conv encoder for real data dimensions."""
    # Adjust architecture for lower sampling rate (20 Hz) and different channel counts
    model = ConvEncoder(C_in=C_in, C_out=C_out, H=32, K=9, N_blocks=3, dropout=0.1).to(device)
    loss_fn = TimeDomainMSE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=1.0)
        val_metrics = validate(model, val_loader, loss_fn, device)
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            logger.info("  Epoch %d: val_loss=%.6f (best=%.6f)", epoch, val_metrics["val_loss"], best_val)

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_ds, device, fs=20.0):
    """Evaluate and return mean metrics."""
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


def run_pooled_experiment():
    """Train on all subjects pooled, evaluate on held-out trials."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load all subjects
    all_scalp_windows = []
    all_inear_windows = []
    C_scalp = C_inear = None

    for subj in range(1, 16):
        try:
            scalp_trials, inear_trials, scalp_ch, inear_ch, fs = load_subject(subj)
            logger.info("Subject %d: %d trials, %d scalp ch (%s), %d in-ear ch (%s), fs=%g Hz",
                        subj, len(scalp_trials), len(scalp_ch), ",".join(scalp_ch[:5]) + "...",
                        len(inear_ch), ",".join(inear_ch), fs)

            if C_scalp is None:
                C_scalp = len(scalp_ch)
                C_inear = len(inear_ch)
                logger.info("Channel config: %d scalp -> %d in-ear", C_scalp, C_inear)

            scalp_w, inear_w = window_trials(scalp_trials, inear_trials, window_size=40, stride=20)
            if scalp_w.shape[0] > 0:
                all_scalp_windows.append(scalp_w)
                all_inear_windows.append(inear_w)
                logger.info("  -> %d windows", scalp_w.shape[0])
        except Exception as e:
            logger.warning("Subject %d failed: %s", subj, e)

    scalp = np.concatenate(all_scalp_windows)
    inear = np.concatenate(all_inear_windows)
    logger.info("Total: %d windows, scalp shape %s, inear shape %s", scalp.shape[0], scalp.shape, inear.shape)

    # Chronological split: 70/15/15
    n = scalp.shape[0]
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_ds = EEGDataset(scalp[:n_train], inear[:n_train])
    val_ds = EEGDataset(scalp[n_train:n_train + n_val], inear[n_train:n_train + n_val])
    test_ds = EEGDataset(scalp[n_train + n_val:], inear[n_train + n_val:])
    logger.info("Split: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))

    results = {}

    # 1. Closed-form
    logger.info("=== Closed-form ===")
    cf_model = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf_model.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf_model = cf_model.to(device)
    cf_metrics = evaluate_model(cf_model, test_ds, device, fs=20.0)
    logger.info("  r=%.4f, RMSE=%.4f, SNR=%.2f dB", cf_metrics["pearson_r"], cf_metrics["rmse"], cf_metrics["snr_db"])
    logger.info("  Per-channel r: %s", [f"{r:.3f}" for r in cf_metrics["pearson_r_per_ch"]])
    results["closed_form"] = cf_metrics

    # 2. Conv Encoder
    logger.info("=== Conv Encoder (200 epochs) ===")
    conv_model = train_conv_encoder(train_ds, val_ds, C_scalp, C_inear, device, epochs=200, window_size=40)
    conv_metrics = evaluate_model(conv_model, test_ds, device, fs=20.0)
    logger.info("  r=%.4f, RMSE=%.4f, SNR=%.2f dB", conv_metrics["pearson_r"], conv_metrics["rmse"], conv_metrics["snr_db"])
    logger.info("  Per-channel r: %s", [f"{r:.3f}" for r in conv_metrics["pearson_r_per_ch"]])
    results["conv_encoder"] = conv_metrics

    # Save
    out_dir = Path("results/real_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ear_saad_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to results/real_data/ear_saad_results.json")

    return results


def run_loso_experiment():
    """Leave-one-subject-out on real data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== LOSO on real data ===")

    # Load all subjects
    subject_data = {}
    C_scalp = C_inear = None

    for subj in range(1, 16):
        try:
            scalp_trials, inear_trials, scalp_ch, inear_ch, fs = load_subject(subj)
            scalp_w, inear_w = window_trials(scalp_trials, inear_trials, window_size=40, stride=20)
            if scalp_w.shape[0] > 0:
                subject_data[subj] = (scalp_w, inear_w)
                if C_scalp is None:
                    C_scalp = scalp_w.shape[1]
                    C_inear = inear_w.shape[1]
        except Exception as e:
            logger.warning("Subject %d failed: %s", subj, e)

    fold_results = []
    for held_out in sorted(subject_data.keys()):
        # Train on all but held_out
        train_scalp = np.concatenate([s for k, (s, _) in subject_data.items() if k != held_out])
        train_inear = np.concatenate([e for k, (_, e) in subject_data.items() if k != held_out])
        test_scalp, test_inear = subject_data[held_out]

        train_ds = EEGDataset(train_scalp, train_inear)
        test_ds = EEGDataset(test_scalp, test_inear)

        # Closed-form only (fast)
        cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
        cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
        cf = cf.to(device)
        metrics = evaluate_model(cf, test_ds, device, fs=20.0)

        fold_results.append({"subject": held_out, **metrics})
        logger.info("  Subject %d: r=%.4f, SNR=%.2f dB", held_out, metrics["pearson_r"], metrics["snr_db"])

    mean_r = np.mean([f["pearson_r"] for f in fold_results])
    std_r = np.std([f["pearson_r"] for f in fold_results])
    mean_snr = np.mean([f["snr_db"] for f in fold_results])
    std_snr = np.std([f["snr_db"] for f in fold_results])

    logger.info("LOSO Summary: r=%.4f +/- %.4f, SNR=%.2f +/- %.2f dB", mean_r, std_r, mean_snr, std_snr)

    loso_results = {
        "folds": fold_results,
        "mean_r": float(mean_r),
        "std_r": float(std_r),
        "mean_snr": float(mean_snr),
        "std_snr": float(std_snr),
    }

    out_dir = Path("results/real_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ear_saad_loso.json", "w") as f:
        json.dump(loso_results, f, indent=2)

    return loso_results


if __name__ == "__main__":
    pooled = run_pooled_experiment()
    loso = run_loso_experiment()
