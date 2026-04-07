"""Leave-One-Subject-Out (LOSO) cross-validation.

Tests how well spatial filters generalize across subjects by training
on N-1 subjects and evaluating on the held-out subject.

Usage:
    uv run python scripts/cross_subject.py --model closed_form
    uv run python scripts/cross_subject.py --model linear_spatial
    uv run python scripts/cross_subject.py --model fir_filter
    uv run python scripts/cross_subject.py --model conv_encoder
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.data.preprocess import preprocess_raw
from src.data.synthetic import generate_synthetic_data
from src.losses import TimeDomainMSE
from src.metrics.evaluation import compute_all_metrics
from src.models import ClosedFormLinear, LinearSpatialFilter, SpatioTemporalFIR, ConvEncoder
from src.train import train_one_epoch, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "closed_form": {"C_in": 21, "C_out": 4},
    "linear_spatial": {"C_in": 21, "C_out": 4},
    "fir_filter": {"C_in": 21, "C_out": 4, "filter_length": 17, "mode": "acausal"},
    "conv_encoder": {"C_in": 21, "C_out": 4, "H": 64, "K": 17, "N_blocks": 4, "dropout": 0.1},
}

TRAIN_CONFIGS = {
    "linear_spatial": {"epochs": 100, "lr": 1e-4, "grad_clip": 1.0},
    "fir_filter": {"epochs": 200, "lr": 5e-4, "grad_clip": 1.0},
    "conv_encoder": {"epochs": 300, "lr": 1e-3, "grad_clip": 1.0},
}


def preprocess_subject(subj_data, fs=256.0, window_size=256, stride=128):
    """Preprocess a single subject's data into windows."""
    result = preprocess_raw(
        subj_data["scalp"], subj_data["inear"],
        fs=fs, target_fs=fs, window_size=window_size, stride=stride,
    )
    return result["scalp"], result["inear"]


def train_model(model_type, train_ds, val_ds, device):
    """Train a gradient-based model and return the best state dict."""
    cfg = TRAIN_CONFIGS[model_type]
    model_params = MODEL_CONFIGS[model_type]

    if model_type == "linear_spatial":
        model = LinearSpatialFilter(**model_params).to(device)
        # Closed-form initialization
        cf = ClosedFormLinear(C_in=21, C_out=4)
        cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
        with torch.no_grad():
            model.W.weight.copy_(cf.W)
    elif model_type == "fir_filter":
        model = SpatioTemporalFIR(**model_params).to(device)
        # Center-tap initialization
        cf = ClosedFormLinear(C_in=21, C_out=4)
        cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
        with torch.no_grad():
            model.conv.weight.zero_()
            center = model.filter_length // 2
            model.conv.weight[:, :, center] = cf.W
    elif model_type == "conv_encoder":
        model = ConvEncoder(**model_params).to(device)

    loss_fn = TimeDomainMSE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-3)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, cfg["grad_clip"])
        val_metrics = validate(model, val_loader, loss_fn, device)
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_loso(model_type: str, n_subjects: int = 20, n_samples: int = 153600):
    """Run leave-one-subject-out cross-validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("LOSO evaluation: model=%s, n_subjects=%d, device=%s", model_type, n_subjects, device)

    # Generate all subjects
    subjects = generate_synthetic_data(
        n_subjects=n_subjects, n_samples=n_samples, fs=256.0, snr_db=10.0,
    )

    # Preprocess each subject into windows
    subject_windows = []
    for i, subj in enumerate(subjects):
        scalp, inear = preprocess_subject(subj)
        if scalp.shape[0] > 0:
            subject_windows.append((scalp, inear))
            logger.info("Subject %d: %d windows", i, scalp.shape[0])

    all_fold_metrics = []

    for held_out in range(len(subject_windows)):
        logger.info("=== Fold %d/%d (holding out subject %d) ===", held_out + 1, len(subject_windows), held_out)

        # Split: train on all but held_out, test on held_out
        train_scalp = np.concatenate([s for i, (s, _) in enumerate(subject_windows) if i != held_out])
        train_inear = np.concatenate([e for i, (_, e) in enumerate(subject_windows) if i != held_out])
        test_scalp, test_inear = subject_windows[held_out]

        # Use 10% of training data as validation
        n_train = int(0.9 * train_scalp.shape[0])
        val_scalp, val_inear = train_scalp[n_train:], train_inear[n_train:]
        train_scalp, train_inear = train_scalp[:n_train], train_inear[:n_train]

        train_ds = EEGDataset(train_scalp, train_inear)
        val_ds = EEGDataset(val_scalp, val_inear)
        test_ds = EEGDataset(test_scalp, test_inear)

        # Build and train model
        if model_type == "closed_form":
            model = ClosedFormLinear(**MODEL_CONFIGS["closed_form"])
            model.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
            model = model.to(device)
        else:
            model = train_model(model_type, train_ds, val_ds, device)

        # Evaluate on held-out subject
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        all_pred, all_target = [], []
        with torch.no_grad():
            for scalp, inear in test_loader:
                pred = model(scalp.to(device))
                all_pred.append(pred.cpu().numpy())
                all_target.append(inear.numpy())

        pred = np.concatenate(all_pred)
        target = np.concatenate(all_target)
        metrics = compute_all_metrics(pred, target, fs=256.0)

        fold_summary = {
            "subject": held_out,
            "pearson_r": float(metrics["pearson_r"].mean()),
            "rmse": float(metrics["rmse"].mean()),
            "snr_db": float(metrics["snr_db"].mean()),
        }
        all_fold_metrics.append(fold_summary)
        logger.info("  Subject %d: r=%.4f, RMSE=%.4f, SNR=%.2f dB",
                     held_out, fold_summary["pearson_r"], fold_summary["rmse"], fold_summary["snr_db"])

    # Summary
    mean_r = np.mean([m["pearson_r"] for m in all_fold_metrics])
    std_r = np.std([m["pearson_r"] for m in all_fold_metrics])
    mean_snr = np.mean([m["snr_db"] for m in all_fold_metrics])
    std_snr = np.std([m["snr_db"] for m in all_fold_metrics])

    logger.info("=== LOSO Summary (%s) ===", model_type)
    logger.info("  r = %.4f +/- %.4f", mean_r, std_r)
    logger.info("  SNR = %.2f +/- %.2f dB", mean_snr, std_snr)

    results = {
        "model": model_type,
        "n_subjects": len(subject_windows),
        "folds": all_fold_metrics,
        "mean_r": float(mean_r),
        "std_r": float(std_r),
        "mean_snr": float(mean_snr),
        "std_snr": float(std_snr),
    }

    # Save
    out_dir = Path("results/cross_subject")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{model_type}_loso.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_dir / f"{model_type}_loso.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="LOSO cross-validation")
    parser.add_argument("--model", type=str, default="closed_form",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n-subjects", type=int, default=20)
    args = parser.parse_args()
    run_loso(args.model, n_subjects=args.n_subjects)


if __name__ == "__main__":
    main()
