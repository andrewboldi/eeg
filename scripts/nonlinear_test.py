"""Test whether Conv Encoder outperforms linear models on nonlinear forward models.

Generates synthetic data with Y = f(M @ X) + noise where f is a nonlinearity,
then compares closed-form vs conv encoder performance.

Usage:
    uv run python scripts/nonlinear_test.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset, make_splits
from src.data.preprocess import preprocess_raw
from src.data.synthetic import generate_synthetic_scalp, _default_mixing_matrix
from src.losses import TimeDomainMSE
from src.metrics.evaluation import compute_all_metrics
from src.models import ClosedFormLinear, ConvEncoder
from src.train import train_one_epoch, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_nonlinear_data(
    nonlinearity: str = "tanh",
    strength: float = 0.5,
    n_subjects: int = 20,
    n_samples: int = 153600,
    fs: float = 256.0,
    snr_db: float = 10.0,
    seed: int = 42,
):
    """Generate data with nonlinear forward model.

    Y = (1-s) * (M @ X) + s * f(M @ X) + noise

    where s is strength (0=linear, 1=fully nonlinear).
    """
    M = _default_mixing_matrix()
    rng = np.random.default_rng(seed + 2000)
    scalp_data = generate_synthetic_scalp(
        n_channels=21, n_samples=n_samples, fs=fs, n_subjects=n_subjects, seed=seed,
    )

    nonlinear_fn = {
        "tanh": np.tanh,
        "relu": lambda x: np.maximum(x, 0),
        "square": lambda x: np.sign(x) * x**2 / (np.std(x) + 1e-8),
    }[nonlinearity]

    subjects = []
    for scalp in scalp_data:
        linear_part = M @ scalp
        nonlinear_part = nonlinear_fn(linear_part / (np.std(linear_part) + 1e-8))
        nonlinear_part *= np.std(linear_part)  # match scale

        inear_clean = (1 - strength) * linear_part + strength * nonlinear_part

        signal_power = np.mean(inear_clean**2, axis=-1, keepdims=True)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.standard_normal(inear_clean.shape).astype(np.float32)
        noise *= np.sqrt(noise_power)
        inear = (inear_clean + noise).astype(np.float32)

        subjects.append({"scalp": scalp, "inear": inear})

    return subjects


def train_conv_encoder(train_ds, val_ds, device, epochs=200):
    """Train conv encoder with MSE loss."""
    model = ConvEncoder(C_in=21, C_out=4, H=64, K=17, N_blocks=4, dropout=0.1).to(device)
    loss_fn = TimeDomainMSE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=1.0)
        val_metrics = validate(model, val_loader, loss_fn, device)
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            logger.info("  Conv epoch %d: val_loss=%.4f (best=%.4f)", epoch, val_metrics["val_loss"], best_val)

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_ds, device):
    """Evaluate model on test set."""
    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    all_pred, all_target = [], []
    with torch.no_grad():
        for scalp, inear in loader:
            pred = model(scalp.to(device))
            all_pred.append(pred.cpu().numpy())
            all_target.append(inear.numpy())
    pred = np.concatenate(all_pred)
    target = np.concatenate(all_target)
    metrics = compute_all_metrics(pred, target, fs=256.0)
    return float(metrics["pearson_r"].mean()), float(metrics["snr_db"].mean())


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for nonlinearity in ["tanh", "square"]:
        for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
            label = f"{nonlinearity}_s{strength:.2f}"
            logger.info("=== %s ===", label)

            subjects = generate_nonlinear_data(
                nonlinearity=nonlinearity, strength=strength,
                n_subjects=10, n_samples=76800,
            )

            all_scalp, all_inear = [], []
            for subj in subjects:
                result = preprocess_raw(subj["scalp"], subj["inear"],
                                        fs=256.0, target_fs=256.0, window_size=256, stride=128)
                if result["scalp"].shape[0] > 0:
                    all_scalp.append(result["scalp"])
                    all_inear.append(result["inear"])

            scalp = np.concatenate(all_scalp)
            inear = np.concatenate(all_inear)
            splits = make_splits(scalp, inear)

            train_ds = EEGDataset(splits["train"][0], splits["train"][1])
            val_ds = EEGDataset(splits["val"][0], splits["val"][1])
            test_ds = EEGDataset(splits["test"][0], splits["test"][1])

            # Closed-form
            cf_model = ClosedFormLinear(C_in=21, C_out=4)
            cf_model.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
            cf_model = cf_model.to(device)
            cf_r, cf_snr = evaluate_model(cf_model, test_ds, device)

            # Conv encoder
            conv_model = train_conv_encoder(train_ds, val_ds, device, epochs=200)
            conv_r, conv_snr = evaluate_model(conv_model, test_ds, device)

            logger.info("  Closed-form: r=%.4f, SNR=%.2f dB", cf_r, cf_snr)
            logger.info("  Conv Encoder: r=%.4f, SNR=%.2f dB", conv_r, conv_snr)
            logger.info("  Advantage: Δr=%.4f", conv_r - cf_r)

            results[label] = {
                "nonlinearity": nonlinearity,
                "strength": strength,
                "closed_form_r": cf_r,
                "closed_form_snr": cf_snr,
                "conv_encoder_r": conv_r,
                "conv_encoder_snr": conv_snr,
                "delta_r": conv_r - cf_r,
            }

    # Save
    out_dir = Path("results/nonlinear_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to results/nonlinear_test/results.json")

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info("%-20s  %8s  %8s  %8s", "Condition", "CF r", "Conv r", "Δr")
    for label, r in results.items():
        logger.info("%-20s  %8.4f  %8.4f  %+8.4f", label, r["closed_form_r"], r["conv_encoder_r"], r["delta_r"])


if __name__ == "__main__":
    run_experiment()
