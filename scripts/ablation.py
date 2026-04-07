"""Ablation studies.

1. Loss ablation: MSE only vs MSE+spec vs MSE+band vs all
2. Architecture ablation (Model 3): remove residual, remove depthwise-separable
3. Window length ablation: T in {128, 256, 512, 1024}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _base_config(model_type: str = "conv_encoder") -> dict:
    return {
        "data": {
            "source": "synthetic",
            "n_subjects": 5,
            "n_samples": 256 * 300,
            "fs": 256.0,
            "snr_db": 10.0,
            "window_size": 256,
            "stride": 128,
            "processed_path": "data/processed/data.h5",
        },
        "model": {
            "type": model_type,
            "params": {"C_in": 21, "C_out": 4},
        },
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "optimizer": {"type": "adamw", "lr": 3e-4, "weight_decay": 1e-2},
            "scheduler": {"type": "cosine", "T_max": 100},
        },
        "loss": {
            "type": "combined",
            "lambda_spec": 0.1,
            "lambda_band": 0.1,
        },
        "logging": {
            "log_dir": "results/logs/ablation",
            "ckpt_dir": "results/checkpoints/ablation",
            "output_dir": "results/ablation",
        },
    }


def loss_ablation():
    """Compare loss function variants on Model 3."""
    from src.train import train

    configs = {
        "mse_only": {"type": "mse"},
        "mse_spec": {"type": "combined", "lambda_spec": 0.1, "lambda_band": 0.0},
        "mse_band": {"type": "combined", "lambda_spec": 0.0, "lambda_band": 0.1},
        "all_losses": {"type": "combined", "lambda_spec": 0.1, "lambda_band": 0.1},
    }

    results = {}
    for name, loss_cfg in configs.items():
        logger.info("Loss ablation: %s", name)
        cfg = _base_config("conv_encoder")
        cfg["model"]["params"].update({"H": 64, "K": 17, "N_blocks": 4, "dropout": 0.1})
        cfg["loss"] = loss_cfg
        cfg["logging"]["log_dir"] = f"results/logs/ablation/loss_{name}"

        _, metrics = train(cfg)
        results[name] = {
            "pearson_r_mean": metrics["pearson_r_mean"],
            "snr_db_mean": metrics["snr_db_mean"],
            "rmse_mean": metrics["rmse_mean"],
        }
        logger.info("  %s: r=%.4f, SNR=%.2f dB", name, metrics["pearson_r_mean"], metrics["snr_db_mean"])

    return results


def window_length_ablation():
    """Compare different window lengths."""
    from src.train import train

    window_lengths = [128, 256, 512, 1024]
    results = {}

    for T in window_lengths:
        logger.info("Window length ablation: T=%d", T)
        cfg = _base_config("conv_encoder")
        cfg["model"]["params"].update({"H": 64, "K": 17, "N_blocks": 4, "dropout": 0.1})
        cfg["data"]["window_size"] = T
        cfg["data"]["stride"] = T // 2
        # Force re-generation of data for new window size
        cfg["data"]["processed_path"] = f"data/processed/data_T{T}.h5"
        cfg["logging"]["log_dir"] = f"results/logs/ablation/window_T{T}"

        _, metrics = train(cfg)
        results[f"T={T}"] = {
            "pearson_r_mean": metrics["pearson_r_mean"],
            "snr_db_mean": metrics["snr_db_mean"],
            "rmse_mean": metrics["rmse_mean"],
        }
        logger.info("  T=%d: r=%.4f, SNR=%.2f dB", T, metrics["pearson_r_mean"], metrics["snr_db_mean"])

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument(
        "--study",
        choices=["loss", "window", "all"],
        default="all",
    )
    args = parser.parse_args()

    output_dir = Path("results/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.study in ("loss", "all"):
        all_results["loss"] = loss_ablation()

    if args.study in ("window", "all"):
        all_results["window"] = window_length_ablation()

    # Save results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Ablation results saved to %s", output_dir / "ablation_results.json")


if __name__ == "__main__":
    main()
