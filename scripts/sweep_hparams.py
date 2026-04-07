"""Hyperparameter sweep using Optuna.

Sweeps hyperparameters for each model architecture:
  - Model 1: lr, weight_decay
  - Model 2: filter_length, lr, weight_decay
  - Model 3: H, K, N_blocks, lr, weight_decay, dropout
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _base_config() -> dict:
    """Base configuration shared across sweeps."""
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
        "loss": {
            "type": "combined",
            "lambda_spec": 0.1,
            "lambda_band": 0.1,
        },
        "logging": {
            "log_dir": "results/logs",
            "ckpt_dir": "results/checkpoints",
            "output_dir": "results",
        },
    }


def sweep_model1(n_trials: int = 20):
    """Sweep hyperparameters for Linear Spatial Filter."""
    import optuna
    from src.train import train

    def objective(trial: optuna.Trial) -> float:
        cfg = _base_config()
        cfg["model"] = {"type": "linear_spatial", "params": {"C_in": 21, "C_out": 4}}
        cfg["training"] = {
            "epochs": 100,
            "batch_size": 64,
            "optimizer": {
                "type": "adam",
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            },
        }
        cfg["loss"]["lambda_spec"] = trial.suggest_float("lambda_spec", 0.0, 1.0)
        cfg["loss"]["lambda_band"] = trial.suggest_float("lambda_band", 0.0, 1.0)

        _, metrics = train(cfg)
        return metrics["pearson_r_mean"]

    study = optuna.create_study(direction="maximize", study_name="linear_spatial_sweep")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best trial: %s", study.best_trial.params)
    return study


def sweep_model2(n_trials: int = 30):
    """Sweep hyperparameters for FIR Filter."""
    import optuna
    from src.train import train

    def objective(trial: optuna.Trial) -> float:
        filter_length = trial.suggest_categorical("filter_length", [17, 33, 65, 129])
        mode = trial.suggest_categorical("mode", ["causal", "acausal"])

        cfg = _base_config()
        cfg["model"] = {
            "type": "fir_filter",
            "params": {"C_in": 21, "C_out": 4, "filter_length": filter_length, "mode": mode},
        }
        cfg["training"] = {
            "epochs": 200,
            "batch_size": 64,
            "optimizer": {
                "type": "adam",
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            },
        }

        _, metrics = train(cfg)
        return metrics["pearson_r_mean"]

    study = optuna.create_study(direction="maximize", study_name="fir_filter_sweep")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best trial: %s", study.best_trial.params)
    return study


def sweep_model3(n_trials: int = 40):
    """Sweep hyperparameters for Convolutional Encoder."""
    import optuna
    from src.train import train

    def objective(trial: optuna.Trial) -> float:
        H = trial.suggest_categorical("H", [32, 64, 128])
        K = trial.suggest_categorical("K", [9, 17, 33])
        N_blocks = trial.suggest_categorical("N_blocks", [2, 4, 6])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        cfg = _base_config()
        cfg["model"] = {
            "type": "conv_encoder",
            "params": {
                "C_in": 21, "C_out": 4,
                "H": H, "K": K, "N_blocks": N_blocks, "dropout": dropout,
            },
        }
        cfg["training"] = {
            "epochs": 300,
            "batch_size": 64,
            "optimizer": {
                "type": "adamw",
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            },
            "scheduler": {"type": "cosine", "T_max": 300},
        }

        _, metrics = train(cfg)
        return metrics["pearson_r_mean"]

    study = optuna.create_study(direction="maximize", study_name="conv_encoder_sweep")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best trial: %s", study.best_trial.params)
    return study


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--model", choices=["model1", "model2", "model3", "all"], default="all")
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    if args.model in ("model1", "all"):
        sweep_model1(args.n_trials)
    if args.model in ("model2", "all"):
        sweep_model2(args.n_trials)
    if args.model in ("model3", "all"):
        sweep_model3(args.n_trials)


if __name__ == "__main__":
    main()
