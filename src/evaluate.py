"""Evaluation script for trained models.

Usage:
    python -m src.evaluate --config configs/model1_linear.yaml --checkpoint results/checkpoints/linear_spatial_best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import EEGDataset
from .metrics.evaluation import compute_all_metrics, format_metrics_table
from .train import build_model, load_config, prepare_data

logger = logging.getLogger(__name__)


def evaluate(
    cfg: dict,
    checkpoint_path: str | Path | None = None,
    split: str = "test",
) -> dict:
    """Evaluate a model on a dataset split.

    Args:
        cfg: configuration dict
        checkpoint_path: path to model checkpoint (None for closed_form)
        split: dataset split to evaluate on

    Returns:
        metrics dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_ds, val_ds, test_ds = prepare_data(cfg)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    loader = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 64), shuffle=False)

    # Build model
    model_type = cfg["model"]["type"]

    if model_type == "closed_form":
        from .models import ClosedFormLinear

        model = ClosedFormLinear(**cfg["model"].get("params", {}))
        model.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    else:
        model = build_model(cfg)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            logger.info("Loaded checkpoint: %s", checkpoint_path)

    model = model.to(device)
    model.eval()

    # Run inference
    all_pred = []
    all_target = []

    with torch.no_grad():
        for scalp, inear in loader:
            scalp = scalp.to(device)
            pred = model(scalp)
            all_pred.append(pred.cpu().numpy())
            all_target.append(inear.numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)

    fs = cfg.get("data", {}).get("fs", 256.0)
    metrics = compute_all_metrics(pred, target, fs)

    logger.info("Evaluation on '%s' split:\n%s", split, format_metrics_table(metrics))

    return metrics


def compare_models(configs: list[dict], checkpoints: list[str | None]) -> str:
    """Compare multiple models and generate a comparison table.

    Args:
        configs: list of config dicts
        checkpoints: list of checkpoint paths (None for closed_form)

    Returns:
        Formatted comparison table string
    """
    results = []
    for cfg, ckpt in zip(configs, checkpoints):
        model_type = cfg["model"]["type"]
        metrics = evaluate(cfg, ckpt)
        results.append((model_type, metrics))

    # Build comparison table
    lines = []
    header = f"{'Model':<20} {'Pearson r':>10} {'RMSE':>10} {'Rel RMSE':>10} {'SNR (dB)':>10} {'Spec RMSE':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for model_type, m in results:
        lines.append(
            f"{model_type:<20} "
            f"{m['pearson_r_mean']:>10.4f} "
            f"{m['rmse_mean']:>10.4f} "
            f"{m['relative_rmse_mean']:>10.4f} "
            f"{m['snr_db_mean']:>10.2f} "
            f"{m['spectral_rmse_mean']:>10.4f}"
        )

    return "\n".join(lines)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate EEG downsampling model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, default=None, help="Save metrics to .npz file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate(cfg, args.checkpoint, args.split)

    if args.output:
        np.savez(
            args.output,
            **{k: v for k, v in metrics.items() if isinstance(v, np.ndarray)},
            **{k: np.array(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        )
        logger.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
