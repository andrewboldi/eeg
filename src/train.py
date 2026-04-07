"""Training loop shared across all models.

Usage:
    python -m src.train --config configs/model1_linear.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from .data.dataset import EEGDataset, make_splits, save_to_hdf5
from .data.preprocess import preprocess_raw
from .data.synthetic import generate_synthetic_data
from .losses import CombinedLoss, TimeDomainMSE
from .metrics.evaluation import compute_all_metrics, format_metrics_table
from .models import LinearSpatialFilter, SpatioTemporalFIR, ConvEncoder, ClosedFormLinear

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "linear_spatial": LinearSpatialFilter,
    "fir_filter": SpatioTemporalFIR,
    "conv_encoder": ConvEncoder,
    "closed_form": ClosedFormLinear,
}


def load_config(path: str | Path) -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> nn.Module:
    """Build model from config."""
    model_type = cfg["model"]["type"]
    model_params = cfg["model"].get("params", {})
    model_cls = MODEL_REGISTRY[model_type]
    return model_cls(**model_params)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = cfg["training"]["optimizer"]
    opt_type = opt_cfg.get("type", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)

    if opt_type == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config."""
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "mse")

    if loss_type == "combined":
        return CombinedLoss(
            lambda_spec=loss_cfg.get("lambda_spec", 0.1),
            lambda_band=loss_cfg.get("lambda_band", 0.1),
            fs=cfg.get("data", {}).get("fs", 256.0),
        )
    return TimeDomainMSE()


def prepare_data(cfg: dict) -> tuple[EEGDataset, EEGDataset, EEGDataset]:
    """Prepare train/val/test datasets.

    If a preprocessed HDF5 file exists, load from there.
    Otherwise, generate synthetic data and preprocess.
    """
    data_cfg = cfg.get("data", {})
    processed_path = Path(data_cfg.get("processed_path", "data/processed/data.h5"))
    source = data_cfg.get("source", "synthetic")

    if processed_path.exists():
        logger.info("Loading preprocessed data from %s", processed_path)
        train_ds = EEGDataset.from_hdf5(processed_path, "train")
        val_ds = EEGDataset.from_hdf5(processed_path, "val")
        test_ds = EEGDataset.from_hdf5(processed_path, "test")
        return train_ds, val_ds, test_ds

    # Generate and preprocess synthetic data
    logger.info("Generating synthetic data...")
    n_subjects = data_cfg.get("n_subjects", 5)
    n_samples = data_cfg.get("n_samples", 256 * 300)
    fs = data_cfg.get("fs", 256.0)
    snr_db = data_cfg.get("snr_db", 10.0)
    window_size = data_cfg.get("window_size", 256)
    stride = data_cfg.get("stride", 128)

    subjects = generate_synthetic_data(
        n_subjects=n_subjects,
        n_samples=n_samples,
        fs=fs,
        snr_db=snr_db,
    )

    all_scalp = []
    all_inear = []

    for subj_data in subjects:
        result = preprocess_raw(
            subj_data["scalp"],
            subj_data["inear"],
            fs=fs,
            target_fs=fs,  # already at target fs for synthetic
            window_size=window_size,
            stride=stride,
        )
        if result["scalp"].shape[0] > 0:
            all_scalp.append(result["scalp"])
            all_inear.append(result["inear"])

    scalp = np.concatenate(all_scalp, axis=0)
    inear = np.concatenate(all_inear, axis=0)

    splits = make_splits(scalp, inear)

    # Save to HDF5
    for split_name, (s, i) in splits.items():
        save_to_hdf5(processed_path, split_name, s, i)

    train_ds = EEGDataset(splits["train"][0], splits["train"][1])
    val_ds = EEGDataset(splits["val"][0], splits["val"][1])
    test_ds = EEGDataset(splits["test"][0], splits["test"][1])

    return train_ds, val_ds, test_ds


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    loss_components = {}

    for scalp, inear in loader:
        scalp = scalp.to(device)
        inear = inear.to(device)

        optimizer.zero_grad()
        pred = model(scalp)

        if isinstance(loss_fn, CombinedLoss):
            loss, components = loss_fn(pred, inear)
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0.0) + v
        else:
            loss = loss_fn(pred, inear)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg = {"train_loss": total_loss / max(n_batches, 1)}
    for k, v in loss_components.items():
        avg[f"train_{k}"] = v / max(n_batches, 1)
    return avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for scalp, inear in loader:
        scalp = scalp.to(device)
        inear = inear.to(device)
        pred = model(scalp)

        if isinstance(loss_fn, CombinedLoss):
            loss, _ = loss_fn(pred, inear)
        else:
            loss = loss_fn(pred, inear)

        total_loss += loss.item()
        n_batches += 1

    return {"val_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fs: float = 256.0,
) -> dict:
    """Full evaluation with all metrics."""
    model.eval()
    all_pred = []
    all_target = []

    for scalp, inear in loader:
        scalp = scalp.to(device)
        pred = model(scalp)
        all_pred.append(pred.cpu().numpy())
        all_target.append(inear.numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)

    return compute_all_metrics(pred, target, fs)


def train(cfg: dict) -> tuple[nn.Module, dict]:
    """Full training pipeline.

    Returns:
        (trained_model, test_metrics)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Data
    train_ds, val_ds, test_ds = prepare_data(cfg)
    batch_size = cfg["training"].get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model_type = cfg["model"]["type"]

    if model_type == "closed_form":
        model = build_model(cfg)
        model.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
        model = model.to(device)
        logger.info("Closed-form model fitted")
        metrics = evaluate_model(model, test_loader, device, cfg.get("data", {}).get("fs", 256.0))
        logger.info("Test metrics:\n%s", format_metrics_table(metrics))
        return model, metrics

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%d parameters)", model_type, n_params)

    # Loss, optimizer, scheduler
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)

    epochs = cfg["training"].get("epochs", 100)
    scheduler_cfg = cfg["training"].get("scheduler", {})
    scheduler = None
    if scheduler_cfg.get("type") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_cfg.get("T_max", epochs))

    # Logging
    log_dir = Path(cfg.get("logging", {}).get("log_dir", "results/logs")) / model_type
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Checkpointing
    ckpt_dir = Path(cfg.get("logging", {}).get("ckpt_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = validate(model, val_loader, loss_fn, device)

        if scheduler is not None:
            scheduler.step()

        # Log
        for k, v in {**train_metrics, **val_metrics}.items():
            writer.add_scalar(k, v, epoch)

        # Checkpoint best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / f"{model_type}_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %d: train_loss=%.6f, val_loss=%.6f (best=%.6f @ %d)",
                epoch,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                best_val_loss,
                best_epoch,
            )

    writer.close()

    # Load best and evaluate
    model.load_state_dict(torch.load(ckpt_dir / f"{model_type}_best.pt", weights_only=True))
    model = model.to(device)
    metrics = evaluate_model(model, test_loader, device, cfg.get("data", {}).get("fs", 256.0))
    logger.info("Test metrics:\n%s", format_metrics_table(metrics))

    return model, metrics


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train EEG downsampling model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, metrics = train(cfg)

    # Save final metrics
    output_dir = Path(cfg.get("logging", {}).get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = cfg["model"]["type"]

    np.savez(
        output_dir / f"{model_type}_metrics.npz",
        **{k: v for k, v in metrics.items() if isinstance(v, np.ndarray)},
        **{k: np.array(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    )


if __name__ == "__main__":
    main()
