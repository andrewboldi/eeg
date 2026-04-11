"""Scaling law experiment: how does model size affect performance?

Tests 6 model configurations on the 46-channel broadband data:
  - Tiny:   H=32,  blocks=2  (~50K params)
  - Small:  H=64,  blocks=2  (~200K params)
  - Medium: H=128, blocks=4  (~1M params)   ← current iter039
  - Large:  H=192, blocks=6  (~3M params)
  - XL:     H=256, blocks=8  (~6M params)

Uses only Subject 13 fold (fastest) for quick comparison.
Trains each for 150 epochs with early stopping.

Usage:
    PYTHONPATH=. uv run python scripts/scaling_law.py
"""

from __future__ import annotations

import json, logging, time
from pathlib import Path

import h5py, numpy as np, torch
from torch.utils.data import DataLoader
from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear

# Import model class
import sys; sys.path.insert(0, '.')
from models.iter039_deep_broadband import DeepBroadbandModel, CorrMSELoss, validate_correlation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/broadband_46ch.h5")
RESULTS_PATH = Path("results/scaling_law.json")

CONFIGS = [
    {"name": "tiny",   "H": 32,  "n_blocks": 2, "dropout": 0.1},
    {"name": "small",  "H": 64,  "n_blocks": 2, "dropout": 0.1},
    {"name": "medium", "H": 128, "n_blocks": 4, "dropout": 0.15},
    {"name": "large",  "H": 192, "n_blocks": 6, "dropout": 0.15},
    {"name": "xl",     "H": 256, "n_blocks": 8, "dropout": 0.2},
]


def load_fold_13():
    """Load Subject 13 fold from 46-channel data."""
    with h5py.File(DATA_PATH, "r") as f:
        C_in = int(f.attrs["C_scalp"])
        C_out = int(f.attrs["C_inear"])
        data = {}
        for s in range(1, 16):
            k = f"subject_{s:02d}"
            if k in f:
                data[s] = (f[k]["scalp"][:], f[k]["inear"][:])

    held_out = 13
    train_s = np.concatenate([s for k,(s,_) in data.items() if k != held_out and k < 13])
    train_i = np.concatenate([i for k,(_,i) in data.items() if k != held_out and k < 13])
    test_s, test_i = data[held_out]

    n = len(train_s); nv = int(0.15*n)
    train_ds = EEGDataset(train_s[:-nv], train_i[:-nv])
    val_ds = EEGDataset(train_s[-nv:], train_i[-nv:])
    test_ds = EEGDataset(test_s, test_i)

    return train_ds, val_ds, test_ds, C_in, C_out


def train_config(config, train_ds, val_ds, test_ds, C_in, C_out, device):
    """Train one configuration and return metrics."""
    cf = ClosedFormLinear(C_in=C_in, C_out=C_out)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    model = DeepBroadbandModel(
        C_in=C_in, C_out=C_out, T=256,
        H=config["H"], n_blocks=config["n_blocks"], dropout=config["dropout"],
    ).to(device)

    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Config {config['name']}: H={config['H']}, blocks={config['n_blocks']}, "
                f"params={n_params:,}")

    loss_fn = CorrMSELoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                              num_workers=2, pin_memory=True)

    best_val_r = -1.0
    best_state = None
    no_improve = 0
    patience = 30
    val_history = []

    t0 = time.time()
    for epoch in range(1, 151):
        model.train()
        for scalp, inear in train_loader:
            scalp, inear = scalp.to(device), inear.to(device)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(scalp.shape[0], device=device)
            scalp = lam * scalp + (1-lam) * scalp[idx]
            inear = lam * inear + (1-lam) * inear[idx]
            mask = (torch.rand(scalp.shape[0], scalp.shape[1], 1, device=device) > 0.15).float()
            scalp = scalp * mask / 0.85

            optimizer.zero_grad()
            loss = loss_fn(model(scalp), inear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_r = validate_correlation(model, val_loader, device)
        val_history.append(val_r)

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    train_time = time.time() - t0

    # Test evaluation
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_r = validate_correlation(model, test_loader, device)

    result = {
        "name": config["name"],
        "H": config["H"],
        "n_blocks": config["n_blocks"],
        "n_params": n_params,
        "best_val_r": best_val_r,
        "test_r": test_r,
        "best_epoch": len(val_history) - no_improve,
        "total_epochs": len(val_history),
        "train_time_s": train_time,
        "val_history": val_history,
    }

    logger.info(f"  -> val_r={best_val_r:.4f}, test_r={test_r:.4f}, "
                f"epochs={len(val_history)}, time={train_time:.0f}s")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_ds, val_ds, test_ds, C_in, C_out = load_fold_13()
    logger.info(f"Data: {C_in}->{C_out}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # CF baseline
    cf = ClosedFormLinear(C_in=C_in, C_out=C_out)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf = cf.to(device)
    cf_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    cf_r = validate_correlation(cf, cf_loader, device)
    logger.info(f"CF baseline test_r: {cf_r:.4f}")

    results = [{"name": "cf_baseline", "n_params": C_in*C_out, "test_r": cf_r,
                "best_val_r": cf_r, "H": 0, "n_blocks": 0}]

    for config in CONFIGS:
        try:
            r = train_config(config, train_ds, val_ds, test_ds, C_in, C_out, device)
            results.append(r)
        except Exception as e:
            logger.warning(f"Config {config['name']} failed: {e}")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Remove val_history for cleaner output
    clean = [{k: v for k, v in r.items() if k != 'val_history'} for r in results]
    with open(RESULTS_PATH, "w") as f:
        json.dump(clean, f, indent=2)

    # Print summary
    logger.info("\n=== SCALING LAW RESULTS ===")
    logger.info(f"{'Config':<10} {'Params':>10} {'Val r':>8} {'Test r':>8} {'Epochs':>8}")
    logger.info("-" * 50)
    for r in results:
        logger.info(f"{r['name']:<10} {r.get('n_params',0):>10,} {r['best_val_r']:>8.4f} "
                     f"{r['test_r']:>8.4f} {r.get('total_epochs',''):>8}")


if __name__ == "__main__":
    main()
