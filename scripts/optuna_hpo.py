"""Optuna hyperparameter optimization for TinyDeep on 46-channel broadband data.

Trains on Subject 13 fold only (for speed). Uses TPESampler + MedianPruner.
Reports intermediate values every 10 epochs for pruning.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 uv run python scripts/optuna_hpo.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/broadband_46ch.h5")
RESULTS_PATH = Path("results/optuna_hpo.json")
TRAIN_SUBJECTS = list(range(1, 13))
HELD_OUT = 13
N_TRIALS = 50
MAX_EPOCHS = 150


# ---------------------------------------------------------------------------
# Model architecture (same as iter043 TinyDeep, fully parameterized)
# ---------------------------------------------------------------------------

class MultiScaleConv(nn.Module):
    def __init__(self, C_in, H, kernels=(3, 7, 15, 31)):
        super().__init__()
        h = H // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(C_in, h, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(h), nn.GELU())
            for k in kernels
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


class TinyDeep(nn.Module):
    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1,
                 kernels=(3, 7, 15, 31)):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H, kernels=kernels)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                  nn.BatchNorm1d(H), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)
        self.skip = nn.Conv1d(C_in, C_out, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.temporal(x)
        h = self.down(h).transpose(1, 2)
        h = self.transformer(h)
        h = h.transpose(1, 2)
        h = self.up(h)[:, :, :self.T]
        h = self.out_norm(h.transpose(1, 2))
        h = self.out_proj(h).transpose(1, 2)
        return h + skip


# ---------------------------------------------------------------------------
# Loss and validation
# ---------------------------------------------------------------------------

class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Data loading (cached at module level)
# ---------------------------------------------------------------------------

_cached_data = {}


def load_fold_data(device):
    """Load Subject 13 fold data, caching for reuse across trials."""
    if _cached_data:
        return _cached_data["train_ds"], _cached_data["val_ds"], _cached_data["C_in"], _cached_data["C_out"]

    with h5py.File(DATA_PATH, "r") as f:
        C_in = int(f.attrs["C_scalp"])
        C_out = int(f.attrs["C_inear"])
        logger.info(f"46ch data: {C_in} input -> {C_out} output")
        data = {}
        for s in range(1, 16):
            k = f"subject_{s:02d}"
            if k in f:
                data[s] = (f[k]["scalp"][:], f[k]["inear"][:])

    train_s = np.concatenate([s for k, (s, _) in data.items() if k != HELD_OUT and k in TRAIN_SUBJECTS])
    train_i = np.concatenate([i for k, (_, i) in data.items() if k != HELD_OUT and k in TRAIN_SUBJECTS])

    n = len(train_s)
    nv = int(0.15 * n)
    train_ds = EEGDataset(train_s[:-nv], train_i[:-nv])
    val_ds = EEGDataset(train_s[-nv:], train_i[-nv:])

    logger.info(f"Fold S{HELD_OUT}: train={len(train_ds)}, val={len(val_ds)}")

    _cached_data.update(train_ds=train_ds, val_ds=val_ds, C_in=C_in, C_out=C_out)
    return train_ds, val_ds, C_in, C_out


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, C_in, C_out = load_fold_data(device)

    # --- Hyperparameters ---
    H = trial.suggest_categorical("H", [32, 48, 64, 96])
    n_blocks = trial.suggest_categorical("n_blocks", [1, 2, 3])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    loss_alpha = trial.suggest_float("loss_alpha", 0.3, 0.7)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.8)
    channel_dropout = trial.suggest_float("channel_dropout", 0.05, 0.25)
    kernels = trial.suggest_categorical("kernels", ["3,7,15,31", "3,7,15", "3,7,15,31,63"])
    kernel_tuple = tuple(int(k) for k in kernels.split(","))

    # --- CF init for skip connection ---
    cf = ClosedFormLinear(C_in=C_in, C_out=C_out)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # --- Build model ---
    T = train_ds.scalp.shape[2]
    model = TinyDeep(C_in=C_in, C_out=C_out, T=T, H=H, n_blocks=n_blocks,
                     dropout=dropout, kernels=kernel_tuple).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trial {trial.number}: H={H}, blocks={n_blocks}, kernels={kernels}, "
                f"params={n_params:,}, lr={lr:.1e}, wd={weight_decay:.1e}")

    # --- Training setup ---
    loss_fn = CorrMSELoss(a=loss_alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1.0, None, 0
    keep_prob = 1.0 - channel_dropout

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            # Mixup
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > channel_dropout).float()
            x = x * mask / keep_prob
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vr = validate_correlation(model, vl, device)

        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        # Report intermediate value every 10 epochs for pruning
        if ep % 10 == 0:
            trial.report(best_r, ep)
            if trial.should_prune():
                logger.info(f"  Trial {trial.number} pruned at epoch {ep} (best_r={best_r:.4f})")
                raise optuna.exceptions.TrialPruned()

        if ep % 25 == 0:
            logger.info(f"  Trial {trial.number} ep {ep}: val_r={vr:.4f} (best={best_r:.4f})")

        if no_imp >= 30:
            logger.info(f"  Trial {trial.number} early stopped at epoch {ep}")
            break

    logger.info(f"Trial {trial.number} finished: best_r={best_r:.4f}")
    return best_r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        study_name="tinydeep_46ch_hpo",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    elapsed = time.time() - t0

    # --- Gather results ---
    best = study.best_trial
    logger.info(f"\nBest trial {best.number}: val_r={best.value:.4f}")
    logger.info(f"Best params: {best.params}")
    logger.info(f"Total time: {elapsed/3600:.1f} hours")

    results = {
        "best_trial": best.number,
        "best_val_r": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "elapsed_hours": elapsed / 3600,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "all_trials": [],
    }

    for t in study.trials:
        trial_info = {
            "number": t.number,
            "value": t.value,
            "state": t.state.name,
            "params": t.params,
            "duration_s": (t.datetime_complete - t.datetime_start).total_seconds()
            if t.datetime_complete and t.datetime_start else None,
        }
        results["all_trials"].append(trial_info)

    # Sort trials by value descending (best first), handling None for pruned trials
    results["all_trials"].sort(key=lambda x: x["value"] if x["value"] is not None else -999, reverse=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {RESULTS_PATH}")

    # Print top 10
    logger.info("\n--- Top 10 Trials ---")
    for t in results["all_trials"][:10]:
        if t["value"] is not None:
            logger.info(f"  Trial {t['number']}: r={t['value']:.4f}  {t['params']}")


if __name__ == "__main__":
    main()
