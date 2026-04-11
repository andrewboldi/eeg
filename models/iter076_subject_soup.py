"""Iteration 076: Subject-Subset Model Soup.

Train TinyDeep models on different random subsets of the training data
(simulating subject diversity), then greedily average weights of compatible
models (model soup / weight averaging).

From weight_averaging.md: models trained on overlapping-but-different data
subsets learn complementary features. Greedy soup starts with the best
single model and iteratively adds a candidate's weights (uniform average)
only if the averaged model improves validation correlation.

Hypothesis: Greedy weight-averaging across 5 subset-trained TinyDeep models
will reduce cross-subject overfitting and improve generalization by ~0.002 r.

Confidence: 60% -- weight averaging is well-established but gains depend on
model diversity, which may be limited with 75% overlap between subsets.
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Architecture (TinyDeep from iter043)
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
    """Tiny deep model (~55K params) with transformer encoder."""

    def __init__(self, C_in, C_out, T=40, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.temporal = MultiScaleConv(C_in, H)
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
# Loss & validation
# ---------------------------------------------------------------------------

class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm = p - p.mean(-1, keepdim=True)
        tm = t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
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
            r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Training a single TinyDeep on a data subset
# ---------------------------------------------------------------------------

def train_single(
    train_ds: EEGDataset,
    val_loader: DataLoader,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
    cf_W: torch.Tensor,
    subset_indices: np.ndarray,
    seed: int,
    max_epochs: int = 120,
    patience: int = 25,
) -> tuple[TinyDeep, float, dict]:
    """Train one TinyDeep on a subset of training data. Returns (model, best_val_r, state_dict)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = train_ds.scalp.shape[2]
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1).to(device)

    # Init skip connection from CF
    with torch.no_grad():
        model.skip.weight.copy_(cf_W.float().unsqueeze(-1))

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    subset = Subset(train_ds, subset_indices.tolist())
    tl = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1.0, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]
            # Channel dropout
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vr = validate_correlation(model, val_loader, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"    Subset model (seed={seed}): val_r={best_r:.4f}, stopped at epoch {ep}")
    return model, best_r, best_state


# ---------------------------------------------------------------------------
# Weight averaging utilities
# ---------------------------------------------------------------------------

def average_state_dicts(sd_list: list[dict]) -> OrderedDict:
    """Uniform average of a list of state dicts."""
    avg = OrderedDict()
    n = len(sd_list)
    for key in sd_list[0]:
        stacked = torch.stack([sd[key].float() for sd in sd_list])
        avg[key] = stacked.mean(dim=0)
    return avg


def greedy_soup(
    candidates: list[tuple[float, dict]],
    model_template: TinyDeep,
    val_loader: DataLoader,
    device: torch.device,
) -> OrderedDict:
    """Greedy model soup: start with best, add others if they improve val_r.

    Args:
        candidates: list of (val_r, state_dict) sorted by val_r descending
        model_template: a TinyDeep instance to load weights into for evaluation
        val_loader: validation DataLoader
        device: torch device

    Returns:
        The souped state dict.
    """
    # Sort by validation performance (best first)
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    soup_sds = [candidates[0][1]]
    best_soup_r = candidates[0][0]
    print(f"  Soup: start with best model (val_r={best_soup_r:.4f})")

    for i, (cand_r, cand_sd) in enumerate(candidates[1:], 2):
        # Try adding this candidate
        trial_sds = soup_sds + [cand_sd]
        trial_avg = average_state_dicts(trial_sds)
        model_template.load_state_dict({k: v.to(device) for k, v in trial_avg.items()})
        trial_r = validate_correlation(model_template, val_loader, device)

        if trial_r > best_soup_r:
            soup_sds = trial_sds
            best_soup_r = trial_r
            print(f"  Soup: added model {i} (individual r={cand_r:.4f}) -> soup r={trial_r:.4f}")
        else:
            print(f"  Soup: rejected model {i} (individual r={cand_r:.4f}, "
                  f"soup would be r={trial_r:.4f} < {best_soup_r:.4f})")

    print(f"  Final soup: {len(soup_sds)} models, val_r={best_soup_r:.4f}")
    return average_state_dicts(soup_sds)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Subject-subset model soup.

    1. Fit CF baseline for skip-connection init
    2. Create 5 random 75% subsets of training data (simulating 9/12 subjects)
    3. Train TinyDeep on each subset
    4. Greedy soup: average weights of compatible models
    """
    N_SUBSETS = 5
    SUBSET_FRAC = 0.75  # ~9/12 subjects worth of data

    # Step 1: Closed-form baseline for skip init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
    cf_W = cf.W.clone()
    print(f"CF baseline fitted")

    # Validation loader (shared across all models)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Step 2: Create random subsets
    n_train = len(train_ds)
    n_subset = int(n_train * SUBSET_FRAC)
    rng = np.random.RandomState(42)

    subsets = []
    for i in range(N_SUBSETS):
        idx = rng.choice(n_train, size=n_subset, replace=False)
        idx.sort()
        subsets.append(idx)
        print(f"  Subset {i}: {len(idx)} / {n_train} samples")

    # Step 3: Train models on each subset
    candidates = []
    for i, idx in enumerate(subsets):
        print(f"\nTraining subset model {i + 1}/{N_SUBSETS}...")
        model, val_r, state_dict = train_single(
            train_ds, val_loader, C_scalp, C_inear, device,
            cf_W, idx, seed=42 + i,
            max_epochs=120, patience=25,
        )
        candidates.append((val_r, state_dict))

    # Step 4: Greedy soup
    T = train_ds.scalp.shape[2]
    soup_model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1).to(device)

    print("\nBuilding greedy soup...")
    soup_sd = greedy_soup(candidates, soup_model, val_loader, device)
    soup_model.load_state_dict({k: v.to(device) for k, v in soup_sd.items()})

    final_r = validate_correlation(soup_model, val_loader, device)
    n_params = sum(p.numel() for p in soup_model.parameters())
    print(f"\nFinal soup model: val_r={final_r:.4f}, params={n_params:,}")

    return soup_model
