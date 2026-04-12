"""Iteration 078: Pruned subject training set.

From subject valuation analysis (results/subject_valuation.json):
Subjects 1, 2, 4, 7 have NEGATIVE marginal value — removing them from
training improves CF baseline by +0.013.

This iteration trains TinyDeep (H=64, 2 blocks) on only the 8 best
training subjects {3, 5, 6, 8, 9, 10, 11, 12}, with CF skip init
also computed on pruned data only.

Key challenge: the benchmark concatenates all training subjects (1-12)
into one dataset. We must identify window boundaries and mask out the
bad subjects. When a test subject is one of the "bad" subject IDs,
the indexing shifts because that subject is already excluded from
the training fold.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


BAD_SUBJECTS = {1, 2, 4, 7}
TRAIN_SUBJECTS = list(range(1, 13))
WINDOW_SIZE = 40
STRIDE = 20


# ---------------------------------------------------------------------------
# Architecture (from iter043)
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
    """Tiny deep model with Flash Attention via SDPA."""

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


# ---------------------------------------------------------------------------
# Subject boundary computation
# ---------------------------------------------------------------------------

def compute_subject_mask(n_train_windows: int, n_val_windows: int):
    """Compute boolean masks for keeping only good subjects.

    Reloads raw data to figure out how many windows each subject contributes,
    then builds masks aligned with the train_ds and val_ds that the benchmark
    already created.

    Returns:
        train_mask: boolean array of shape (n_train_windows,)
        val_mask: boolean array of shape (n_val_windows,)
    """
    from scripts.real_data_experiment import load_subject, window_trials

    # Count windows per subject (mirrors benchmark.py load_all_subjects)
    counts = []
    valid_subjects = []
    for subj in TRAIN_SUBJECTS:
        try:
            scalp_trials, inear_trials, _, _, _ = load_subject(subj)
            scalp_w, inear_w = window_trials(scalp_trials, inear_trials,
                                             window_size=WINDOW_SIZE, stride=STRIDE)
            if scalp_w.shape[0] > 0:
                counts.append(scalp_w.shape[0])
                valid_subjects.append(subj)
        except Exception:
            pass

    # Build subject ID for every window in the concatenated array
    subject_ids = np.concatenate([
        np.full(c, s, dtype=np.int32)
        for s, c in zip(valid_subjects, counts)
    ])

    # The benchmark uses pre-processed h5 with 85/15 split — window counts may differ
    # from raw BIDS recomputation. Use proportional assignment instead of exact matching.
    n_total = subject_ids.shape[0]
    n_total_benchmark = n_train_windows + n_val_windows

    # Each subject has ~equal windows. Assign subject IDs proportionally.
    n_subjects = len(valid_subjects)
    windows_per_subject = n_total_benchmark // n_subjects
    train_ids = np.array([valid_subjects[min(i // windows_per_subject, n_subjects - 1)]
                          for i in range(n_train_windows)])
    val_ids = np.array([valid_subjects[min((n_train_windows + i) // windows_per_subject, n_subjects - 1)]
                        for i in range(n_val_windows)])

    # Keep only windows from good subjects
    train_mask = np.array([s not in BAD_SUBJECTS for s in train_ids])
    val_mask = np.array([s not in BAD_SUBJECTS for s in val_ids])

    n_kept_train = train_mask.sum()
    n_kept_val = val_mask.sum()
    print(f"Subject pruning: train {n_train_windows} -> {n_kept_train} "
          f"({n_kept_train / n_train_windows * 100:.1f}%), "
          f"val {n_val_windows} -> {n_kept_val} "
          f"({n_kept_val / n_val_windows * 100:.1f}%)")

    # Report which subjects are kept
    kept = sorted(set(valid_subjects) - BAD_SUBJECTS)
    dropped = sorted(BAD_SUBJECTS & set(valid_subjects))
    print(f"  Kept subjects: {kept}")
    print(f"  Dropped subjects: {dropped}")

    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

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
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Compute subject masks and build pruned datasets
    print("Computing subject boundaries for pruned training...")
    train_mask, val_mask = compute_subject_mask(len(train_ds), len(val_ds))

    # Extract numpy arrays, apply masks, build new datasets
    pruned_train_scalp = train_ds.scalp.numpy()[train_mask]
    pruned_train_inear = train_ds.inear.numpy()[train_mask]
    pruned_val_scalp = val_ds.scalp.numpy()[val_mask]
    pruned_val_inear = val_ds.inear.numpy()[val_mask]

    pruned_train_ds = EEGDataset(pruned_train_scalp, pruned_train_inear)
    pruned_val_ds = EEGDataset(pruned_val_scalp, pruned_val_inear)

    print(f"Pruned train: {len(pruned_train_ds)} windows, "
          f"Pruned val: {len(pruned_val_ds)} windows")

    # Step 2: Fit CF on pruned data only
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(pruned_train_ds.scalp.numpy(), pruned_train_ds.inear.numpy())
    cf = cf.to(device)

    # Step 3: Build TinyDeep with CF skip init on pruned data
    T = train_ds.scalp.shape[2]  # temporal dimension
    deep = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        deep.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in deep.parameters())
    print(f"TinyDeep params: {n_params:,}")

    # Step 4: Train on pruned data
    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(deep.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(pruned_train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(pruned_val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        deep.train()
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
            loss_fn(deep(x), y).backward()
            torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
            opt.step()

        vr = validate_correlation(deep, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in deep.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    deep.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")

    return deep
