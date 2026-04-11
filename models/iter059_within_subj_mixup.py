"""Iteration 059: Within-Subject Mixup.

Cross-subject mixup HURTS because it blends incompatible spatial topographies.
Within-subject-only mixup preserves consistent electrode-brain geometry while
still providing regularization through temporal diversity.

Implementation: Track subject boundaries in the concatenated training set,
then only mix windows belonging to the same subject during training.

Architecture: TinyDeep (H=64, 2 blocks) — same as iter043.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Architecture (identical to iter043 TinyDeep)
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
    """Tiny deep model (55K params) with Flash Attention via SDPA."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1):
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
# Loss & validation (same as iter043)
# ---------------------------------------------------------------------------

class CorrMSELoss(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, p, t):
        mse = ((p - t) ** 2).mean()
        pm, tm = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
        return self.a * mse + (1 - self.a) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            pm, tm = p - p.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Dataset with subject labels
# ---------------------------------------------------------------------------

class EEGDatasetWithSubject(Dataset):
    """EEGDataset that also returns a subject ID per window."""

    def __init__(self, scalp: torch.Tensor, inear: torch.Tensor, subject_ids: torch.Tensor):
        self.scalp = scalp
        self.inear = inear
        self.subject_ids = subject_ids

    def __len__(self):
        return self.scalp.shape[0]

    def __getitem__(self, idx):
        return self.scalp[idx], self.inear[idx], self.subject_ids[idx]


# ---------------------------------------------------------------------------
# Within-subject mixup
# ---------------------------------------------------------------------------

def within_subject_mixup(x, y, subj_ids, alpha=0.4, device="cpu"):
    """Apply mixup only between samples from the SAME subject.

    For each sample, find another sample with the same subject ID and mix.
    Falls back to no mixing for subjects with only 1 sample in the batch.
    """
    batch_size = x.shape[0]
    lam = torch.tensor(np.random.beta(alpha, alpha), device=device, dtype=x.dtype)

    # Build permutation that only shuffles within same-subject groups
    mix_idx = torch.arange(batch_size, device=device)
    unique_subjs = subj_ids.unique()

    for s in unique_subjs:
        mask = (subj_ids == s).nonzero(as_tuple=True)[0]
        if mask.shape[0] < 2:
            continue  # Can't mix a single sample
        # Random permutation within this subject's samples
        perm = mask[torch.randperm(mask.shape[0], device=device)]
        mix_idx[mask] = perm

    x_mixed = lam * x + (1 - lam) * x[mix_idx]
    y_mixed = lam * y + (1 - lam) * y[mix_idx]
    return x_mixed, y_mixed


# ---------------------------------------------------------------------------
# Compute subject boundaries from raw data
# ---------------------------------------------------------------------------

def compute_subject_window_counts():
    """Compute how many windows each training subject contributes.

    Mirrors the logic in benchmark.py: load_all_subjects -> window_trials
    for subjects 1-12, then take first 90% as training set.
    """
    from scripts.real_data_experiment import load_subject, window_trials

    WINDOW_SIZE = 40
    STRIDE = 20
    TRAIN_SUBJECTS = list(range(1, 13))

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

    # Build subject ID array for all concatenated windows
    subject_ids = np.concatenate([
        np.full(c, s, dtype=np.int32)
        for s, c in zip(valid_subjects, counts)
    ])

    # Benchmark takes first 90% as training
    n_total = subject_ids.shape[0]
    n_train = int(0.9 * n_total)
    train_subject_ids = subject_ids[:n_train]

    return train_subject_ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Compute subject IDs for training windows
    print("Computing subject boundaries for within-subject mixup...")
    train_subject_ids = compute_subject_window_counts()

    assert train_subject_ids.shape[0] == len(train_ds), (
        f"Subject ID count {train_subject_ids.shape[0]} != train_ds size {len(train_ds)}"
    )
    train_subject_ids = torch.as_tensor(train_subject_ids, dtype=torch.long)

    unique, counts = np.unique(train_subject_ids.numpy(), return_counts=True)
    print(f"Training subjects: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Step 2: Fit closed-form baseline for skip-connection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 3: Build TinyDeep with CF skip init
    T = train_ds.scalp.shape[2]
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyDeep params: {n_params:,}")

    # Step 4: Create subject-aware training dataset
    train_ds_subj = EEGDatasetWithSubject(
        train_ds.scalp, train_ds.inear, train_subject_ids
    )

    loss_fn = CorrMSELoss(a=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    tl = DataLoader(train_ds_subj, batch_size=128, shuffle=True,
                    num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False,
                    num_workers=2, pin_memory=True)

    # Step 5: Train with within-subject mixup
    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        for x, y, subj_ids in tl:
            x, y = x.to(device), y.to(device)
            subj_ids = subj_ids.to(device)

            # Within-subject mixup
            x, y = within_subject_mixup(x, y, subj_ids, alpha=0.4, device=device)

            # Channel dropout (15%)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > 0.15).float()
            x = x * mask / 0.85

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
        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f})")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")
    return model
