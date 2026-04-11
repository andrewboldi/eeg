"""Iteration 049: Adversarial Subject-Invariant Training with Gradient Reversal.

Adds a subject classifier head with a Gradient Reversal Layer (GRL) to the
TinyDeep encoder. The encoder learns features that are GOOD for in-ear prediction
but BAD for identifying which subject they came from, forcing subject-invariant
representations.

Since the benchmark doesn't pass subject IDs, we use K-means clustering on
per-window mean channel amplitudes to create ~12 pseudo-subject labels as a proxy.

Loss = L_regression + 0.1 * L_subject_adversarial

Hypothesis: Cross-subject variability is the main bottleneck (Subject 14 ~0.27 vs
Subject 13 ~0.46). Adversarial training should reduce this gap by learning
representations that generalize across subjects.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lam=0.1):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lam)


# ---------------------------------------------------------------------------
# Multi-Scale Convolution
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


# ---------------------------------------------------------------------------
# Adversarial TinyDeep
# ---------------------------------------------------------------------------

class AdversarialTinyDeep(nn.Module):
    """TinyDeep with adversarial subject classifier head."""

    def __init__(self, C_in, C_out, n_subjects, T=256, H=64, n_blocks=2,
                 dropout=0.1, grl_lambda=0.1):
        super().__init__()
        self.T = T
        self.n_subjects = n_subjects

        # Shared backbone
        self.temporal = MultiScaleConv(C_in, H)
        self.down = nn.Sequential(nn.Conv1d(H, H, 4, stride=4, bias=False),
                                  nn.BatchNorm1d(H), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=H * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # Main prediction head
        self.up = nn.ConvTranspose1d(H, H, 4, stride=4, bias=False)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, C_out)

        # CF skip connection
        self.skip = nn.Conv1d(C_in, C_out, 1)

        # Subject classifier head with gradient reversal
        self.grl = GradientReversal(lam=grl_lambda)
        self.subject_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, n_subjects),
        )

    def _encode(self, x):
        """Shared encoder: returns transformer features (B, T_down, H)."""
        h = self.temporal(x)
        h = self.down(h).transpose(1, 2)
        h = self.transformer(h)
        return h

    def forward(self, x, return_subject_logits=False):
        skip = self.skip(x)

        # Shared encoder
        h = self._encode(x)  # (B, T_down, H)

        # Main prediction head
        h_up = h.transpose(1, 2)
        h_up = self.up(h_up)[:, :, :self.T]
        h_up = self.out_norm(h_up.transpose(1, 2))
        pred = self.out_proj(h_up).transpose(1, 2)
        pred = pred + skip

        if not return_subject_logits:
            return pred

        # Subject classifier head with gradient reversal
        h_rev = self.grl(h)              # (B, T_down, H) — gradients reversed
        h_pool = h_rev.mean(dim=1)       # (B, H) — temporal pooling
        subj_logits = self.subject_head(h_pool)  # (B, n_subjects)

        return pred, subj_logits


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
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            p = model(x, return_subject_logits=False)
            pm = p - p.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Pseudo-subject clustering
# ---------------------------------------------------------------------------

def assign_pseudo_subjects(scalp_data: np.ndarray, n_clusters: int = 12) -> np.ndarray:
    """Cluster windows into pseudo-subjects using K-means on mean channel amplitudes.

    Args:
        scalp_data: (N, C, T) scalp EEG array
        n_clusters: number of pseudo-subject clusters

    Returns:
        labels: (N,) integer cluster assignments
    """
    from sklearn.cluster import MiniBatchKMeans

    # Feature: mean absolute amplitude per channel per window
    features = np.abs(scalp_data).mean(axis=2)  # (N, C)
    # Standardize
    mu = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mu) / std

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    labels = kmeans.fit_predict(features)
    return labels


# ---------------------------------------------------------------------------
# Build and train
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit CF for skip-connection initialization
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Assign pseudo-subject labels via K-means clustering
    n_clusters = 12
    train_labels = assign_pseudo_subjects(train_ds.scalp.numpy(), n_clusters=n_clusters)
    val_labels = assign_pseudo_subjects(val_ds.scalp.numpy(), n_clusters=n_clusters)

    print(f"Pseudo-subject cluster sizes (train): {np.bincount(train_labels)}")

    # Create datasets with pseudo-subject labels
    train_labeled = TensorDataset(
        train_ds.scalp, train_ds.inear,
        torch.from_numpy(train_labels).long()
    )
    val_labeled = TensorDataset(
        val_ds.scalp, val_ds.inear,
        torch.from_numpy(val_labels).long()
    )

    # Step 3: Build adversarial model
    grl_lambda = 0.1
    model = AdversarialTinyDeep(
        C_in=C_scalp, C_out=C_inear, n_subjects=n_clusters,
        T=256, H=64, n_blocks=2, dropout=0.1, grl_lambda=grl_lambda,
    ).to(device)

    # Initialize skip connection from CF
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"AdversarialTinyDeep params: {n_params:,}")

    # Step 4: Train
    reg_loss_fn = CorrMSELoss(a=0.5)
    subj_loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    tl = DataLoader(train_labeled, batch_size=128, shuffle=True,
                    num_workers=2, pin_memory=True)
    vl = DataLoader(val_labeled, batch_size=128, shuffle=False,
                    num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0

    for ep in range(1, 151):
        model.train()
        ep_reg_loss, ep_subj_loss, n_batches = 0.0, 0.0, 0

        for x, y, subj_ids in tl:
            x, y = x.to(device), y.to(device)
            subj_ids = subj_ids.to(device)

            # Mixup augmentation (on x and y, but keep subject labels unmixed)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x_mix = lam * x + (1 - lam) * x[idx]
            y_mix = lam * y + (1 - lam) * y[idx]

            # Channel dropout
            mask = (torch.rand(x_mix.shape[0], x_mix.shape[1], 1, device=device) > 0.15).float()
            x_mix = x_mix * mask / 0.85

            # Forward with subject logits
            pred, subj_logits = model(x_mix, return_subject_logits=True)

            # Regression loss on mixed targets
            loss_reg = reg_loss_fn(pred, y_mix)

            # Subject classification loss on original (unmixed) labels
            loss_subj = subj_loss_fn(subj_logits, subj_ids)

            # Total loss: regression + adversarial subject loss
            # Note: GRL already negates gradients for the encoder, so we ADD the loss
            loss = loss_reg + grl_lambda * loss_subj

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_reg_loss += loss_reg.item()
            ep_subj_loss += loss_subj.item()
            n_batches += 1

        # Validate on prediction quality (ignoring subject classifier)
        vr = validate_correlation(model, vl, device)

        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 25 == 0:
            avg_reg = ep_reg_loss / max(n_batches, 1)
            avg_subj = ep_subj_loss / max(n_batches, 1)
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f}) "
                  f"reg_loss={avg_reg:.4f} subj_loss={avg_subj:.4f}")

        if no_imp >= 30:
            print(f"  Early stopping at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val_r: {best_r:.4f}")

    return model
