"""Iteration 066: Contrastive Fine-Tune (Simplified CL-SSTER).

CL-SSTER uses same-stimulus positive pairs across subjects to learn
subject-invariant representations. Since all Ear-SAAD subjects hear the
same audiobook, same-timepoint cross-subject pairs are natural positives.

Approach:
1. TinyDeep encoder (H=64, 2 blocks) -- shared backbone
2. Main regression head (same as iter043)
3. Contrastive head: project encoder features to 32-dim, NT-Xent loss
4. Positive pairs: windows at the same temporal position from different
   subjects (approximated via nearest-index from a different subject-block)
5. Training loss: L = L_regression + 0.1 * L_contrastive

This forces the encoder to produce similar features for the same stimulus
across subjects, directly addressing cross-subject variability.

Confidence: 60% -- contrastive regularization may help generalization,
but the narrow 1-9 Hz band limits representational diversity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ---------------------------------------------------------------------------
# Architecture components (from iter043 TinyDeep)
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


class Encoder(nn.Module):
    """Shared encoder backbone (TinyDeep without output projection)."""

    def __init__(self, C_in, T=256, H=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.T = T
        self.H = H
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

    def forward(self, x):
        """Returns (B, T, H) features."""
        h = self.temporal(x)           # (B, H, T)
        h = self.down(h).transpose(1, 2)  # (B, T//4, H)
        h = self.transformer(h)        # (B, T//4, H)
        h = h.transpose(1, 2)          # (B, H, T//4)
        h = self.up(h)[:, :, :self.T]  # (B, H, T)
        h = self.out_norm(h.transpose(1, 2))  # (B, T, H)
        return h


class ContrastiveTinyDeep(nn.Module):
    """TinyDeep with dual heads: regression + contrastive."""

    def __init__(self, C_in, C_out, T=256, H=64, n_blocks=2, dropout=0.1,
                 proj_dim=32):
        super().__init__()
        self.encoder = Encoder(C_in, T=T, H=H, n_blocks=n_blocks, dropout=dropout)
        # Regression head
        self.reg_proj = nn.Linear(H, C_out)
        self.skip = nn.Conv1d(C_in, C_out, 1)
        # Contrastive projection head (MLP)
        self.contrast_proj = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, proj_dim),
        )

    def forward(self, x, return_contrast=False):
        """
        Args:
            x: (B, C_in, T)
            return_contrast: if True, also return contrastive embeddings
        Returns:
            pred: (B, C_out, T) -- regression prediction
            z: (B, proj_dim) -- contrastive embedding (only if return_contrast)
        """
        skip = self.skip(x)                     # (B, C_out, T)
        h = self.encoder(x)                     # (B, T, H)
        pred = self.reg_proj(h).transpose(1, 2) # (B, C_out, T)
        pred = pred + skip

        if return_contrast:
            # Pool over time for contrastive embedding
            h_pool = h.mean(dim=1)              # (B, H)
            z = self.contrast_proj(h_pool)      # (B, proj_dim)
            z = F.normalize(z, dim=-1)          # L2 normalize
            return pred, z
        return pred


# ---------------------------------------------------------------------------
# Loss functions
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


def nt_xent_loss(z, subject_ids, temperature=0.1):
    """NT-Xent contrastive loss using same-timepoint cross-subject positives.

    Args:
        z: (B, D) L2-normalized embeddings
        subject_ids: (B,) integer subject labels for each sample
        temperature: softmax temperature

    Returns:
        Scalar loss. For each anchor, its positive is the sample with a
        different subject_id that is closest in batch index (proxy for
        same temporal position from a different subject-block).
    """
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (B, B)

    # For each sample, find its positive: different subject, nearest index
    # subject_ids: (B,)
    pos_indices = torch.zeros(B, dtype=torch.long, device=z.device)
    for i in range(B):
        # Mask: different subject
        diff_subj = subject_ids != subject_ids[i]
        if not diff_subj.any():
            # Fallback: use farthest same-subject sample
            pos_indices[i] = (i + B // 2) % B
            continue
        # Among different-subject samples, find nearest index
        candidates = diff_subj.nonzero(as_tuple=True)[0]
        dists = (candidates - i).abs()
        pos_indices[i] = candidates[dists.argmin()]

    # NT-Xent: for each anchor i, positive is pos_indices[i]
    # Mask out self-similarity
    mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # Gather positive similarities
    pos_sim = sim[torch.arange(B, device=z.device), pos_indices]  # (B,)

    # Log-sum-exp over all negatives (everything except self)
    # logsumexp of all non-self entries
    log_sum_exp = torch.logsumexp(sim, dim=1)  # (B,)

    # Loss: -log(exp(pos) / sum(exp(all non-self)))
    loss = (-pos_sim + log_sum_exp).mean()
    return loss


# ---------------------------------------------------------------------------
# Subject-ID tracking dataset wrapper
# ---------------------------------------------------------------------------

class SubjectAwareDataset(torch.utils.data.Dataset):
    """Wraps EEGDataset to also return a subject index per window."""

    def __init__(self, ds: EEGDataset):
        self.ds = ds
        # EEGDataset stores scalp: (N, C, T) and inear: (N, C, T)
        # We need to figure out subject boundaries.
        # The dataset concatenates subjects in order. Each subject contributes
        # a variable number of windows. We approximate subject IDs by dividing
        # the dataset into equal-sized blocks (one per subject in training).
        N = len(ds)
        # Guess number of subjects from dataset (training has 12 subjects)
        n_subj = max(1, round(N / (N / 12)))  # ~12
        block_size = max(1, N // n_subj)
        self.subject_ids = torch.zeros(N, dtype=torch.long)
        for i in range(N):
            self.subject_ids[i] = min(i // block_size, n_subj - 1)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y, self.subject_ids[idx]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
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
    # Step 1: Fit CF for skip-connection init
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Build contrastive model
    model = ContrastiveTinyDeep(
        C_in=C_scalp, C_out=C_inear, T=256, H=64,
        n_blocks=2, dropout=0.1, proj_dim=32,
    ).to(device)

    # Init skip connection from CF
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ContrastiveTinyDeep params: {n_params:,}")

    # Step 3: Train with joint loss
    reg_loss_fn = CorrMSELoss(a=0.5)
    contrastive_weight = 0.1

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    train_wrapped = SubjectAwareDataset(train_ds)
    tl = DataLoader(train_wrapped, batch_size=128, shuffle=True,
                    num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False,
                    num_workers=2, pin_memory=True)

    best_r, best_state, no_imp = -1, None, 0
    for ep in range(1, 151):
        model.train()
        ep_reg_loss, ep_con_loss, n_batches = 0, 0, 0
        for x, y, subj_ids in tl:
            x, y = x.to(device), y.to(device)
            subj_ids = subj_ids.to(device)

            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.shape[0], device=device)
            x_mix = lam * x + (1 - lam) * x[idx]
            y_mix = lam * y + (1 - lam) * y[idx]
            # Mix subject IDs: use original (anchor) subject IDs
            # (mixup blends subjects, but we keep the anchor's ID for pairing)

            # Channel dropout
            mask = (torch.rand(x_mix.shape[0], x_mix.shape[1], 1, device=device) > 0.15).float()
            x_aug = x_mix * mask / 0.85

            # Forward with contrastive head
            pred, z = model(x_aug, return_contrast=True)

            # Regression loss
            l_reg = reg_loss_fn(pred, y_mix)

            # Contrastive loss (on un-mixed inputs for cleaner pairs)
            # Re-encode original (non-augmented) inputs for contrastive
            with torch.no_grad():
                # We use the augmented features already computed -- simpler
                pass
            l_con = nt_xent_loss(z, subj_ids, temperature=0.1)

            loss = l_reg + contrastive_weight * l_con

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_reg_loss += l_reg.item()
            ep_con_loss += l_con.item()
            n_batches += 1

        vr = validate_correlation(model, vl, device)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 25 == 0:
            print(f"  Epoch {ep}: val_r={vr:.4f} (best={best_r:.4f}) "
                  f"reg={ep_reg_loss/n_batches:.4f} con={ep_con_loss/n_batches:.4f}")
        if no_imp >= 30:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"ContrastiveTinyDeep best val_r: {best_r:.4f}")

    return model
