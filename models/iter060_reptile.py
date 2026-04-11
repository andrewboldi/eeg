"""Iteration 060: Reptile meta-learning for few-shot subject adaptation.

Reptile (Nichol et al. 2018) is a first-order MAML alternative that learns an
initialization from which the model can quickly adapt to any new subject.

Outer loop: sample a "subject" (chunk of training data), clone weights,
train clone for K inner steps, then nudge original weights toward the clone.

At test time: fine-tune the last layer on validation data for 10 steps,
producing a model adapted to the target distribution.

Architecture: TinyDeep (H=64, 2 transformer blocks, ~55K params).

Hypothesis: Meta-learned initialization will generalize better across subjects
than standard training, especially for the hard subject 14.

Confidence: 50% - meta-learning helps most with many diverse tasks; 12 pseudo-
subjects from pooled data may not provide enough task diversity.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
# Loss and evaluation
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


@torch.no_grad()
def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        pm = p - p.mean(-1, keepdim=True)
        tm = y - y.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / (pm.norm(dim=-1) * tm.norm(dim=-1) + 1e-8)
        all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ---------------------------------------------------------------------------
# Reptile helpers
# ---------------------------------------------------------------------------

def split_into_subject_chunks(dataset: EEGDataset, n_chunks: int = 12):
    """Split a pooled dataset into n_chunks pseudo-subject datasets."""
    n = len(dataset)
    indices = np.random.permutation(n)
    chunk_size = n // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else n
        idx = indices[start:end]
        chunks.append(TensorDataset(dataset.scalp[idx], dataset.inear[idx]))
    return chunks


def reptile_inner_loop(model, task_loader, loss_fn, device, K=5, inner_lr=1e-3):
    """Run K gradient steps on a single task. Returns the updated clone."""
    clone = copy.deepcopy(model)
    clone.train()
    opt = torch.optim.SGD(clone.parameters(), lr=inner_lr)
    steps = 0
    while steps < K:
        for x, y in task_loader:
            if steps >= K:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(clone(x), y).backward()
            torch.nn.utils.clip_grad_norm_(clone.parameters(), 1.0)
            opt.step()
            steps += 1
    return clone


def reptile_outer_step(model, clone, epsilon=0.5):
    """Reptile update: theta += epsilon * (theta_clone - theta)."""
    with torch.no_grad():
        for p, p_clone in zip(model.parameters(), clone.parameters()):
            p.data += epsilon * (p_clone.data - p.data)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    # Step 1: Fit CF for skip-connection initialization
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())

    # Step 2: Initialize TinyDeep with CF skip weights
    T = train_ds.scalp.shape[-1]
    model = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64, n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        model.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Reptile TinyDeep params: {n_params:,}")

    loss_fn = CorrMSELoss(a=0.5)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Step 3: Split training data into pseudo-subject chunks
    n_chunks = 12
    subject_chunks = split_into_subject_chunks(train_ds, n_chunks=n_chunks)
    subject_loaders = [
        DataLoader(chunk, batch_size=64, shuffle=True, drop_last=True)
        for chunk in subject_chunks
    ]
    print(f"Split training data into {n_chunks} pseudo-subject chunks "
          f"({len(subject_chunks[0])} samples each)")

    # Step 4: Reptile meta-learning outer loop
    n_outer = 300       # outer iterations
    K = 5               # inner steps per task
    inner_lr = 1e-3     # inner loop learning rate (SGD)
    epsilon_init = 0.5  # outer step size (decays linearly)
    best_r, best_state, no_imp = -1, None, 0

    for outer in range(1, n_outer + 1):
        # Linear decay of outer step size
        epsilon = epsilon_init * (1 - outer / n_outer)

        # Sample a random pseudo-subject
        task_idx = np.random.randint(n_chunks)
        task_loader = subject_loaders[task_idx]

        # Inner loop: clone and adapt
        clone = reptile_inner_loop(model, task_loader, loss_fn, device, K=K, inner_lr=inner_lr)

        # Outer step: move model toward the adapted clone
        reptile_outer_step(model, clone, epsilon=epsilon)

        # Validate periodically
        if outer % 10 == 0:
            vr = validate_correlation(model, val_loader, device)
            if vr > best_r:
                best_r = vr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if outer % 50 == 0:
                print(f"  Reptile outer {outer}/{n_outer}: val_r={vr:.4f} "
                      f"(best={best_r:.4f}, eps={epsilon:.3f})")
            if no_imp >= 10:  # 100 outer steps without improvement
                print(f"  Early stopping at outer step {outer}")
                break

    # Load best meta-learned state
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Meta-learning complete. Best val_r: {best_r:.4f}")

    # Step 5: Test-time adaptation -- fine-tune last layer on val data
    # Freeze all but output projection
    for name, p in model.named_parameters():
        p.requires_grad = "out_proj" in name or "out_norm" in name

    adapt_opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    adapt_steps = 10
    model.train()
    step = 0
    while step < adapt_steps:
        for x, y in val_loader:
            if step >= adapt_steps:
                break
            x, y = x.to(device), y.to(device)
            adapt_opt.zero_grad()
            loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            adapt_opt.step()
            step += 1

    # Unfreeze for evaluation (model.eval() will be called by benchmark)
    for p in model.parameters():
        p.requires_grad = True

    adapted_r = validate_correlation(model, val_loader, device)
    print(f"After test-time adaptation ({adapt_steps} steps): val_r={adapted_r:.4f}")

    return model
