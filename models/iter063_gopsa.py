"""Iteration 063: GOPSA-inspired Riemannian alignment for cross-subject EEG.

Simplified GOPSA (Geodesic Optimization on SPD manifold, NeurIPS 2024):
1. Compute spatial covariance per training subject
2. Compute Riemannian (geometric) mean of all training covariances
3. Align each subject: X_aligned = R_mean^{1/2} @ R_i^{-1/2} @ X_i
4. Train TinyDeep (H=64, 2 blocks) on aligned data
5. At test time: estimate covariance from input batch, align, then predict

Key difference from iter042 (Euclidean alignment to identity):
- Aligns to the GEOMETRIC mean, preserving the SPD manifold structure
- Uses Riemannian mean instead of Euclidean mean of covariances
- The geometric mean is the Frechet mean on the SPD manifold

Hypothesis: Riemannian alignment reduces cross-subject distribution shift
while preserving discriminative spatial structure better than whitening to identity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm, inv

from src.data.dataset import EEGDataset
from src.models import ClosedFormLinear


# ── Riemannian geometry on SPD matrices ──────────────────────────────────────

def _spd_sqrt(A):
    """Matrix square root for SPD matrix, real-valued."""
    s = sqrtm(A)
    return np.real(s).astype(np.float64)


def _spd_invsqrt(A):
    """Inverse matrix square root for SPD matrix."""
    s = sqrtm(A)
    return np.real(inv(s)).astype(np.float64)


def riemannian_mean(covs, tol=1e-10, max_iter=50):
    """Compute the Riemannian (Frechet/geometric) mean of SPD matrices.

    Uses the iterative fixed-point algorithm:
        M_{k+1} = M_k^{1/2} @ expm(mean(logm(M_k^{-1/2} @ C_i @ M_k^{-1/2}))) @ M_k^{1/2}

    Simplified version using the tangent-space approach:
        Repeat: M = M^{1/2} @ geom_mean_of(M^{-1/2} @ C_i @ M^{-1/2}) @ M^{1/2}

    For numerical stability, we use the iterative algorithm from Moakher (2005).
    """
    from scipy.linalg import logm, expm

    n = len(covs)
    # Initialize with Euclidean mean
    M = np.mean(covs, axis=0).copy()

    for iteration in range(max_iter):
        M_sqrt = _spd_sqrt(M)
        M_invsqrt = _spd_invsqrt(M)

        # Compute mean in tangent space at M
        S = np.zeros_like(M)
        for C in covs:
            # Transport C to tangent space at M
            T = M_invsqrt @ C @ M_invsqrt
            S += np.real(logm(T))
        S /= n

        # Check convergence
        norm = np.linalg.norm(S, 'fro')
        if norm < tol:
            break

        # Update M: move along geodesic
        M = M_sqrt @ np.real(expm(S)) @ M_sqrt
        # Ensure symmetry
        M = (M + M.T) / 2

    return M


def compute_subject_covariance(scalp_data):
    """Compute mean spatial covariance for a set of windows.

    Args:
        scalp_data: (N, C, T) numpy array
    Returns:
        (C, C) covariance matrix
    """
    N, C, T = scalp_data.shape
    cov = np.zeros((C, C), dtype=np.float64)
    for i in range(N):
        x = scalp_data[i].astype(np.float64)  # (C, T)
        cov += x @ x.T / T
    cov /= N
    # Regularize for numerical stability
    cov += 1e-6 * np.eye(C, dtype=np.float64)
    return cov


def align_data(scalp_data, R_mean_sqrt, R_subj_invsqrt):
    """Align subject data: X_aligned = R_mean^{1/2} @ R_subj^{-1/2} @ X

    Args:
        scalp_data: (N, C, T) numpy array
        R_mean_sqrt: (C, C) square root of geometric mean covariance
        R_subj_invsqrt: (C, C) inverse square root of subject covariance
    Returns:
        (N, C, T) aligned data
    """
    # Combined alignment matrix: R_mean^{1/2} @ R_subj^{-1/2}
    A = (R_mean_sqrt @ R_subj_invsqrt).astype(np.float32)
    N, C, T = scalp_data.shape
    aligned = np.zeros_like(scalp_data)
    for i in range(N):
        aligned[i] = A @ scalp_data[i]
    return aligned


# ── Neural network components ────────────────────────────────────────────────

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
    """Tiny deep model (H=64, 2 blocks) with skip connection."""

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


class GOPSAModel(nn.Module):
    """Wraps TinyDeep with Riemannian alignment at inference time.

    Stores the geometric mean sqrt matrix. At test time, computes the
    test subject's covariance from the full batch, aligns, then predicts.
    """

    def __init__(self, deep_model, R_mean_sqrt, C_scalp):
        super().__init__()
        self.deep = deep_model
        # Store alignment reference as buffer (not parameter)
        self.register_buffer('R_mean_sqrt',
                             torch.tensor(R_mean_sqrt.astype(np.float32)))
        self.C_scalp = C_scalp
        # Cache for test-time alignment matrix
        self._test_align_matrix = None

    def set_test_alignment(self, test_scalp_np):
        """Pre-compute alignment matrix for a test subject.

        Call this before running forward on test data.
        Args:
            test_scalp_np: (N, C, T) numpy array of all test windows
        """
        R_test = compute_subject_covariance(test_scalp_np)
        R_test_invsqrt = _spd_invsqrt(R_test).astype(np.float32)
        R_mean_sqrt_np = self.R_mean_sqrt.cpu().numpy()
        A = R_mean_sqrt_np @ R_test_invsqrt
        self._test_align_matrix = torch.tensor(A, dtype=torch.float32,
                                                device=self.R_mean_sqrt.device)

    def forward(self, x):
        if self._test_align_matrix is not None:
            # Test-time alignment: apply stored alignment matrix
            # x: (B, C, T), A: (C, C)
            x = torch.einsum('ij,bjt->bit', self._test_align_matrix, x)
        return self.deep(x)


# ── Loss and validation ──────────────────────────────────────────────────────

class CorrMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pm = pred - pred.mean(-1, keepdim=True)
        tm = target - target.mean(-1, keepdim=True)
        r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
        return self.alpha * mse + (1 - self.alpha) * (1 - r.mean())


def validate_correlation(model, loader, device):
    model.eval()
    all_r = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pm = pred - pred.mean(-1, keepdim=True)
            tm = y - y.mean(-1, keepdim=True)
            r = (pm * tm).sum(-1) / ((pm ** 2).sum(-1).sqrt() * (tm ** 2).sum(-1).sqrt() + 1e-8)
            all_r.append(r.cpu())
    return torch.cat(all_r).mean().item()


# ── Main entry point ─────────────────────────────────────────────────────────

def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    """Build GOPSA-aligned TinyDeep model.

    Steps:
    1. Estimate per-subject covariances from training data (heuristic splitting)
    2. Compute Riemannian geometric mean
    3. Align all training/val data to geometric mean
    4. Train TinyDeep on aligned data
    5. Return GOPSAModel that aligns test data at inference
    """
    scalp_train = train_ds.scalp.numpy()  # (N, C, T)
    inear_train = train_ds.inear.numpy()
    scalp_val = val_ds.scalp.numpy()
    inear_val = val_ds.inear.numpy()

    N_train = scalp_train.shape[0]
    N_val = scalp_val.shape[0]
    T = scalp_train.shape[2]

    print(f"GOPSA: {N_train} train windows, {N_val} val windows, C={C_scalp}, T={T}")

    # ── Step 1: Estimate per-subject covariances ──
    # The dataset concatenates subjects chronologically.
    # We split into pseudo-subjects based on equal chunks.
    # Using 12 chunks (one per training subject) as a heuristic.
    n_subjects = 12
    chunk_size = N_train // n_subjects
    subject_covs = []

    print("Computing per-subject covariances...")
    for i in range(n_subjects):
        start = i * chunk_size
        end = start + chunk_size if i < n_subjects - 1 else N_train
        chunk = scalp_train[start:end]
        cov = compute_subject_covariance(chunk)
        subject_covs.append(cov)
        if i < 3:
            print(f"  Subject {i}: cov trace = {np.trace(cov):.4f}, "
                  f"cond = {np.linalg.cond(cov):.1f}")

    # ── Step 2: Compute Riemannian geometric mean ──
    print("Computing Riemannian geometric mean...")
    R_mean = riemannian_mean(subject_covs)
    R_mean_sqrt = _spd_sqrt(R_mean)
    print(f"  R_mean trace = {np.trace(R_mean):.4f}, cond = {np.linalg.cond(R_mean):.1f}")

    # ── Step 3: Align training and validation data ──
    print("Aligning training data...")
    aligned_train = np.zeros_like(scalp_train)
    for i in range(n_subjects):
        start = i * chunk_size
        end = start + chunk_size if i < n_subjects - 1 else N_train
        R_subj_invsqrt = _spd_invsqrt(subject_covs[i])
        aligned_train[start:end] = align_data(
            scalp_train[start:end], R_mean_sqrt, R_subj_invsqrt
        )

    # Align validation data as a single block (mixed subjects)
    print("Aligning validation data...")
    R_val = compute_subject_covariance(scalp_val)
    R_val_invsqrt = _spd_invsqrt(R_val)
    aligned_val = align_data(scalp_val, R_mean_sqrt, R_val_invsqrt)

    # Verify alignment worked
    for i in range(min(3, n_subjects)):
        start = i * chunk_size
        end = start + chunk_size
        cov_after = compute_subject_covariance(aligned_train[start:end])
        print(f"  Aligned subject {i}: trace={np.trace(cov_after):.4f}, "
              f"vs R_mean trace={np.trace(R_mean):.4f}")

    # Build aligned datasets
    train_aligned = EEGDataset(aligned_train, inear_train)
    val_aligned = EEGDataset(aligned_val, inear_val)

    # ── Step 4: Fit CF on aligned data for skip-connection init ──
    cf = ClosedFormLinear(C_in=C_scalp, C_out=C_inear)
    cf.fit(train_aligned.scalp.numpy(), train_aligned.inear.numpy())

    # ── Step 5: Train TinyDeep on aligned data ──
    deep = TinyDeep(C_in=C_scalp, C_out=C_inear, T=T, H=64,
                    n_blocks=2, dropout=0.1).to(device)
    with torch.no_grad():
        deep.skip.weight.copy_(cf.W.float().unsqueeze(-1))

    n_params = sum(p.numel() for p in deep.parameters())
    print(f"TinyDeep params: {n_params:,}")

    loss_fn = CorrMSELoss(alpha=0.5)
    opt = torch.optim.AdamW(deep.parameters(), lr=3e-4, weight_decay=1e-2)
    tl = DataLoader(train_aligned, batch_size=128, shuffle=True,
                    num_workers=2, pin_memory=True)
    vl = DataLoader(val_aligned, batch_size=128, shuffle=False,
                    num_workers=2, pin_memory=True)

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

    # ── Step 6: Wrap in GOPSAModel for test-time alignment ──
    model = GOPSAModel(deep, R_mean_sqrt, C_scalp).to(device)

    # Monkey-patch the model's forward to handle test-time alignment automatically.
    # The benchmark calls model(x) for each test subject's full batch.
    # We override eval() to signal test mode and compute alignment on first forward.
    original_forward = model.forward
    _alignment_computed = [False]
    _batch_buffer = []

    def _gopsa_forward(x):
        """Test-time forward with automatic covariance estimation.

        On the first call after eval(), estimates covariance from this batch
        and sets the alignment matrix. Subsequent calls use the same alignment.
        This works because benchmark evaluates one subject at a time.
        """
        if not model.training and model._test_align_matrix is None:
            # First batch of a new test subject - estimate covariance
            x_np = x.detach().cpu().numpy()
            model.set_test_alignment(x_np)
        return original_forward(x)

    model.forward = _gopsa_forward

    # Override eval to reset alignment for each new test subject
    original_eval = model.eval

    def _gopsa_eval():
        model._test_align_matrix = None
        return original_eval()

    model.eval = _gopsa_eval

    return model
