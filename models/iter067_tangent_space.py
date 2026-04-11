"""Iteration 067: Augmented Covariance + Tangent Space + Ridge Regression.

Riemannian geometry approach:
1. For each training window: stack scalp + inear -> augmented signal
2. Compute (C_scalp + C_inear) x (C_scalp + C_inear) covariance per window
3. Project all covariances to tangent space at the geometric mean (pyriemann)
4. Extract upper triangle -> feature vector
5. Train Ridge regression: tangent-space features of SCALP-ONLY covariance -> in-ear channels

Key insight: During training we use augmented covariances to learn the cross-covariance
structure between scalp and in-ear geometrically. At test time we only have scalp,
so we train Ridge from scalp-only tangent features to in-ear predictions.

Actually, the cleaner approach that works at test time:
- Compute scalp-only covariance per window -> tangent space features
- Ridge regression from tangent features -> flattened in-ear window

This captures the spatial structure of scalp EEG in a Riemannian-aware way,
then uses Ridge to map to in-ear. Non-neural, pure geometry.

References:
- Barachant et al. 2012 (tangent space for BCI)
- Engemann et al. 2020 (covariance pipelines for regression)
- docs/research/riemannian_eeg.md
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from src.data.dataset import EEGDataset

logger = logging.getLogger(__name__)


def _compute_covs(X: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Compute covariance matrices per window with shrinkage.

    Args:
        X: (N, C, T) signal windows
        reg: regularization added to diagonal

    Returns:
        covs: (N, C, C) SPD covariance matrices
    """
    N, C, T = X.shape
    covs = np.zeros((N, C, C), dtype=np.float64)
    for i in range(N):
        covs[i] = X[i] @ X[i].T / T
        covs[i] += reg * np.eye(C)
    return covs


def _compute_augmented_covs(
    scalp: np.ndarray, inear: np.ndarray, reg: float = 1e-6
) -> np.ndarray:
    """Compute augmented covariance matrices [scalp; inear] per window.

    Args:
        scalp: (N, C_scalp, T)
        inear: (N, C_inear, T)
        reg: regularization

    Returns:
        covs: (N, C_scalp+C_inear, C_scalp+C_inear) SPD matrices
    """
    N, C_s, T = scalp.shape
    C_i = inear.shape[1]
    C_aug = C_s + C_i

    covs = np.zeros((N, C_aug, C_aug), dtype=np.float64)
    for i in range(N):
        aug = np.concatenate([scalp[i], inear[i]], axis=0)  # (C_aug, T)
        covs[i] = aug @ aug.T / T
        covs[i] += reg * np.eye(C_aug)
    return covs


def _tangent_space_manual(covs: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    """Project SPD matrices to tangent space at reference point C_ref.

    Uses the logarithmic map:
        S_i = C_ref^{-1/2} @ log(C_ref^{-1/2} @ Cov_i @ C_ref^{-1/2}) @ C_ref^{-1/2}

    Then extracts upper triangle (with sqrt(2) off-diagonal scaling).

    Args:
        covs: (N, C, C) SPD matrices
        C_ref: (C, C) reference point (geometric mean)

    Returns:
        features: (N, C*(C+1)/2) tangent vectors
    """
    from scipy.linalg import logm, sqrtm, inv

    N, C, _ = covs.shape
    n_features = C * (C + 1) // 2

    # C_ref^{-1/2}
    C_ref_isqrt = np.real(inv(sqrtm(C_ref)))

    features = np.zeros((N, n_features), dtype=np.float64)

    for i in range(N):
        # Whitened covariance
        S = C_ref_isqrt @ covs[i] @ C_ref_isqrt
        # Matrix logarithm
        S_log = np.real(logm(S))
        # Extract upper triangle with sqrt(2) scaling for off-diagonal
        idx = 0
        for r in range(C):
            for c in range(r, C):
                if r == c:
                    features[i, idx] = S_log[r, c]
                else:
                    features[i, idx] = S_log[r, c] * np.sqrt(2)
                idx += 1

    return features


def _geometric_mean_iterative(
    covs: np.ndarray, tol: float = 1e-8, max_iter: int = 50
) -> np.ndarray:
    """Compute the Riemannian geometric (Frechet) mean of SPD matrices.

    Uses the iterative fixed-point algorithm.

    Args:
        covs: (N, C, C) SPD matrices
        tol: convergence tolerance
        max_iter: maximum iterations

    Returns:
        mean: (C, C) geometric mean
    """
    from scipy.linalg import sqrtm, inv, logm, expm

    N, C, _ = covs.shape
    # Initialize with arithmetic mean
    G = np.mean(covs, axis=0)

    for it in range(max_iter):
        G_isqrt = np.real(inv(sqrtm(G)))
        G_sqrt = np.real(sqrtm(G))

        # Compute mean of log maps
        S = np.zeros((C, C), dtype=np.float64)
        for i in range(N):
            S += np.real(logm(G_isqrt @ covs[i] @ G_isqrt))
        S /= N

        # Update
        G_new = G_sqrt @ np.real(expm(S)) @ G_sqrt

        # Check convergence
        diff = np.linalg.norm(G_new - G) / np.linalg.norm(G)
        G = G_new

        if diff < tol:
            logger.info("Geometric mean converged in %d iterations (diff=%.2e)", it + 1, diff)
            break
    else:
        logger.warning("Geometric mean did not converge after %d iterations (diff=%.2e)", max_iter, diff)

    return G


class TangentSpaceRidgeWrapper(nn.Module):
    """Wraps tangent-space + Ridge regression as an nn.Module.

    At forward() time:
    1. Compute per-window scalp covariance
    2. Project to tangent space at stored geometric mean
    3. Apply Ridge weights to predict in-ear
    """

    def __init__(self, C_scalp: int, C_inear: int, T: int):
        super().__init__()
        self.C_scalp = C_scalp
        self.C_inear = C_inear
        self.T = T
        self.n_cov_features = C_scalp * (C_scalp + 1) // 2

        # Stored parameters (set during fit)
        self.register_buffer("C_ref_isqrt", torch.zeros(C_scalp, C_scalp))
        self.register_buffer("ridge_W", torch.zeros(C_inear * T, self.n_cov_features))
        self.register_buffer("ridge_b", torch.zeros(C_inear * T))
        self.register_buffer("reg_diag", torch.eye(C_scalp) * 1e-6)

    def _cov_and_tangent_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Compute covariance and tangent projection in PyTorch.

        Args:
            x: (B, C_scalp, T)

        Returns:
            features: (B, n_cov_features)
        """
        B, C, T = x.shape
        # Covariance per window: (B, C, C)
        covs = torch.bmm(x, x.transpose(1, 2)) / T
        covs = covs + self.reg_diag.unsqueeze(0)

        # Whitened covariance: C_ref^{-1/2} @ Cov @ C_ref^{-1/2}
        # (C, C) @ (B, C, C) @ (C, C) -> (B, C, C)
        isqrt = self.C_ref_isqrt  # (C, C)
        S = torch.matmul(torch.matmul(isqrt.unsqueeze(0), covs), isqrt.unsqueeze(0))

        # Matrix logarithm per window (no batch logm in PyTorch, use eigendecomposition)
        # S is SPD after whitening, so eigenvalues are positive
        # logm(S) = V @ diag(log(lambda)) @ V^T
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        # Clamp eigenvalues to avoid log(0)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        log_eigenvalues = torch.log(eigenvalues)
        # Reconstruct: V @ diag(log_lambda) @ V^T
        S_log = torch.matmul(
            eigenvectors * log_eigenvalues.unsqueeze(-2),
            eigenvectors.transpose(-2, -1)
        )

        # Extract upper triangle with sqrt(2) off-diagonal scaling
        features = torch.zeros(B, self.n_cov_features, device=x.device, dtype=x.dtype)
        idx = 0
        for r in range(C):
            for c in range(r, C):
                if r == c:
                    features[:, idx] = S_log[:, r, c]
                else:
                    features[:, idx] = S_log[:, r, c] * 1.4142135623730951
                idx += 1

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_scalp, T)

        Returns:
            (B, C_inear, T)
        """
        B, C, T = x.shape

        # Get tangent space features
        features = self._cov_and_tangent_torch(x)  # (B, n_features)

        # Apply Ridge: (B, n_features) @ (n_features, C_inear*T) + bias -> (B, C_inear*T)
        out = torch.matmul(features, self.ridge_W.T) + self.ridge_b  # (B, C_inear*T)

        # Reshape to (B, C_inear, T)
        return out.view(B, self.C_inear, T)


def build_and_train(
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    C_scalp: int,
    C_inear: int,
    device: torch.device,
) -> nn.Module:
    """Build tangent-space + Ridge model.

    Steps:
    1. Compute scalp covariance matrices for all training windows
    2. Compute geometric mean (Riemannian)
    3. Project to tangent space
    4. Train Ridge regression from tangent features to flattened in-ear
    5. Wrap in nn.Module
    """
    scalp_np = train_ds.scalp.numpy().astype(np.float64)  # (N, C_scalp, T)
    inear_np = train_ds.inear.numpy().astype(np.float64)  # (N, C_inear, T)
    N, C_s, T = scalp_np.shape
    C_i = inear_np.shape[1]

    print(f"Tangent Space + Ridge: N={N}, C_scalp={C_s}, C_inear={C_i}, T={T}")
    print(f"Covariance features: {C_s}x{C_s} -> {C_s * (C_s + 1) // 2} tangent features")

    # Step 1: Compute scalp covariance matrices
    print("Computing covariance matrices...")
    covs = _compute_covs(scalp_np, reg=1e-6)
    print(f"Covariances: {covs.shape}, condition numbers: "
          f"median={np.median([np.linalg.cond(c) for c in covs[:100]]):.1f}")

    # Step 2: Compute geometric mean
    print("Computing Riemannian geometric mean...")
    try:
        from pyriemann.utils.mean import mean_riemann
        C_ref = mean_riemann(covs)
        print("Used pyriemann for geometric mean")
    except ImportError:
        print("pyriemann not available, using manual iterative geometric mean")
        C_ref = _geometric_mean_iterative(covs)

    print(f"Geometric mean condition number: {np.linalg.cond(C_ref):.1f}")

    # Step 3: Project to tangent space
    print("Projecting to tangent space...")
    try:
        from pyriemann.tangentspace import TangentSpace
        ts = TangentSpace(metric="riemann")
        ts.reference_ = C_ref
        ts.metric = "riemann"
        # pyriemann TangentSpace.transform expects (N, C, C) covariances
        X_tangent = ts.transform(covs)
        print(f"Used pyriemann TangentSpace: {X_tangent.shape}")
    except (ImportError, Exception) as e:
        print(f"pyriemann TangentSpace failed ({e}), using manual projection")
        X_tangent = _tangent_space_manual(covs, C_ref)
        print(f"Manual tangent space: {X_tangent.shape}")

    # Step 4: Prepare targets (flatten in-ear windows)
    Y_flat = inear_np.reshape(N, C_i * T)  # (N, C_inear * T)
    print(f"Features: {X_tangent.shape}, Targets: {Y_flat.shape}")

    # Step 5: Ridge regression with cross-validated alpha
    print("Training Ridge regression...")
    # Try a few alpha values on validation set
    val_scalp_np = val_ds.scalp.numpy().astype(np.float64)
    val_inear_np = val_ds.inear.numpy().astype(np.float64)
    N_val = val_scalp_np.shape[0]

    val_covs = _compute_covs(val_scalp_np, reg=1e-6)

    try:
        val_tangent = ts.transform(val_covs)
    except Exception:
        val_tangent = _tangent_space_manual(val_covs, C_ref)

    val_Y_flat = val_inear_np.reshape(N_val, C_i * T)

    best_alpha = 1.0
    best_val_r = -1.0

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_tangent, Y_flat)
        pred_val = ridge.predict(val_tangent)

        # Compute mean Pearson r across channels
        pred_reshaped = pred_val.reshape(N_val, C_i, T)
        target_reshaped = val_inear_np

        r_vals = []
        for ch in range(C_i):
            for win in range(N_val):
                p = pred_reshaped[win, ch]
                t = target_reshaped[win, ch]
                p_m = p - p.mean()
                t_m = t - t.mean()
                denom = np.sqrt((p_m ** 2).sum() * (t_m ** 2).sum())
                if denom > 1e-12:
                    r_vals.append((p_m * t_m).sum() / denom)
        mean_r = np.mean(r_vals) if r_vals else 0.0

        print(f"  alpha={alpha:8.2f}: val r={mean_r:.6f}")

        if mean_r > best_val_r:
            best_val_r = mean_r
            best_alpha = alpha

    print(f"Best alpha={best_alpha}, val r={best_val_r:.6f}")

    # Fit final model with best alpha
    ridge_final = Ridge(alpha=best_alpha, fit_intercept=True)
    ridge_final.fit(X_tangent, Y_flat)

    # Step 6: Build wrapper module
    from scipy.linalg import sqrtm, inv

    C_ref_isqrt = np.real(inv(sqrtm(C_ref)))

    model = TangentSpaceRidgeWrapper(C_s, C_i, T)
    model.C_ref_isqrt = torch.tensor(C_ref_isqrt, dtype=torch.float32)
    model.ridge_W = torch.tensor(ridge_final.coef_, dtype=torch.float32)
    model.ridge_b = torch.tensor(ridge_final.intercept_, dtype=torch.float32)
    model.reg_diag = torch.eye(C_s, dtype=torch.float32) * 1e-6

    # Verify on validation set
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        val_scalp_t = torch.tensor(val_scalp_np, dtype=torch.float32, device=device)
        pred_t = model(val_scalp_t)
        pred_np = pred_t.cpu().numpy()

        r_vals = []
        for ch in range(C_i):
            for win in range(N_val):
                p = pred_np[win, ch]
                t = val_inear_np[win, ch]
                p_m = p - p.mean()
                t_m = t - t.mean()
                denom = np.sqrt((p_m ** 2).sum() * (t_m ** 2).sum())
                if denom > 1e-12:
                    r_vals.append((p_m * t_m).sum() / denom)
        torch_r = np.mean(r_vals) if r_vals else 0.0

    print(f"PyTorch forward() verification: val r={torch_r:.6f}")
    print(f"Consistency check (numpy vs torch): diff={abs(best_val_r - torch_r):.6f}")

    return model
