"""Closed-form linear spatial filter solution.

W* = R_YX @ R_XX^{-1}

where:
  R_YX = (1/N) sum_i Y_i @ X_i^T   in R^{4 x 21}
  R_XX = (1/N) sum_i X_i @ X_i^T   in R^{21 x 21}

This serves as the baseline — every learned model must beat this.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


class ClosedFormLinear(nn.Module):
    """Closed-form optimal linear spatial filter.

    Computes W* = R_YX @ inv(R_XX) from training data, then applies it
    as a fixed (non-trainable) linear layer.
    """

    def __init__(self, C_in: int = 21, C_out: int = 4):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        # Will be set by fit()
        self.register_buffer("W", torch.zeros(C_out, C_in))

    def fit(
        self,
        scalp: NDArray | torch.Tensor,
        inear: NDArray | torch.Tensor,
        reg: float = 1e-4,
    ) -> None:
        """Compute the closed-form solution from training data.

        Args:
            scalp: (N, C_in, T) training scalp windows
            inear: (N, C_out, T) training in-ear windows
            reg: Tikhonov regularization (added to diagonal of R_XX)
        """
        if isinstance(scalp, torch.Tensor):
            scalp = scalp.numpy()
        if isinstance(inear, torch.Tensor):
            inear = inear.numpy()

        scalp = scalp.astype(np.float64)
        inear = inear.astype(np.float64)

        N, C_in, T = scalp.shape
        _, C_out, _ = inear.shape

        # R_XX = (1/N) sum X_i @ X_i^T, shape (C_in, C_in)
        # R_YX = (1/N) sum Y_i @ X_i^T, shape (C_out, C_in)
        R_XX = np.zeros((C_in, C_in))
        R_YX = np.zeros((C_out, C_in))

        for i in range(N):
            R_XX += scalp[i] @ scalp[i].T  # (C_in, T) @ (T, C_in) = (C_in, C_in)
            R_YX += inear[i] @ scalp[i].T  # (C_out, T) @ (T, C_in) = (C_out, C_in)

        R_XX /= N
        R_YX /= N

        # Tikhonov regularization
        R_XX += reg * np.eye(C_in)

        # W* = R_YX @ inv(R_XX)
        W_star = R_YX @ np.linalg.inv(R_XX)

        self.W = torch.tensor(W_star, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C_in, T)

        Returns:
            (batch, C_out, T)
        """
        # W: (C_out, C_in), x: (B, C_in, T)
        # einsum: oc,bct -> bot
        return torch.einsum("oc,bct->bot", self.W, x)

    @property
    def weight_matrix(self) -> torch.Tensor:
        """Return the (C_out, C_in) weight matrix."""
        return self.W
