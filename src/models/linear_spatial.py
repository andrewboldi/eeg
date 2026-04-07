"""Model 1: Linear Spatial Filter.

Y_hat = W @ X, where W in R^{4 x 21}.
Each output channel is a learned linear combination of all input channels.
The same weights apply at every time point (time-invariant).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearSpatialFilter(nn.Module):
    """Time-invariant linear spatial filter.

    Parameters: C_out x C_in (default: 4 x 21 = 84)
    """

    def __init__(self, C_in: int = 21, C_out: int = 4):
        super().__init__()
        self.W = nn.Linear(C_in, C_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C_in, T)

        Returns:
            (batch, C_out, T)
        """
        # (B, C_in, T) -> (B, T, C_in) -> Linear -> (B, T, C_out) -> (B, C_out, T)
        return self.W(x.transpose(1, 2)).transpose(1, 2)

    @property
    def weight_matrix(self) -> torch.Tensor:
        """Return the (C_out, C_in) weight matrix."""
        return self.W.weight.data
