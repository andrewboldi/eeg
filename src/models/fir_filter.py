"""Model 2: Spatio-Temporal FIR Filter (1D Convolution).

y_c(t) = sum_j sum_tau w_{c,j,tau} x_j(t - tau)

Equivalent to nn.Conv1d with in_channels=21, out_channels=4, kernel_size=L.

Variants:
  - Causal: padding on left only (for real-time BCI)
  - Acausal: symmetric padding (for offline analysis)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalFIR(nn.Module):
    """Spatio-temporal FIR filter via 1D convolution.

    Parameters: C_out x C_in x L (e.g., 4 x 21 x 65 = 5,460)
    """

    def __init__(
        self,
        C_in: int = 21,
        C_out: int = 4,
        filter_length: int = 65,
        mode: Literal["causal", "acausal"] = "acausal",
    ):
        super().__init__()
        self.mode = mode
        self.filter_length = filter_length

        if mode == "acausal":
            # Symmetric (same) padding
            self.conv = nn.Conv1d(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=filter_length,
                padding=filter_length // 2,
                bias=False,
            )
        else:
            # Causal: no built-in padding, we pad manually on the left
            self.conv = nn.Conv1d(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=filter_length,
                padding=0,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C_in, T)

        Returns:
            (batch, C_out, T)
        """
        if self.mode == "causal":
            # Pad left with (filter_length - 1) zeros
            x = F.pad(x, (self.filter_length - 1, 0))
        return self.conv(x)

    @property
    def filters(self) -> torch.Tensor:
        """Return the (C_out, C_in, L) filter weights."""
        return self.conv.weight.data
