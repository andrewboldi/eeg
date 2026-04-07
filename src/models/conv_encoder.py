"""Model 3: Convolutional Encoder (Nonlinear).

Architecture:
  Block 1: Spatial convolution (pointwise)
  Block 2: Temporal convolution blocks with depthwise-separable convs + residual
  Block 3: Projection to output channels
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Single temporal processing block with depthwise-separable convolution."""

    def __init__(self, H: int, K: int, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(H, H, kernel_size=K, padding=K // 2, groups=H)
        self.bn1 = nn.BatchNorm1d(H)
        self.pointwise = nn.Conv1d(H, H, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(H)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.dropout(x)
        x = self.bn2(self.pointwise(x))
        return self.act(x + residual)


class ConvEncoder(nn.Module):
    """Nonlinear convolutional encoder for scalp-to-in-ear mapping.

    Parameters: ~50K–200K depending on H.
    """

    def __init__(
        self,
        C_in: int = 21,
        C_out: int = 4,
        H: int = 64,
        K: int = 17,
        N_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(C_in, H, kernel_size=1),
            nn.BatchNorm1d(H),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *[TemporalBlock(H, K, dropout) for _ in range(N_blocks)]
        )
        self.proj = nn.Conv1d(H, C_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C_in, T)

        Returns:
            (batch, C_out, T)
        """
        x = self.spatial(x)
        x = self.temporal(x)
        return self.proj(x)
