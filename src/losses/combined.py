"""Loss functions for EEG reconstruction.

Losses:
  1. Time-domain MSE
  2. Spectral loss (log-magnitude FFT)
  3. Band power loss (delta, theta, alpha, beta, gamma)
  4. Combined weighted loss
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Standard EEG frequency bands (Hz)
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


class TimeDomainMSE(nn.Module):
    """Time-domain mean squared error."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, T)
            target: (B, C, T)
        """
        return torch.mean((pred - target) ** 2)


class SpectralLoss(nn.Module):
    """Spectral loss in log-magnitude FFT domain.

    L_spec = mean(|log|FFT(y)| - log|FFT(y_hat)||^2)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, T)
            target: (B, C, T)
        """
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        pred_mag = torch.log(torch.abs(pred_fft) + self.eps)
        target_mag = torch.log(torch.abs(target_fft) + self.eps)

        return torch.mean((pred_mag - target_mag) ** 2)


class BandPowerLoss(nn.Module):
    """Band power loss across standard EEG frequency bands.

    Computes power in each band via FFT and penalizes differences.
    """

    def __init__(self, fs: float = 256.0, bands: dict[str, tuple[float, float]] | None = None):
        super().__init__()
        self.fs = fs
        self.bands = bands or EEG_BANDS

    def _band_power(
        self, fft_mag_sq: torch.Tensor, freqs: torch.Tensor, low: float, high: float
    ) -> torch.Tensor:
        """Compute power in a frequency band.

        Args:
            fft_mag_sq: (B, C, F) squared magnitude spectrum
            freqs: (F,) frequency values
            low: lower band edge
            high: upper band edge
        """
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() == 0:
            return torch.zeros(fft_mag_sq.shape[:2], device=fft_mag_sq.device)
        return fft_mag_sq[:, :, mask].sum(dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, T)
            target: (B, C, T)
        """
        T = pred.shape[-1]
        freqs = torch.fft.rfftfreq(T, 1.0 / self.fs).to(pred.device)

        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        pred_power = torch.abs(pred_fft) ** 2
        target_power = torch.abs(target_fft) ** 2

        loss = torch.tensor(0.0, device=pred.device)
        for low, high in self.bands.values():
            pred_bp = self._band_power(pred_power, freqs, low, high)
            target_bp = self._band_power(target_power, freqs, low, high)
            loss = loss + torch.mean((pred_bp - target_bp) ** 2)

        return loss / len(self.bands)


class CombinedLoss(nn.Module):
    """Combined loss: L = L_time + lambda1 * L_spec + lambda2 * L_band."""

    def __init__(
        self,
        lambda_spec: float = 0.1,
        lambda_band: float = 0.1,
        fs: float = 256.0,
    ):
        super().__init__()
        self.time_loss = TimeDomainMSE()
        self.spec_loss = SpectralLoss()
        self.band_loss = BandPowerLoss(fs=fs)
        self.lambda_spec = lambda_spec
        self.lambda_band = lambda_band

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            (total_loss, dict of individual loss values for logging)
        """
        l_time = self.time_loss(pred, target)
        l_spec = self.spec_loss(pred, target)
        l_band = self.band_loss(pred, target)

        total = l_time + self.lambda_spec * l_spec + self.lambda_band * l_band

        components = {
            "loss_time": l_time.item(),
            "loss_spec": l_spec.item(),
            "loss_band": l_band.item(),
            "loss_total": total.item(),
        }
        return total, components
