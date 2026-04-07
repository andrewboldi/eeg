"""Evaluation metrics for EEG reconstruction.

Per-channel metrics:
  - Pearson correlation
  - RMSE
  - Relative RMSE (RMSE / std(y))
  - SNR (dB)

Spectral metrics:
  - Band power correlation
  - Magnitude-squared coherence
  - Spectral RMSE
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import coherence as scipy_coherence

# EEG frequency bands
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def pearson_correlation(pred: NDArray, target: NDArray) -> NDArray:
    """Per-channel Pearson correlation.

    Args:
        pred: (C, T) or (N, C, T)
        target: same shape as pred

    Returns:
        (C,) array of correlation coefficients
    """
    if pred.ndim == 3:
        # Concatenate across windows for per-channel correlation
        pred = pred.reshape(pred.shape[1], -1)
        target = target.reshape(target.shape[1], -1)

    C = pred.shape[0]
    r = np.zeros(C)
    for c in range(C):
        r[c] = np.corrcoef(pred[c], target[c])[0, 1]
    return r


def rmse(pred: NDArray, target: NDArray) -> NDArray:
    """Per-channel RMSE.

    Args:
        pred: (C, T) or (N, C, T)
        target: same shape

    Returns:
        (C,) array of RMSE values
    """
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[1], -1)
        target = target.reshape(target.shape[1], -1)

    return np.sqrt(np.mean((pred - target) ** 2, axis=-1))


def relative_rmse(pred: NDArray, target: NDArray) -> NDArray:
    """Per-channel relative RMSE (RMSE / std(target)).

    Args:
        pred: (C, T) or (N, C, T)
        target: same shape

    Returns:
        (C,) array
    """
    if pred.ndim == 3:
        target_flat = target.reshape(target.shape[1], -1)
    else:
        target_flat = target

    r = rmse(pred, target)
    std = np.std(target_flat, axis=-1)
    std = np.where(std < 1e-8, 1.0, std)
    return r / std


def snr_db(pred: NDArray, target: NDArray) -> NDArray:
    """Per-channel SNR in dB.

    SNR = 10 * log10(|y|^2 / |y - y_hat|^2)

    Args:
        pred: (C, T) or (N, C, T)
        target: same shape

    Returns:
        (C,) array of SNR values in dB
    """
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[1], -1)
        target = target.reshape(target.shape[1], -1)

    signal_power = np.mean(target**2, axis=-1)
    noise_power = np.mean((target - pred) ** 2, axis=-1)
    noise_power = np.maximum(noise_power, 1e-10)
    return 10.0 * np.log10(signal_power / noise_power)


def band_power(signal: NDArray, fs: float, band: tuple[float, float]) -> NDArray:
    """Compute power in a frequency band for each channel.

    Args:
        signal: (C, T)
        fs: sampling frequency
        band: (low_freq, high_freq)

    Returns:
        (C,) power values
    """
    freqs = np.fft.rfftfreq(signal.shape[-1], 1.0 / fs)
    fft_vals = np.fft.rfft(signal, axis=-1)
    power = np.abs(fft_vals) ** 2

    mask = (freqs >= band[0]) & (freqs < band[1])
    return power[:, mask].sum(axis=-1)


def band_power_correlation(
    pred: NDArray,
    target: NDArray,
    fs: float = 256.0,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, NDArray]:
    """Per-band Pearson correlation of band power across windows.

    Args:
        pred: (N, C, T) predicted windows
        target: (N, C, T) target windows
        fs: sampling frequency
        bands: frequency bands

    Returns:
        Dict mapping band name to (C,) correlation array
    """
    if bands is None:
        bands = EEG_BANDS

    N, C, T = pred.shape
    result = {}

    for band_name, (low, high) in bands.items():
        pred_bp = np.zeros((N, C))
        target_bp = np.zeros((N, C))

        for i in range(N):
            pred_bp[i] = band_power(pred[i], fs, (low, high))
            target_bp[i] = band_power(target[i], fs, (low, high))

        # Correlation across windows for each channel
        corr = np.zeros(C)
        for c in range(C):
            if np.std(pred_bp[:, c]) < 1e-10 or np.std(target_bp[:, c]) < 1e-10:
                corr[c] = 0.0
            else:
                corr[c] = np.corrcoef(pred_bp[:, c], target_bp[:, c])[0, 1]
        result[band_name] = corr

    return result


def magnitude_squared_coherence(
    pred: NDArray,
    target: NDArray,
    fs: float = 256.0,
    nperseg: int = 256,
) -> tuple[NDArray, NDArray]:
    """Magnitude-squared coherence between predicted and target.

    Args:
        pred: (C, T)
        target: (C, T)
        fs: sampling frequency
        nperseg: segment length for Welch method

    Returns:
        (freqs, coherence) where coherence is (C, n_freqs)
    """
    C = pred.shape[0]
    nperseg = min(nperseg, pred.shape[-1])
    freqs = None
    coh_all = []

    for c in range(C):
        f, cxy = scipy_coherence(target[c], pred[c], fs=fs, nperseg=nperseg)
        if freqs is None:
            freqs = f
        coh_all.append(cxy)

    return freqs, np.stack(coh_all)


def spectral_rmse(pred: NDArray, target: NDArray, fs: float = 256.0) -> NDArray:
    """RMSE in log-power spectral density.

    Args:
        pred: (C, T) or (N, C, T)
        target: same shape

    Returns:
        (C,) spectral RMSE values
    """
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[1], -1)
        target = target.reshape(target.shape[1], -1)

    eps = 1e-10
    pred_psd = np.log10(np.abs(np.fft.rfft(pred, axis=-1)) ** 2 + eps)
    target_psd = np.log10(np.abs(np.fft.rfft(target, axis=-1)) ** 2 + eps)

    return np.sqrt(np.mean((pred_psd - target_psd) ** 2, axis=-1))


def compute_all_metrics(
    pred: NDArray,
    target: NDArray,
    fs: float = 256.0,
) -> dict[str, any]:
    """Compute all evaluation metrics.

    Args:
        pred: (N, C, T) predicted windows
        target: (N, C, T) target windows
        fs: sampling frequency

    Returns:
        Dictionary with all metric results.
    """
    results = {}

    # Per-channel metrics
    results["pearson_r"] = pearson_correlation(pred, target)
    results["rmse"] = rmse(pred, target)
    results["relative_rmse"] = relative_rmse(pred, target)
    results["snr_db"] = snr_db(pred, target)

    # Averages
    results["pearson_r_mean"] = float(np.mean(results["pearson_r"]))
    results["rmse_mean"] = float(np.mean(results["rmse"]))
    results["relative_rmse_mean"] = float(np.mean(results["relative_rmse"]))
    results["snr_db_mean"] = float(np.mean(results["snr_db"]))

    # Spectral metrics
    results["band_power_corr"] = band_power_correlation(pred, target, fs)
    results["spectral_rmse"] = spectral_rmse(pred, target, fs)
    results["spectral_rmse_mean"] = float(np.mean(results["spectral_rmse"]))

    # Coherence (on concatenated data)
    pred_cat = pred.reshape(pred.shape[1], -1)
    target_cat = target.reshape(target.shape[1], -1)
    freqs, coh = magnitude_squared_coherence(pred_cat, target_cat, fs)
    results["coherence_freqs"] = freqs
    results["coherence"] = coh
    results["coherence_mean"] = float(np.mean(coh))

    return results


def format_metrics_table(results: dict) -> str:
    """Format metrics as a readable table."""
    lines = []
    lines.append(f"{'Metric':<25} {'Mean':>10}  Per-channel")
    lines.append("-" * 70)

    for key in ("pearson_r", "rmse", "relative_rmse", "snr_db", "spectral_rmse"):
        vals = results[key]
        mean = results[f"{key}_mean"]
        ch_str = ", ".join(f"{v:.4f}" for v in vals)
        lines.append(f"{key:<25} {mean:>10.4f}  [{ch_str}]")

    lines.append("")
    lines.append("Band power correlations:")
    for band_name, corrs in results["band_power_corr"].items():
        ch_str = ", ".join(f"{v:.4f}" for v in corrs)
        lines.append(f"  {band_name:<10} [{ch_str}]  mean={np.mean(corrs):.4f}")

    lines.append(f"\nCoherence (mean): {results['coherence_mean']:.4f}")

    return "\n".join(lines)
