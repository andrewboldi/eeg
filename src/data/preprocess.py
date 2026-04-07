"""EEG preprocessing pipeline.

Steps:
  1. Band-pass filter: 0.5–45 Hz (4th order Butterworth, zero-phase)
  2. Notch filter: 50 Hz (or 60 Hz)
  3. Downsample to 256 Hz
  4. Re-reference scalp channels to common average
  5. Artifact rejection: reject windows where |V| > 150 µV
  6. Z-score normalize per channel per subject
  7. Segment into fixed windows
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from math import gcd

logger = logging.getLogger(__name__)


def bandpass_filter(
    data: NDArray,
    fs: float,
    low: float = 0.5,
    high: float = 45.0,
    order: int = 4,
) -> NDArray:
    """Apply zero-phase Butterworth bandpass filter.

    Args:
        data: (n_channels, n_samples)
        fs: sampling frequency
        low: low cutoff frequency
        high: high cutoff frequency
        order: filter order
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(data.dtype)


def notch_filter(
    data: NDArray,
    fs: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> NDArray:
    """Apply notch filter to remove power line interference."""
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data, axis=-1).astype(data.dtype)


def downsample(data: NDArray, fs_orig: float, fs_target: float = 256.0) -> NDArray:
    """Downsample data using polyphase resampling."""
    if fs_orig == fs_target:
        return data
    ratio = fs_orig / fs_target
    # Find integer up/down factors
    up = int(fs_target)
    down = int(fs_orig)
    g = gcd(up, down)
    up, down = up // g, down // g
    return resample_poly(data, up, down, axis=-1).astype(data.dtype)


def common_average_reference(data: NDArray) -> NDArray:
    """Re-reference to common average."""
    return data - data.mean(axis=0, keepdims=True)


def reject_artifacts(
    scalp: NDArray,
    inear: NDArray,
    threshold_uv: float = 150.0,
    window_size: int = 256,
    stride: int = 128,
) -> tuple[NDArray, NDArray]:
    """Reject windows where any channel exceeds threshold.

    Args:
        scalp: (C_in, T_total)
        inear: (C_out, T_total)
        threshold_uv: rejection threshold in µV
        window_size: samples per window
        stride: stride between windows

    Returns:
        Tuple of (clean_scalp_windows, clean_inear_windows) each (N, C, window_size)
    """
    n_samples = scalp.shape[1]
    scalp_windows = []
    inear_windows = []

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        s_win = scalp[:, start:end]
        i_win = inear[:, start:end]

        # Reject if any channel exceeds threshold
        if np.max(np.abs(s_win)) > threshold_uv:
            continue
        if np.max(np.abs(i_win)) > threshold_uv:
            continue

        scalp_windows.append(s_win)
        inear_windows.append(i_win)

    if not scalp_windows:
        logger.warning("All windows rejected during artifact rejection!")
        return np.empty((0, scalp.shape[0], window_size)), np.empty((0, inear.shape[0], window_size))

    return np.stack(scalp_windows), np.stack(inear_windows)


def zscore_normalize(data: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Z-score normalize per channel.

    Args:
        data: (N_windows, C, T) or (C, T)

    Returns:
        (normalized_data, means, stds)
    """
    if data.ndim == 3:
        # (N, C, T) -> compute stats over N and T
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True)
    else:
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)

    std = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std, mean.squeeze(), std.squeeze()


def preprocess_raw(
    scalp_data: NDArray,
    inear_data: NDArray,
    fs: float,
    *,
    target_fs: float = 256.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 45.0,
    notch_freq: float = 50.0,
    artifact_threshold: float = 150.0,
    window_size: int = 256,
    stride: int = 128,
    normalize: bool = True,
) -> dict:
    """Full preprocessing pipeline.

    Args:
        scalp_data: (C_in, T) raw scalp EEG
        inear_data: (C_out, T) raw in-ear EEG
        fs: original sampling frequency
        target_fs: target sampling frequency after downsampling
        bandpass_low: bandpass low cutoff
        bandpass_high: bandpass high cutoff
        notch_freq: notch filter frequency (50 or 60 Hz)
        artifact_threshold: artifact rejection threshold in µV
        window_size: window size in samples (at target_fs)
        stride: stride in samples (at target_fs)
        normalize: whether to z-score normalize

    Returns:
        Dictionary with 'scalp', 'inear', and normalization stats
    """
    # 1. Bandpass filter
    scalp_data = bandpass_filter(scalp_data, fs, bandpass_low, bandpass_high)
    inear_data = bandpass_filter(inear_data, fs, bandpass_low, bandpass_high)

    # 2. Notch filter
    scalp_data = notch_filter(scalp_data, fs, notch_freq)
    inear_data = notch_filter(inear_data, fs, notch_freq)

    # 3. Downsample
    scalp_data = downsample(scalp_data, fs, target_fs)
    inear_data = downsample(inear_data, fs, target_fs)

    # 4. Common average reference (scalp only)
    scalp_data = common_average_reference(scalp_data)

    # 5. Artifact rejection + windowing
    scalp_windows, inear_windows = reject_artifacts(
        scalp_data, inear_data, artifact_threshold, window_size, stride
    )

    result = {"scalp": scalp_windows, "inear": inear_windows}

    # 6. Z-score normalize
    if normalize and scalp_windows.shape[0] > 0:
        scalp_windows, s_mean, s_std = zscore_normalize(scalp_windows)
        inear_windows, i_mean, i_std = zscore_normalize(inear_windows)
        result.update(
            scalp=scalp_windows,
            inear=inear_windows,
            scalp_mean=s_mean,
            scalp_std=s_std,
            inear_mean=i_mean,
            inear_std=i_std,
        )

    logger.info(
        "Preprocessing complete: %d clean windows of shape (%d, %d) -> (%d, %d)",
        scalp_windows.shape[0],
        scalp_windows.shape[1],
        scalp_windows.shape[2],
        inear_windows.shape[1],
        inear_windows.shape[2],
    )
    return result
