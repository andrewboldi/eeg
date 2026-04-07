"""Synthetic data generator (Dataset C fallback).

Generates paired scalp + in-ear EEG using a known linear forward model:
    Y_synth = M @ X + epsilon

where M has high weights on temporal/parietal channels (T7, T8, P7, P8).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .download import SCALP_CHANNELS_10_20

logger = logging.getLogger(__name__)

# Default mixing matrix: in-ear channels are weighted sums of nearby scalp channels
# Channels with high contribution: T7(7), T8(11), P7(12), P8(16)
# Index mapping from SCALP_CHANNELS_10_20:
#   0:Fp1, 1:Fp2, 2:F7, 3:F3, 4:Fz, 5:F4, 6:F8,
#   7:T7, 8:C3, 9:Cz, 10:C4, 11:T8,
#   12:P7, 13:P3, 14:Pz, 15:P4, 16:P8,
#   17:O1, 18:Oz, 19:O2, 20:A1


def _default_mixing_matrix() -> NDArray:
    """Create the 4x21 mixing matrix M.

    In-ear channels:
        EarL1: mainly T7, P7, C3
        EarL2: mainly T7, F7, P7
        EarR1: mainly T8, P8, C4
        EarR2: mainly T8, F8, P8
    """
    M = np.zeros((4, 21), dtype=np.float64)

    # EarL1: left ear, deep insertion
    M[0, 7] = 0.45   # T7
    M[0, 12] = 0.30  # P7
    M[0, 8] = 0.15   # C3
    M[0, 2] = 0.05   # F7
    M[0, 17] = 0.05  # O1

    # EarL2: left ear, outer
    M[1, 7] = 0.35   # T7
    M[1, 2] = 0.25   # F7
    M[1, 12] = 0.20  # P7
    M[1, 8] = 0.10   # C3
    M[1, 20] = 0.10  # A1

    # EarR1: right ear, deep insertion
    M[2, 11] = 0.45  # T8
    M[2, 16] = 0.30  # P8
    M[2, 10] = 0.15  # C4
    M[2, 6] = 0.05   # F8
    M[2, 19] = 0.05  # O2

    # EarR2: right ear, outer
    M[3, 11] = 0.35  # T8
    M[3, 6] = 0.25   # F8
    M[3, 16] = 0.20  # P8
    M[3, 10] = 0.10  # C4
    M[3, 20] = 0.10  # A1

    return M


def generate_synthetic_scalp(
    n_channels: int = 21,
    n_samples: int = 256 * 300,  # 5 minutes at 256 Hz
    fs: float = 256.0,
    n_subjects: int = 5,
    seed: int = 42,
) -> list[NDArray]:
    """Generate synthetic scalp EEG with realistic spectral properties.

    Generates multi-channel signals with:
      - 1/f background noise
      - Alpha oscillation (~10 Hz) in posterior channels
      - Occasional eye blinks (frontal channels)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    subjects = []

    for subj in range(n_subjects):
        data = np.zeros((n_channels, n_samples), dtype=np.float64)

        # 1/f background for all channels
        freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
        for ch in range(n_channels):
            spectrum = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
            # 1/f spectrum (avoid DC)
            psd = np.ones_like(freqs)
            psd[1:] = 1.0 / np.sqrt(freqs[1:])
            spectrum *= psd
            data[ch] = np.fft.irfft(spectrum, n=n_samples)

        # Alpha oscillation (8-12 Hz) stronger in posterior channels
        alpha_freq = 9.5 + rng.uniform(-1, 1)
        alpha = np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2 * np.pi))
        # Posterior channels: O1(17), Oz(18), O2(19), P3(13), Pz(14), P4(15)
        for ch in [13, 14, 15, 17, 18, 19]:
            data[ch] += alpha * rng.uniform(3, 8)

        # Scale to roughly µV range
        data *= 20.0

        subjects.append(data.astype(np.float32))

    return subjects


def generate_synthetic_data(
    n_subjects: int = 5,
    n_samples: int = 256 * 300,
    fs: float = 256.0,
    snr_db: float = 10.0,
    seed: int = 42,
    mixing_matrix: NDArray | None = None,
) -> list[dict[str, NDArray]]:
    """Generate paired (scalp, in-ear) synthetic data.

    Args:
        n_subjects: number of synthetic subjects
        n_samples: samples per subject
        fs: sampling frequency
        snr_db: signal-to-noise ratio for in-ear signals
        seed: random seed
        mixing_matrix: (4, 21) mixing matrix; uses default if None

    Returns:
        List of dicts with 'scalp' and 'inear' arrays, each (C, T).
    """
    if mixing_matrix is None:
        mixing_matrix = _default_mixing_matrix()

    rng = np.random.default_rng(seed + 1000)
    scalp_data = generate_synthetic_scalp(
        n_channels=21, n_samples=n_samples, fs=fs, n_subjects=n_subjects, seed=seed
    )

    subjects = []
    for scalp in scalp_data:
        # Apply forward model
        inear_clean = mixing_matrix @ scalp  # (4, T)

        # Add noise at specified SNR
        signal_power = np.mean(inear_clean**2, axis=-1, keepdims=True)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.standard_normal(inear_clean.shape).astype(np.float32)
        noise *= np.sqrt(noise_power)
        inear = (inear_clean + noise).astype(np.float32)

        subjects.append({"scalp": scalp, "inear": inear})

    logger.info(
        "Generated synthetic data: %d subjects, %d samples each, SNR=%.1f dB",
        n_subjects, n_samples, snr_db,
    )
    return subjects


def get_ground_truth_mixing_matrix() -> NDArray:
    """Return the ground-truth mixing matrix for validation."""
    return _default_mixing_matrix()
