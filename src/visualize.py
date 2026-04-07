"""Visualization module for EEG reconstruction results.

Generates:
  1. Time trace overlay (true vs predicted)
  2. Power spectral density comparison
  3. Band power scatter plots
  4. Topomap of spatial filter weights (Model 1)
  5. Learned FIR filter impulse responses (Model 2)
  6. Training curves
  7. Metric comparison table (LaTeX)
  8. Coherence plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .data.download import SCALP_CHANNELS_10_20, IN_EAR_CHANNELS
from .metrics.evaluation import EEG_BANDS, band_power, magnitude_squared_coherence


def _save_or_show(fig: plt.Figure, path: str | Path | None, dpi: int = 150):
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_time_traces(
    pred: NDArray,
    target: NDArray,
    fs: float = 256.0,
    duration_sec: float = 5.0,
    start_sample: int = 0,
    channel_names: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Plot time trace overlay: true vs predicted for all 4 output channels.

    Args:
        pred: (C_out, T) or first window from (N, C_out, T)
        target: same shape
        fs: sampling frequency
        duration_sec: duration to plot in seconds
        start_sample: starting sample index
        channel_names: names for output channels
    """
    if pred.ndim == 3:
        # Concatenate first few windows
        pred = pred.reshape(pred.shape[1], -1)
        target = target.reshape(target.shape[1], -1)

    n_samples = int(duration_sec * fs)
    end = min(start_sample + n_samples, pred.shape[1])
    t = np.arange(start_sample, end) / fs

    C = pred.shape[0]
    if channel_names is None:
        channel_names = IN_EAR_CHANNELS[:C]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for c in range(min(C, 4)):
        ax = axes[c]
        ax.plot(t, target[c, start_sample:end], "b-", alpha=0.7, linewidth=0.8, label="True")
        ax.plot(t, pred[c, start_sample:end], "r-", alpha=0.7, linewidth=0.8, label="Predicted")
        ax.set_title(channel_names[c])
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    axes[-2].set_xlabel("Time (s)")
    fig.suptitle("Time Trace: True vs Predicted In-Ear EEG", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_psd(
    pred: NDArray,
    target: NDArray,
    fs: float = 256.0,
    channel_names: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Plot power spectral density comparison per channel.

    Args:
        pred: (C, T) or (N, C, T)
        target: same shape
    """
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[1], -1)
        target = target.reshape(target.shape[1], -1)

    C = pred.shape[0]
    if channel_names is None:
        channel_names = IN_EAR_CHANNELS[:C]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for c in range(min(C, 4)):
        ax = axes[c]
        freqs = np.fft.rfftfreq(pred.shape[1], 1.0 / fs)

        pred_psd = np.abs(np.fft.rfft(pred[c])) ** 2
        target_psd = np.abs(np.fft.rfft(target[c])) ** 2

        # Smooth with moving average
        window = min(20, len(freqs) // 10)
        if window > 1:
            kernel = np.ones(window) / window
            pred_psd_smooth = np.convolve(pred_psd, kernel, mode="same")
            target_psd_smooth = np.convolve(target_psd, kernel, mode="same")
        else:
            pred_psd_smooth = pred_psd
            target_psd_smooth = target_psd

        mask = (freqs >= 0.5) & (freqs <= 45)
        ax.semilogy(freqs[mask], target_psd_smooth[mask], "b-", alpha=0.7, label="True")
        ax.semilogy(freqs[mask], pred_psd_smooth[mask], "r-", alpha=0.7, label="Predicted")
        ax.set_title(channel_names[c])
        ax.set_ylabel("Power")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-2].set_xlabel("Frequency (Hz)")
    fig.suptitle("Power Spectral Density: True vs Predicted", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_band_power_scatter(
    pred: NDArray,
    target: NDArray,
    fs: float = 256.0,
    save_path: str | Path | None = None,
):
    """Scatter plot of true vs predicted band power per band.

    Args:
        pred: (N, C, T)
        target: (N, C, T)
    """
    bands = EEG_BANDS
    N, C, T = pred.shape

    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 4))

    for idx, (band_name, (low, high)) in enumerate(bands.items()):
        ax = axes[idx]
        pred_bp = np.array([band_power(pred[i], fs, (low, high)) for i in range(N)])
        target_bp = np.array([band_power(target[i], fs, (low, high)) for i in range(N)])

        # Flatten across channels
        pred_flat = pred_bp.flatten()
        target_flat = target_bp.flatten()

        ax.scatter(target_flat, pred_flat, alpha=0.3, s=5)

        # Reference line
        lims = [
            min(target_flat.min(), pred_flat.min()),
            max(target_flat.max(), pred_flat.max()),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)

        # R^2
        corr = np.corrcoef(target_flat, pred_flat)[0, 1]
        ax.set_title(f"{band_name}\n$r^2$={corr**2:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    fig.suptitle("Band Power: True vs Predicted", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_spatial_weights(
    W: NDArray,
    channel_names: list[str] | None = None,
    output_names: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Visualize spatial filter weight matrix as a heatmap.

    For a proper topomap, MNE with electrode positions is needed.
    This provides a heatmap fallback.

    Args:
        W: (C_out, C_in) weight matrix
    """
    if channel_names is None:
        channel_names = SCALP_CHANNELS_10_20[: W.shape[1]]
    if output_names is None:
        output_names = IN_EAR_CHANNELS[: W.shape[0]]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(output_names)))
    ax.set_yticklabels(output_names)
    ax.set_xlabel("Scalp Channel")
    ax.set_ylabel("In-Ear Channel")
    ax.set_title("Learned Spatial Filter Weights (W)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_spatial_topomaps(
    W: NDArray,
    output_names: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Plot topographic maps of spatial filter weights using MNE.

    Args:
        W: (C_out, C_in) weight matrix (C_in should be 21 for 10-20 system)
    """
    try:
        import mne
    except ImportError:
        plot_spatial_weights(W, save_path=save_path)
        return

    if output_names is None:
        output_names = IN_EAR_CHANNELS[: W.shape[0]]

    # Create MNE info with standard 10-20 positions
    ch_names_10_20 = SCALP_CHANNELS_10_20[: W.shape[1]]
    # Map A1 to standard name if needed
    ch_names_mne = [ch if ch != "A1" else "M1" for ch in ch_names_10_20]

    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info = mne.create_info(ch_names_mne, sfreq=256, ch_types="eeg")
        info.set_montage(montage, on_missing="ignore")

        fig, axes = plt.subplots(1, W.shape[0], figsize=(4 * W.shape[0], 4))
        if W.shape[0] == 1:
            axes = [axes]

        for c in range(W.shape[0]):
            mne.viz.plot_topomap(W[c], info, axes=axes[c], show=False, cmap="RdBu_r")
            axes[c].set_title(output_names[c])

        fig.suptitle("Spatial Filter Topographic Maps", fontsize=14)
        fig.tight_layout()
        _save_or_show(fig, save_path)
    except Exception:
        # Fall back to heatmap if topomap fails
        plot_spatial_weights(W, save_path=save_path)


def plot_fir_filters(
    filters: NDArray,
    fs: float = 256.0,
    top_k: int = 5,
    channel_names: list[str] | None = None,
    output_names: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Plot impulse responses of FIR filters.

    Args:
        filters: (C_out, C_in, L) filter weights
        fs: sampling frequency
        top_k: number of top contributing input channels to show per output
    """
    C_out, C_in, L = filters.shape
    if channel_names is None:
        channel_names = SCALP_CHANNELS_10_20[:C_in]
    if output_names is None:
        output_names = IN_EAR_CHANNELS[:C_out]

    t = np.arange(L) / fs * 1000  # time in ms

    fig, axes = plt.subplots(C_out, 1, figsize=(12, 3 * C_out), sharex=True)
    if C_out == 1:
        axes = [axes]

    for c in range(C_out):
        ax = axes[c]
        # Find top-k contributing channels by L2 norm
        norms = np.linalg.norm(filters[c], axis=1)
        top_indices = np.argsort(norms)[-top_k:][::-1]

        for j in top_indices:
            ax.plot(t, filters[c, j], label=f"{channel_names[j]} (‖w‖={norms[j]:.3f})")

        ax.set_title(f"Output: {output_names[c]}")
        ax.set_ylabel("Weight")
        ax.legend(fontsize=8, loc="upper right")
        ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

    axes[-1].set_xlabel("Time lag (ms)")
    fig.suptitle("FIR Filter Impulse Responses", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str | Path | None = None,
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Train")
    ax.plot(epochs, val_losses, "r-", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_coherence(
    pred: NDArray,
    target: NDArray,
    fs: float = 256.0,
    channel_names: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Plot magnitude-squared coherence vs frequency.

    Args:
        pred: (C, T) or (N, C, T) — will be concatenated if 3D
        target: same shape
    """
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[1], -1)
        target = target.reshape(target.shape[1], -1)

    C = pred.shape[0]
    if channel_names is None:
        channel_names = IN_EAR_CHANNELS[:C]

    freqs, coh = magnitude_squared_coherence(pred, target, fs)

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in range(C):
        ax.plot(freqs, coh[c], label=channel_names[c], alpha=0.8)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title("Magnitude-Squared Coherence: Predicted vs True")
    ax.set_xlim(0.5, 45)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def generate_latex_table(
    model_metrics: dict[str, dict],
    save_path: str | Path | None = None,
) -> str:
    """Generate LaTeX comparison table.

    Args:
        model_metrics: dict mapping model name to metrics dict

    Returns:
        LaTeX table string
    """
    cols = ["Pearson $r$", "RMSE", "Rel. RMSE", "SNR (dB)", "Spec. RMSE"]
    keys = ["pearson_r_mean", "rmse_mean", "relative_rmse_mean", "snr_db_mean", "spectral_rmse_mean"]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Model comparison on test set}")
    lines.append("\\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(cols) + " \\\\")
    lines.append("\\midrule")

    for model_name, metrics in model_metrics.items():
        vals = []
        for k in keys:
            v = metrics.get(k, 0.0)
            vals.append(f"{v:.4f}")
        lines.append(f"{model_name} & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table = "\n".join(lines)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table)

    return table
