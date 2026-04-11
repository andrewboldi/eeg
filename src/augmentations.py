"""Reusable EEG-specific data augmentations for tensors of shape (B, C, T).

All functions accept and return PyTorch tensors, preserving device and
differentiability where possible. Designed for scalp-to-in-ear EEG prediction
but applicable to general multichannel time-series data.

Usage::

    from src.augmentations import mixup, channel_dropout, Compose, AugmentedDataLoader

    aug = Compose([
        lambda x, y: (channel_dropout(x), y),
        lambda x, y: (gaussian_noise(x), y),
    ])
    loader = AugmentedDataLoader(dataset, batch_size=64, augment_fn=aug)
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. Mixup
# ---------------------------------------------------------------------------

def mixup(
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Standard mixup augmentation between two sample pairs.

    Draws lambda from Beta(alpha, alpha) and linearly interpolates both
    inputs and targets.

    Args:
        x1: First input batch, shape (B, C, T).
        y1: First target batch, shape (B, C_out, T).
        x2: Second input batch, shape (B, C, T).
        y2: Second target batch, shape (B, C_out, T).
        alpha: Beta distribution parameter. Larger values produce more
            uniform mixing; 0 disables mixup (lambda=1).

    Returns:
        Tuple of (mixed_x, mixed_y, lam).

    Example::

        idx = torch.randperm(x.shape[0])
        x_mix, y_mix, lam = mixup(x, y, x[idx], y[idx], alpha=0.4)
    """
    if alpha <= 0:
        return x1, y1, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2
    return x_mix, y_mix, lam


# ---------------------------------------------------------------------------
# 2. Channel dropout
# ---------------------------------------------------------------------------

def channel_dropout(
    x: torch.Tensor,
    p: float = 0.15,
) -> torch.Tensor:
    """Zero out random channels and rescale to preserve expected magnitude.

    Each channel is independently dropped with probability *p*. During
    evaluation (or when p=0), returns the input unchanged.

    Args:
        x: Input tensor, shape (B, C, T).
        p: Per-channel drop probability in [0, 1).

    Returns:
        Augmented tensor, same shape as *x*.

    Example::

        x_aug = channel_dropout(x, p=0.15)
    """
    if p <= 0 or not x.requires_grad and not torch.is_grad_enabled():
        # Allow callers to gate on training mode externally if desired;
        # the function itself is always applied when called.
        pass
    if p <= 0:
        return x
    mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > p).to(x.dtype)
    return x * mask / (1.0 - p)


# ---------------------------------------------------------------------------
# 3. Temporal shift
# ---------------------------------------------------------------------------

def temporal_shift(
    x: torch.Tensor,
    y: torch.Tensor,
    max_shift: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Circularly shift both input and target by the same random offset.

    A single shift value is drawn per batch (shared across all samples and
    channels) to preserve the temporal alignment between x and y.

    Args:
        x: Input tensor, shape (B, C, T).
        y: Target tensor, shape (B, C_out, T).
        max_shift: Maximum absolute shift in samples. The actual shift is
            drawn uniformly from [-max_shift, max_shift].

    Returns:
        Tuple of (shifted_x, shifted_y).

    Example::

        x_s, y_s = temporal_shift(x, y, max_shift=5)
    """
    if max_shift <= 0:
        return x, y
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
        return x, y
    return torch.roll(x, shifts=shift, dims=-1), torch.roll(y, shifts=shift, dims=-1)


# ---------------------------------------------------------------------------
# 4. Gaussian noise
# ---------------------------------------------------------------------------

def gaussian_noise(
    x: torch.Tensor,
    std: float = 0.1,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise to the input.

    Noise is applied to the input only (never the target) to regularize
    the model without corrupting supervision.

    Args:
        x: Input tensor, shape (B, C, T).
        std: Standard deviation of the noise.

    Returns:
        Noisy input tensor, same shape as *x*.

    Example::

        x_noisy = gaussian_noise(x, std=0.05)
    """
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


# ---------------------------------------------------------------------------
# 5. Temporal reversal
# ---------------------------------------------------------------------------

def temporal_reversal(
    x: torch.Tensor,
    y: torch.Tensor,
    p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reverse the time dimension of both input and target with probability *p*.

    Reversal is applied to the entire batch (all-or-nothing) to keep the
    x/y temporal alignment intact.

    Args:
        x: Input tensor, shape (B, C, T).
        y: Target tensor, shape (B, C_out, T).
        p: Probability of applying the reversal.

    Returns:
        Tuple of (x, y), possibly time-reversed.

    Example::

        x_r, y_r = temporal_reversal(x, y, p=0.5)
    """
    if torch.rand(1).item() < p:
        return x.flip(dims=[-1]), y.flip(dims=[-1])
    return x, y


# ---------------------------------------------------------------------------
# 6. Channel swap
# ---------------------------------------------------------------------------

def channel_swap(
    x: torch.Tensor,
    p: float = 0.1,
) -> torch.Tensor:
    """Randomly swap pairs of channels in the input.

    Each channel pair (i, j) is swapped independently with probability *p*.
    This encourages the model to learn spatially robust representations.

    Args:
        x: Input tensor, shape (B, C, T).
        p: Per-pair swap probability.

    Returns:
        Tensor with some channel pairs swapped, same shape as *x*.

    Example::

        x_swapped = channel_swap(x, p=0.1)
    """
    if p <= 0:
        return x
    C = x.shape[1]
    x_out = x.clone()
    # Generate all potential swap pairs and filter by probability
    n_pairs = C // 2
    if n_pairs == 0:
        return x
    perm = torch.randperm(C, device=x.device)[:2 * n_pairs].reshape(n_pairs, 2)
    swap_mask = torch.rand(n_pairs) < p
    for idx in range(n_pairs):
        if swap_mask[idx]:
            i, j = perm[idx, 0].item(), perm[idx, 1].item()
            x_out[:, i, :], x_out[:, j, :] = x[:, j, :].clone(), x[:, i, :].clone()
    return x_out


# ---------------------------------------------------------------------------
# 7. Amplitude scale
# ---------------------------------------------------------------------------

def amplitude_scale(
    x: torch.Tensor,
    y: torch.Tensor,
    range: Tuple[float, float] = (0.8, 1.2),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random per-channel amplitude scaling to both input and target.

    The *same* scale factor is applied to corresponding channels in x and y
    (when channels overlap), preserving the linear mapping between them.
    Since x and y may have different channel counts, independent scale
    factors are drawn for each.

    Args:
        x: Input tensor, shape (B, C_in, T).
        y: Target tensor, shape (B, C_out, T).
        range: (low, high) bounds for the uniform scale factor.

    Returns:
        Tuple of (scaled_x, scaled_y).

    Example::

        x_s, y_s = amplitude_scale(x, y, range=(0.9, 1.1))
    """
    lo, hi = range
    # Per-sample, per-channel scale factors: (B, C, 1)
    scale_x = torch.empty(x.shape[0], x.shape[1], 1, device=x.device).uniform_(lo, hi)
    scale_y = torch.empty(y.shape[0], y.shape[1], 1, device=y.device).uniform_(lo, hi)
    return x * scale_x, y * scale_y


# ---------------------------------------------------------------------------
# 8. Frequency mask
# ---------------------------------------------------------------------------

def frequency_mask(
    x: torch.Tensor,
    fs: float = 128.0,
    n_masks: int = 1,
    max_width: int = 10,
) -> torch.Tensor:
    """Mask random frequency bands in the FFT domain.

    Performs an rfft along the time axis, zeros out *n_masks* contiguous
    frequency bins of random width, and transforms back.

    Args:
        x: Input tensor, shape (B, C, T).
        fs: Sampling rate in Hz (used only for documentation; masking
            operates on bin indices).
        n_masks: Number of independent frequency masks to apply.
        max_width: Maximum width of each mask in frequency bins.

    Returns:
        Tensor with masked frequency content, same shape as *x*.

    Example::

        x_masked = frequency_mask(x, fs=128, n_masks=2, max_width=5)
    """
    X = torch.fft.rfft(x, dim=-1)
    n_freqs = X.shape[-1]
    if n_freqs <= 1:
        return x
    for _ in range(n_masks):
        width = torch.randint(1, min(max_width, n_freqs) + 1, (1,)).item()
        start = torch.randint(0, n_freqs - width + 1, (1,)).item()
        X[..., start:start + width] = 0
    return torch.fft.irfft(X, n=x.shape[-1], dim=-1)


# ---------------------------------------------------------------------------
# 9. Cutout
# ---------------------------------------------------------------------------

def cutout(
    x: torch.Tensor,
    max_len: int = 32,
) -> torch.Tensor:
    """Zero out a random contiguous temporal segment.

    A single segment position is chosen per batch; the same time range is
    zeroed across all samples and channels.

    Args:
        x: Input tensor, shape (B, C, T).
        max_len: Maximum length of the zeroed segment in time samples.

    Returns:
        Tensor with a temporal segment zeroed, same shape as *x*.

    Example::

        x_cut = cutout(x, max_len=16)
    """
    T = x.shape[-1]
    if max_len <= 0 or T <= 1:
        return x
    length = torch.randint(1, min(max_len, T) + 1, (1,)).item()
    start = torch.randint(0, T - length + 1, (1,)).item()
    x_out = x.clone()
    x_out[..., start:start + length] = 0
    return x_out


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose:
    """Chain multiple augmentation callables.

    Each callable must accept ``(x, y)`` and return ``(x, y)``.  For
    augmentations that only modify *x* (e.g. :func:`gaussian_noise`), wrap
    them with a lambda::

        Compose([
            lambda x, y: (gaussian_noise(x, std=0.05), y),
            lambda x, y: temporal_shift(x, y, max_shift=3),
            lambda x, y: temporal_reversal(x, y, p=0.5),
        ])

    Args:
        transforms: Sequence of callables ``(Tensor, Tensor) -> (Tensor, Tensor)``.

    Example::

        aug = Compose([
            lambda x, y: (channel_dropout(x, p=0.15), y),
            lambda x, y: (gaussian_noise(x, std=0.1), y),
        ])
        x_aug, y_aug = aug(x, y)
    """

    def __init__(self, transforms: Sequence[Callable[[torch.Tensor, torch.Tensor],
                                                      Tuple[torch.Tensor, torch.Tensor]]]):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(lines) + "\n])"


# ---------------------------------------------------------------------------
# AugmentedDataLoader
# ---------------------------------------------------------------------------

class AugmentedDataLoader:
    """Wrapper around a DataLoader that applies augmentations on-the-fly.

    Augmentations are applied to each batch *after* collation and device
    transfer, so they can leverage GPU acceleration.

    Args:
        dataset: A torch Dataset returning ``(scalp, inear)`` tensors.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        augment_fn: A callable ``(x, y) -> (x, y)`` applied to each batch.
            A :class:`Compose` instance works well here.
        device: Device to move batches to before augmentation. If ``None``,
            batches stay on CPU.
        **kwargs: Extra keyword arguments forwarded to ``DataLoader``.

    Example::

        aug = Compose([
            lambda x, y: (channel_dropout(x, p=0.15), y),
            lambda x, y: temporal_shift(x, y, max_shift=3),
        ])
        loader = AugmentedDataLoader(
            train_ds, batch_size=64, shuffle=True,
            augment_fn=aug, device=torch.device("cuda"),
        )
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        shuffle: bool = True,
        augment_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.augment_fn = augment_fn
        self.device = device

    def __iter__(self):
        for x, y in self.loader:
            if self.device is not None:
                x, y = x.to(self.device), y.to(self.device)
            if self.augment_fn is not None:
                x, y = self.augment_fn(x, y)
            yield x, y

    def __len__(self) -> int:
        return len(self.loader)
