"""Preprocess raw BIDS Ear-SAAD data to broadband (1-45 Hz, 128 Hz).

Reads .set files from the BIDS dataset, applies:
1. Bandpass filter: 1-45 Hz (4th order Butterworth, zero-phase)
2. Downsample to 128 Hz (from original ~512 or 1024 Hz)
3. Channel classification: scalp / in-ear / around-ear
4. NaN interpolation for bad channels
5. Z-score normalize per channel per trial
6. Window into 2s segments (256 samples at 128 Hz) with 50% overlap

Outputs: data/processed/broadband.h5 with same format as benchmark expects.

Usage:
    uv run python scripts/preprocess_broadband.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import h5py
import mne
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BIDS_ROOT = Path("data/raw/ear_saad/bids_dataset")

# Channel classification (from the MATLAB script)
INEAR_CHANNELS = {"ELA", "ELB", "ELC", "ELE", "ELI", "ELT",
                  "ERA", "ERB", "ERC", "ERE", "ERI", "ERT"}
AROUND_EAR_PREFIXES = ("cEL", "cER")
EXCLUDE_CHANNELS = {"M1", "M2", "Fp1-cr", "EOGvu", "EOGvo", "EOGhl", "EOGhr",
                     "EOG1", "EOG2", "EOG3", "EOG4"}

# Preprocessing params
BP_LOW = 1.0
BP_HIGH = 45.0
FS_TARGET = 128  # Higher than 20 Hz, captures up to 45 Hz
WINDOW_SIZE = 256  # 2 seconds at 128 Hz
WINDOW_STRIDE = 128  # 1 second overlap


def bandpass_filter(data, fs, low=BP_LOW, high=BP_HIGH, order=4):
    """Zero-phase Butterworth bandpass. data: (C, T)"""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(data.dtype)


def downsample(data, fs_orig, fs_target):
    """Polyphase resampling. data: (C, T)"""
    if fs_orig == fs_target:
        return data
    up = int(fs_target)
    down = int(fs_orig)
    g = gcd(up, down)
    up, down = up // g, down // g
    return resample_poly(data, up, down, axis=-1).astype(data.dtype)


def load_subject_bids(subject_id: int):
    """Load raw EEG from BIDS .set file.

    Returns: data (C, T), channel_names, fs, events
    """
    sub_id = f"{subject_id:02d}"
    set_file = BIDS_ROOT / f"sub-{sub_id}" / "ses-01" / "eeg" / \
               f"sub-{sub_id}_ses-01_task-selectiveAttention_eeg.set"

    if not set_file.exists():
        raise FileNotFoundError(f"Missing: {set_file}")

    # Load with MNE
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    data = raw.get_data()  # (C, T) in volts
    fs = raw.info['sfreq']
    ch_names = raw.ch_names

    # Load events
    events_file = BIDS_ROOT / f"sub-{sub_id}" / "ses-01" / "eeg" / \
                  f"sub-{sub_id}_ses-01_task-selectiveAttention_events.tsv"
    events = None
    if events_file.exists():
        import pandas as pd
        events = pd.read_csv(events_file, sep='\t')

    return data, ch_names, fs, events


def classify_channels(ch_names):
    """Classify channels into scalp, in-ear, around-ear."""
    scalp_idx, inear_idx, around_idx = [], [], []
    scalp_names, inear_names, around_names = [], [], []

    for i, ch in enumerate(ch_names):
        ch_clean = ch.strip()
        if ch_clean in EXCLUDE_CHANNELS or 'EOG' in ch_clean.upper():
            continue
        elif ch_clean in INEAR_CHANNELS:
            inear_idx.append(i)
            inear_names.append(ch_clean)
        elif any(ch_clean.startswith(p) for p in AROUND_EAR_PREFIXES):
            around_idx.append(i)
            around_names.append(ch_clean)
        else:
            scalp_idx.append(i)
            scalp_names.append(ch_clean)

    return (scalp_idx, scalp_names), (inear_idx, inear_names), (around_idx, around_names)


def process_subject(subject_id: int):
    """Process one subject: filter, downsample, window."""
    data, ch_names, fs, events = load_subject_bids(subject_id)
    logger.info(f"Subject {subject_id}: {len(ch_names)} channels, fs={fs} Hz, "
                f"{data.shape[1]/fs:.0f}s total")

    (scalp_idx, scalp_names), (inear_idx, inear_names), (around_idx, around_names) = \
        classify_channels(ch_names)

    logger.info(f"  Channels: {len(scalp_names)} scalp, {len(inear_names)} in-ear, "
                f"{len(around_names)} around-ear")

    # Extract channel groups
    scalp_data = data[scalp_idx]    # (C_scalp, T)
    inear_data = data[inear_idx]    # (C_inear, T)

    # Convert to microvolts
    scalp_data = scalp_data * 1e6
    inear_data = inear_data * 1e6

    # Cut into trials using events
    scalp_trials, inear_trials = [], []
    if events is not None and 'onset' in events.columns:
        for _, row in events.iterrows():
            onset_samp = int(row['onset'] * fs)
            duration_samp = int(row['duration'] * fs)
            if onset_samp + duration_samp > data.shape[1]:
                continue

            s_trial = scalp_data[:, onset_samp:onset_samp + duration_samp].copy()
            i_trial = inear_data[:, onset_samp:onset_samp + duration_samp].copy()

            # Bandpass filter
            s_trial = bandpass_filter(s_trial, fs, BP_LOW, BP_HIGH)
            i_trial = bandpass_filter(i_trial, fs, BP_LOW, BP_HIGH)

            # Downsample
            s_trial = downsample(s_trial, fs, FS_TARGET)
            i_trial = downsample(i_trial, fs, FS_TARGET)

            # Interpolate NaNs
            for arr in [s_trial, i_trial]:
                for ch_i in range(arr.shape[0]):
                    row_data = arr[ch_i]
                    nans = np.isnan(row_data)
                    if nans.all():
                        row_data[:] = 0.0
                    elif nans.any():
                        good = ~nans
                        row_data[nans] = np.interp(
                            np.flatnonzero(nans), np.flatnonzero(good), row_data[good]
                        )

            # Z-score per channel
            s_trial = (s_trial - s_trial.mean(1, keepdims=True)) / \
                      (s_trial.std(1, keepdims=True) + 1e-8)
            i_trial = (i_trial - i_trial.mean(1, keepdims=True)) / \
                      (i_trial.std(1, keepdims=True) + 1e-8)

            scalp_trials.append(s_trial.astype(np.float32))
            inear_trials.append(i_trial.astype(np.float32))
    else:
        # No events: treat as single trial
        s_trial = bandpass_filter(scalp_data, fs, BP_LOW, BP_HIGH)
        i_trial = bandpass_filter(inear_data, fs, BP_LOW, BP_HIGH)
        s_trial = downsample(s_trial, fs, FS_TARGET)
        i_trial = downsample(i_trial, fs, FS_TARGET)
        s_trial = (s_trial - s_trial.mean(1, keepdims=True)) / \
                  (s_trial.std(1, keepdims=True) + 1e-8)
        i_trial = (i_trial - i_trial.mean(1, keepdims=True)) / \
                  (i_trial.std(1, keepdims=True) + 1e-8)
        scalp_trials.append(s_trial.astype(np.float32))
        inear_trials.append(i_trial.astype(np.float32))

    # Window trials
    all_scalp, all_inear = [], []
    for s, i in zip(scalp_trials, inear_trials):
        T = s.shape[1]
        for start in range(0, T - WINDOW_SIZE + 1, WINDOW_STRIDE):
            all_scalp.append(s[:, start:start + WINDOW_SIZE])
            all_inear.append(i[:, start:start + WINDOW_SIZE])

    if not all_scalp:
        logger.warning(f"  Subject {subject_id}: no windows!")
        return None, None, scalp_names, inear_names

    scalp_windows = np.stack(all_scalp)
    inear_windows = np.stack(all_inear)
    logger.info(f"  -> {scalp_windows.shape[0]} windows, "
                f"scalp {scalp_windows.shape}, inear {inear_windows.shape}")

    return scalp_windows, inear_windows, scalp_names, inear_names


def main():
    out_path = Path("data/processed/broadband.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_subjects = {}
    C_scalp = C_inear = None

    for subj in range(1, 16):
        try:
            sw, iw, s_names, i_names = process_subject(subj)
            if sw is not None:
                all_subjects[subj] = (sw, iw)
                if C_scalp is None:
                    C_scalp = sw.shape[1]
                    C_inear = iw.shape[1]
                    logger.info(f"Channel config: {C_scalp} scalp -> {C_inear} in-ear")
                    logger.info(f"Scalp: {s_names}")
                    logger.info(f"In-ear: {i_names}")
        except Exception as e:
            logger.warning(f"Subject {subj} failed: {e}")

    if not all_subjects:
        raise RuntimeError("No subjects loaded successfully!")

    # Save per-subject data for LOSO benchmark
    with h5py.File(out_path, "w") as f:
        f.attrs["fs"] = FS_TARGET
        f.attrs["bp_low"] = BP_LOW
        f.attrs["bp_high"] = BP_HIGH
        f.attrs["window_size"] = WINDOW_SIZE
        f.attrs["window_stride"] = WINDOW_STRIDE
        f.attrs["C_scalp"] = int(C_scalp)
        f.attrs["C_inear"] = int(C_inear)

        for subj, (sw, iw) in all_subjects.items():
            grp = f.create_group(f"subject_{subj:02d}")
            grp.create_dataset("scalp", data=sw, compression="gzip")
            grp.create_dataset("inear", data=iw, compression="gzip")
            grp.attrs["n_windows"] = sw.shape[0]

    total_windows = sum(sw.shape[0] for sw, _ in all_subjects.values())
    logger.info(f"Saved {len(all_subjects)} subjects, {total_windows} total windows "
                f"to {out_path}")
    logger.info(f"Data shape: scalp ({C_scalp}, {WINDOW_SIZE}), "
                f"inear ({C_inear}, {WINDOW_SIZE})")
    logger.info(f"Frequency: {BP_LOW}-{BP_HIGH} Hz at {FS_TARGET} Hz")


if __name__ == "__main__":
    main()
