"""Preprocess raw BIDS data with ALL input channels (scalp + around-ear).

Uses 27 scalp + 19 around-ear = 46 input channels to predict 12 in-ear.
Around-ear channels are physically adjacent to in-ear and carry highly
correlated signals — they should dramatically improve prediction.

Also uses 256 Hz sampling (vs 128 Hz) to preserve more high-frequency detail.
Bandpass: 1-100 Hz (capture everything up to Nyquist).

Output: data/processed/broadband_full.h5
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BIDS_ROOT = Path("data/raw/ear_saad/bids_dataset")

INEAR_CHANNELS = {"ELA", "ELB", "ELC", "ELE", "ELI", "ELT",
                  "ERA", "ERB", "ERC", "ERE", "ERI", "ERT"}
EXCLUDE_CHANNELS = {"M1", "M2", "Fp1-cr", "EOGvu", "EOGvo", "EOGhl", "EOGhr",
                     "EOG1", "EOG2", "EOG3", "EOG4"}

BP_LOW = 1.0
BP_HIGH = 90.0  # Wide band
FS_TARGET = 256  # Higher resolution
WINDOW_SIZE = 512  # 2 seconds at 256 Hz
WINDOW_STRIDE = 256  # 1 second overlap


def bandpass_filter(data, fs, low=BP_LOW, high=BP_HIGH, order=4):
    nyq = fs / 2.0
    high_clamped = min(high, nyq - 1)
    b, a = butter(order, [low / nyq, high_clamped / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(data.dtype)


def downsample(data, fs_orig, fs_target):
    if fs_orig == fs_target:
        return data
    up = int(fs_target)
    down = int(fs_orig)
    g = gcd(up, down)
    up, down = up // g, down // g
    return resample_poly(data, up, down, axis=-1).astype(data.dtype)


def process_subject(subject_id: int):
    sub_id = f"{subject_id:02d}"
    set_file = BIDS_ROOT / f"sub-{sub_id}" / "ses-01" / "eeg" / \
               f"sub-{sub_id}_ses-01_task-selectiveAttention_eeg.set"
    events_file = BIDS_ROOT / f"sub-{sub_id}" / "ses-01" / "eeg" / \
                  f"sub-{sub_id}_ses-01_task-selectiveAttention_events.tsv"

    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    data = raw.get_data() * 1e6  # to microvolts
    fs = raw.info['sfreq']
    ch_names = raw.ch_names

    # Classify: input = scalp + around-ear, output = in-ear
    input_idx, input_names = [], []
    inear_idx, inear_names = [], []

    for i, ch in enumerate(ch_names):
        ch_clean = ch.strip()
        if ch_clean in EXCLUDE_CHANNELS or 'EOG' in ch_clean.upper():
            continue
        elif ch_clean in INEAR_CHANNELS:
            inear_idx.append(i)
            inear_names.append(ch_clean)
        else:
            # Both scalp AND around-ear go into input
            input_idx.append(i)
            input_names.append(ch_clean)

    input_data = data[input_idx]
    inear_data = data[inear_idx]

    logger.info(f"Subject {subject_id}: {len(input_names)} input ({len([n for n in input_names if not n.startswith('cE')])} scalp + "
                f"{len([n for n in input_names if n.startswith('cE')])} around-ear), "
                f"{len(inear_names)} in-ear, fs={fs} Hz")

    # Load events and cut trials
    events = pd.read_csv(events_file, sep='\t') if events_file.exists() else None

    input_trials, inear_trials = [], []
    if events is not None and 'onset' in events.columns:
        for _, row in events.iterrows():
            onset = int(row['onset'] * fs)
            dur = int(row['duration'] * fs)
            if onset + dur > data.shape[1]:
                continue

            inp = bandpass_filter(input_data[:, onset:onset+dur].copy(), fs)
            ine = bandpass_filter(inear_data[:, onset:onset+dur].copy(), fs)
            inp = downsample(inp, fs, FS_TARGET)
            ine = downsample(ine, fs, FS_TARGET)

            # NaN interpolation
            for arr in [inp, ine]:
                for c in range(arr.shape[0]):
                    nans = np.isnan(arr[c])
                    if nans.all():
                        arr[c] = 0.0
                    elif nans.any():
                        good = ~nans
                        arr[c, nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(good), arr[c, good])

            # Z-score per channel
            inp = (inp - inp.mean(1, keepdims=True)) / (inp.std(1, keepdims=True) + 1e-8)
            ine = (ine - ine.mean(1, keepdims=True)) / (ine.std(1, keepdims=True) + 1e-8)

            input_trials.append(inp.astype(np.float32))
            inear_trials.append(ine.astype(np.float32))

    # Window
    all_inp, all_ine = [], []
    for inp, ine in zip(input_trials, inear_trials):
        T = inp.shape[1]
        for start in range(0, T - WINDOW_SIZE + 1, WINDOW_STRIDE):
            all_inp.append(inp[:, start:start+WINDOW_SIZE])
            all_ine.append(ine[:, start:start+WINDOW_SIZE])

    if not all_inp:
        return None, None, input_names, inear_names

    return np.stack(all_inp), np.stack(all_ine), input_names, inear_names


def main():
    out_path = Path("data/processed/broadband_full.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subjects = {}
    C_in = C_out = None

    for subj in range(1, 16):
        try:
            inp, ine, in_names, out_names = process_subject(subj)
            if inp is not None:
                subjects[subj] = (inp, ine)
                if C_in is None:
                    C_in, C_out = inp.shape[1], ine.shape[1]
                    logger.info(f"Config: {C_in} input -> {C_out} output")
                    logger.info(f"Input: {in_names}")
                    logger.info(f"Output: {out_names}")
        except Exception as e:
            logger.warning(f"Subject {subj} failed: {e}")

    if not subjects:
        raise RuntimeError("No subjects loaded!")

    with h5py.File(out_path, "w") as f:
        f.attrs["fs"] = FS_TARGET
        f.attrs["bp_low"] = BP_LOW
        f.attrs["bp_high"] = BP_HIGH
        f.attrs["window_size"] = WINDOW_SIZE
        f.attrs["window_stride"] = WINDOW_STRIDE
        f.attrs["C_scalp"] = int(C_in)
        f.attrs["C_inear"] = int(C_out)
        for subj, (inp, ine) in subjects.items():
            grp = f.create_group(f"subject_{subj:02d}")
            grp.create_dataset("scalp", data=inp, compression="gzip")
            grp.create_dataset("inear", data=ine, compression="gzip")

    total = sum(i.shape[0] for i, _ in subjects.values())
    logger.info(f"Saved {len(subjects)} subjects, {total} windows to {out_path}")
    logger.info(f"{C_in} input -> {C_out} output, {BP_LOW}-{BP_HIGH} Hz @ {FS_TARGET} Hz")


if __name__ == "__main__":
    main()
