"""Preprocess broadband 46ch data with longer windows (4s, 8s).

Creates two datasets:
  - broadband_46ch_4s.h5: 512 samples (4s @ 128 Hz), stride 256
  - broadband_46ch_8s.h5: 1024 samples (8s @ 128 Hz), stride 512

More temporal context lets the model capture slower EEG dynamics
and use longer-range temporal correlations for prediction.
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

BP_LOW, BP_HIGH = 1.0, 45.0
FS_TARGET = 128

CONFIGS = [
    {"name": "4s", "window_size": 512, "window_stride": 256},
    {"name": "8s", "window_size": 1024, "window_stride": 512},
]


def process_subject(subject_id, window_size, window_stride):
    sub_id = f"{subject_id:02d}"
    set_file = BIDS_ROOT / f"sub-{sub_id}/ses-01/eeg/sub-{sub_id}_ses-01_task-selectiveAttention_eeg.set"
    events_file = BIDS_ROOT / f"sub-{sub_id}/ses-01/eeg/sub-{sub_id}_ses-01_task-selectiveAttention_events.tsv"

    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    data = raw.get_data() * 1e6
    fs = raw.info['sfreq']
    ch_names = raw.ch_names

    input_idx, inear_idx = [], []
    for i, ch in enumerate(ch_names):
        c = ch.strip()
        if c in EXCLUDE_CHANNELS or 'EOG' in c.upper():
            continue
        elif c in INEAR_CHANNELS:
            inear_idx.append(i)
        else:
            input_idx.append(i)

    inp_data, ine_data = data[input_idx], data[inear_idx]

    events = pd.read_csv(events_file, sep='\t') if events_file.exists() else None
    nyq = fs / 2.0
    b, a = butter(4, [BP_LOW/nyq, BP_HIGH/nyq], btype='band')

    all_inp, all_ine = [], []
    for _, row in (events.iterrows() if events is not None else []):
        onset = int(row['onset'] * fs)
        dur = int(row['duration'] * fs)
        if onset + dur > data.shape[1]:
            continue

        inp = filtfilt(b, a, inp_data[:, onset:onset+dur], axis=-1).astype(np.float32)
        ine = filtfilt(b, a, ine_data[:, onset:onset+dur], axis=-1).astype(np.float32)

        up, down = FS_TARGET, int(fs)
        g = gcd(up, down)
        inp = resample_poly(inp, up//g, down//g, axis=-1).astype(np.float32)
        ine = resample_poly(ine, up//g, down//g, axis=-1).astype(np.float32)

        for arr in [inp, ine]:
            for c in range(arr.shape[0]):
                nans = np.isnan(arr[c])
                if nans.all(): arr[c] = 0.0
                elif nans.any():
                    good = ~nans
                    arr[c, nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(good), arr[c, good])

        inp = (inp - inp.mean(1, keepdims=True)) / (inp.std(1, keepdims=True) + 1e-8)
        ine = (ine - ine.mean(1, keepdims=True)) / (ine.std(1, keepdims=True) + 1e-8)

        T = inp.shape[1]
        for start in range(0, T - window_size + 1, window_stride):
            all_inp.append(inp[:, start:start+window_size])
            all_ine.append(ine[:, start:start+window_size])

    if not all_inp:
        return None, None
    return np.stack(all_inp), np.stack(all_ine)


def main():
    for cfg in CONFIGS:
        out = Path(f"data/processed/broadband_46ch_{cfg['name']}.h5")
        out.parent.mkdir(parents=True, exist_ok=True)
        subjects = {}
        C_in = C_out = None

        logger.info(f"Processing {cfg['name']} windows ({cfg['window_size']} samples)...")
        for s in range(1, 16):
            try:
                inp, ine = process_subject(s, cfg['window_size'], cfg['window_stride'])
                if inp is not None:
                    subjects[s] = (inp, ine)
                    if C_in is None:
                        C_in, C_out = inp.shape[1], ine.shape[1]
                    logger.info(f"  Subject {s}: {inp.shape[0]} windows")
            except Exception as e:
                logger.warning(f"  Subject {s} failed: {e}")

        with h5py.File(out, "w") as f:
            f.attrs.update(fs=FS_TARGET, bp_low=BP_LOW, bp_high=BP_HIGH,
                           window_size=cfg['window_size'], window_stride=cfg['window_stride'],
                           C_scalp=int(C_in), C_inear=int(C_out))
            for s, (inp, ine) in subjects.items():
                grp = f.create_group(f"subject_{s:02d}")
                grp.create_dataset("scalp", data=inp, compression="gzip")
                grp.create_dataset("inear", data=ine, compression="gzip")

        total = sum(i.shape[0] for i, _ in subjects.values())
        logger.info(f"Saved {cfg['name']}: {len(subjects)} subjects, {total} windows to {out}")


if __name__ == "__main__":
    main()
