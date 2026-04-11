# Mobile BCI Dataset -- Pretraining Pipeline Compatibility Analysis

## Dataset Location
`/home/andrew/eeg/data/raw/mobile_bci_ear/` -- 298 .mat files, 18 subjects

## File Naming Convention
`s{NN}_{type}_{paradigm}_{speed}.mat` where:
- type: `scalp` (36ch raw, 32ch preprocessed) or `ear` (14ch cEEGrid)
- paradigm: `ERP` or `SSVEP`
- speed: `0.0`, `0.8`, `1.6`, `2.0` (m/s), or `tr` (training)

Each subject has ~9 ear files + ~9 scalp files = ~18 files total.

## .mat File Structure

| Field | Scalp Example | Ear Example |
|-------|---------------|-------------|
| `raw_x` | (229637, 36) float64 | (341693, 14) int16 |
| `raw_fs` | 500 Hz | 500 Hz |
| `raw_clab` | 36 labels (32 EEG + 4 EOG) | 14 labels (L1-L10, R1-R8) |
| `preprocess_x` | (100, 32, 200) float64 | (100, 14, 300) float64 |
| `preprocess_fs` | 100 Hz | 100 Hz |
| `interp_clab` | e.g. 'AFz' | empty |

## Channel Ordering

### Scalp raw_clab (36 channels):
```
Fp1, Fp2, AFz, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6,
C3, Cz, C4, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8,
PO7, PO3, POz, PO4, PO8, O1, Oz, O2,
HEOGL, HEOGR, VEOGU, VEOGL   <-- 4 EOG channels
```

### Ear raw_clab (14 channels):
```
L1, L2, L4, L5, L6, L7, L9, L10   <-- Left ear (8 cEEGrid)
R1, R2, R4, R5, R7, R8             <-- Right ear (6 cEEGrid)
```

## Issues Found in `load_mobile_bci_data()` (pretrain_unified.py:222)

### BUG 1: Only loads EAR files, never SCALP (Critical)
The loader iterates sorted .mat files and picks the FIRST file per subject ID.
Because `ear` sorts before `scalp` alphabetically, every subject gets loaded from
an ear file (14 channels) and the scalp files (36 channels) are never used.

```
Files actually loaded (verified):
  s01: s01_ear_ERP_0.0.mat  (EAR, 14ch)
  s02: s02_ear_ERP_0.0.mat  (EAR, 14ch)
  ...all 18 subjects load EAR only
```

This means the pipeline is pretraining on 14-channel ear-EEG rather than
32-channel scalp EEG. For scalp encoder pretraining, this is wrong.

### BUG 2: Only 1 recording per subject (Data waste)
The `subjects_seen` set skips all subsequent files once one loads.
Each subject has ~9 ear + ~9 scalp recordings = ~18 total.
Currently only 1 of 18 recordings per subject is used (5.6% of available data).

### BUG 3: Ear data is int16 (potential scale issue)
Ear `raw_x` is stored as `int16` (range -11637 to 15627, mean 4172).
The loader casts to float32 which preserves values, but:
- These are raw ADC counts, not microvolts
- The non-zero mean (~4172) suggests a DC offset that bandpass filtering will remove
- Scalp data is float64 (range -1602 to 1624, mean 6.6) -- likely already in microvolts
- After z-scoring in `clean_and_zscore()` this is OK, but the raw scale difference
  could cause numerical issues in `bandpass_filter()` (large int16 values in filtfilt)

### BUG 4: EOG channels included in scalp data
Scalp files have 36 channels (32 EEG + 4 EOG: HEOGL, HEOGR, VEOGU, VEOGL).
The loader does not strip EOG channels. If scalp files were loaded, the model
would train on eye movement artifacts as if they were EEG channels.

### NON-ISSUE: Transpose heuristic is correct
The loader transposes when `shape[0] > shape[1]` (i.e., more rows than columns).
Since raw_x is (T, C) with T >> C, this correctly produces (C, T).

### NON-ISSUE: No NaN values detected
Both scalp and ear raw data have 0.00% NaN.

## Preprocessing Pipeline Compatibility

The `preprocess_continuous()` pipeline (bandpass 1-45 Hz, downsample to 128 Hz,
z-score, 2s windows) is compatible with Mobile BCI data **in principle**:
- 500 Hz source rate downsamples cleanly to 128 Hz (ratio 500/128 = 125/32)
- 1-45 Hz bandpass is well within Nyquist (250 Hz)
- Z-scoring normalizes away scale differences

## Recommended Fixes

1. **Separate scalp and ear loading**: Load scalp files for scalp pretraining,
   ear files for ear pretraining. Filter by `_scalp_` or `_ear_` in filename.
2. **Load ALL recordings per subject**: Remove `subjects_seen` gate, or change
   to `(subj_id, modality)` tracking to allow one scalp + one ear per subject.
   Better: load all conditions to maximize data (~9x more data per modality).
3. **Strip EOG from scalp**: Filter `raw_clab` to exclude HEOGL/HEOGR/VEOGU/VEOGL
   before passing to `preprocess_continuous()`.
4. **Cast ear int16 to float32 before filtering**: Already done implicitly by
   `np.array(mat["raw_x"], dtype=np.float32)`, but could add explicit DC removal
   (subtract mean) before bandpass to improve numerical stability.
