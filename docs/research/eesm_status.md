# EESM Sleep Dataset Status

## Disk Space
- Available at download time: 27 GB free on /dev/nvme0n1p5 (692 GB total)
- After download: 19 GB free
- Downloaded: ~4.6 GB total (subset only)

## Dataset Sizes (Full)
| Dataset | OpenNeuro ID | Total Size | EEG-only Size | Subjects | Sessions/subj |
|---------|-------------|------------|---------------|----------|---------------|
| EESM19 | ds005185 | 841 GB | ~136 GB (PSG) + 39 GB (ear-only) | 20 | 4 PSG + up to 12 ear-only |
| EESM23 | ds005178 | 95 GB | ~25.7 GB | 10 | 2 PSG + ~10 ear-only |

**CRITICAL: Full datasets far exceed available disk space. Only a subset was downloaded.**

## What Was Downloaded
- EESM19: metadata + sub-001/ses-001 + sub-002/ses-001 (PSG sleep EEG only, ~3.3 GB)
- EESM23: metadata + sub-001/ses-001 + sub-002/ses-001 (PSG + earEEG, ~1.3 GB)
- Paths: `data/raw/eesm19/`, `data/raw/eesm23/`

## Download Commands (for more data)
```bash
# EESM19 - one subject, one PSG session (~1.7 GB each)
aws s3 sync --no-sign-request s3://openneuro.org/ds005185/sub-XXX/ses-001/eeg/ \
  data/raw/eesm19/sub-XXX/ses-001/eeg/ \
  --exclude "*" --include "*task-sleep_acq-PSG*"

# EESM23 - one subject, one session (~0.6 GB each)
aws s3 sync --no-sign-request s3://openneuro.org/ds005178/sub-XXX/ses-001/eeg/ \
  data/raw/eesm23/sub-XXX/ses-001/eeg/ \
  --exclude "*" --include "*eeg.set" --include "*eeg.json" --include "*channels.tsv" --include "*scoring*"
```

## EESM19 (ds005185) - PRIMARY DATASET

### Overview
- 20 subjects, each with 4 PSG nights (sessions 1-4) + up to 12 ear-only nights
- Recorded 2018-2020, Aarhus University
- Home recordings with partial PSG + ear-EEG + actigraphy
- Paper: https://doi.org/10.1038/s41598-019-53115-3

### Channels (25 total in PSG recordings)
**In-ear EEG (12 channels):**
- Left ear: ELA, ELB, ELC, ELT, ELE, ELI
- Right ear: ERA, ERB, ERC, ERT, ERE, ERI

**Scalp EEG (8 channels, 10-20 system):**
- M1, F3, C3, O1, M2, F4, C4, O2

**Other (5 channels):**
- EOGr, EOGl (electrooculography)
- EMGl, EMGr, EMGc (electromyography)

### Technical Details
- **Sample rate: 500 Hz**
- **Format: EEGLAB .set + .fdt (MATLAB/HDF5)**
- **Duration: ~8 hours per session**
- **Reference: average**
- All channels in one file (PSG recordings include ear-EEG)
- NaN values present for faulty electrodes (shielding issues identified and marked)
- No preprocessing applied beyond electrode rejection

### Relevance to Scalp-to-Ear Prediction
**HIGH** - This is the ideal dataset:
- Same 12 in-ear channels as Ear-SAAD (ELA-ELI, ERA-ERI)
- 8 scalp channels (subset of 10-20) recorded simultaneously
- 500 Hz broadband (vs 20 Hz in current Ear-SAAD preprocessing)
- 20 subjects x 4 nights = 80 sessions of paired scalp+ear data
- Sleep data provides diverse brain states (wake, N1, N2, N3, REM)

## EESM23 (ds005178) - SECONDARY DATASET

### Overview
- 10 subjects, 2 PSG nights + ~10 ear-only nights each
- Recorded 2020-2022, Aarhus University
- Paper: associated with EESM series

### Channels
**PSG recording (13 channels - NO ear-EEG in PSG file):**
- EOGr, EOGl, EMGl, EMGr, M1, F3, C3, O1, M2, F4, C4, O2, EMGc

**Ear-EEG recording (5 channels, SEPARATE file):**
- RB, RT, LB, LT, ELE (only 4 ear channels + 1 electrode)

### Technical Details
- **Sample rate: 250 Hz**
- **Format: EEGLAB .set (self-contained, no .fdt)**
- **Duration: ~6.8 hours per session**
- **Reference: average**
- PSG and ear-EEG in separate files (different devices, need time alignment)
- Recording durations differ slightly (24337 vs 24329 seconds)

### Relevance to Scalp-to-Ear Prediction
**MODERATE** - Usable but limited:
- Only 4 ear-EEG channels (vs 12 in EESM19/Ear-SAAD)
- Different channel naming (RB/RT/LB/LT vs ELA-ERI naming)
- Separate recordings require temporal alignment via triggers
- Only 2 sessions per subject have paired PSG+ear data

## Comparison with Current Ear-SAAD Dataset

| Feature | Ear-SAAD (current) | EESM19 | EESM23 |
|---------|-------------------|--------|--------|
| Subjects | 15 | 20 | 10 |
| Scalp channels | 27 (full 10-20) | 8 (partial 10-20) | 8 (partial 10-20) |
| In-ear channels | 12 | 12 (same naming) | 4 (different naming) |
| Sample rate | 256 Hz (preprocessed to 20 Hz) | 500 Hz | 250 Hz |
| Task | Auditory attention | Sleep | Sleep |
| Duration/session | ~30 min | ~8 hours | ~6.8 hours |
| Total data | ~7.5 hours | ~640 hours (PSG only) | ~136 hours |
| Paired scalp+ear | Yes (same file) | Yes (same file) | Needs alignment |

## Key Implications for the Project

1. **EESM19 is the priority dataset** - same 12 ear channels, paired with scalp in one file
2. **Scalp coverage is limited** - only 8 of 27 scalp channels available (M1/M2, F3/F4, C3/C4, O1/O2)
3. **Broadband data** - 500 Hz gives access to full EEG spectrum (delta through gamma)
4. **Massive data volume** - even 2 subjects = ~16h of data (vs ~7.5h total in Ear-SAAD)
5. **Sleep vs auditory task** - different brain dynamics, may need domain adaptation
6. **Storage is the bottleneck** - need ~136 GB for all EESM19 PSG data, only 19 GB free
7. **Potential approach**: download 5-10 subjects' PSG data incrementally as disk space allows, or process/compress on the fly
