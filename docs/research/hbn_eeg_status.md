# HBN-EEG Dataset Download Status

## Source
- **Bucket**: `s3://fcp-indi/data/Projects/HBN/EEG/`
- **Access**: `--no-sign-request` (public, no registration)
- **Total subjects in bucket**: ~4,575

## Pilot Download (2026-04-11)

Downloaded **20 subjects** (RestingState preprocessed .mat only).

### Storage
- **Location**: `data/raw/hbn_eeg/{SUBJECT_ID}/RestingState.mat`
- **Total size**: 1.5 GB
- **Disk free after download**: 29 GB

### Data Format
- **Format**: EEGLAB `.set` structure stored as MATLAB `.mat` (v5)
- **Load with**: `scipy.io.loadmat(path, squeeze_me=True)['result']`
- **Key fields**: `data`, `srate`, `nbchan`, `pnts`, `chanlocs`, `event`

### Recording Parameters
| Parameter | Value |
|-----------|-------|
| Channels | 111 (EGI HydroCel GSN 128, minus reference/peripheral) |
| Channel labels | E1-E124 + Cz (some numbers skipped) |
| Sample rate | 500 Hz |
| Duration | 243-639 seconds per subject (median ~365s) |
| Preprocessing | ICA-cleaned, bad channels removed, re-referenced |
| Reference | Cz |

### Subjects Downloaded
| Subject ID | Duration (s) | Size (MB) |
|------------|-------------|-----------|
| NDARAA075AMK | 364.2 | 68.7 |
| NDARAA112DMH | 243.1 | 46.5 |
| NDARAA117NEJ | 368.5 | 70.2 |
| NDARAA948VFH | 347.0 | 67.3 |
| NDARAB793GL3 | 407.6 | 78.3 |
| NDARAC349YUC | 389.4 | 75.2 |
| NDARAC350BZ0 | 433.6 | 82.8 |
| NDARAC853DTE | 355.5 | 68.0 |
| NDARAC904DMU | 403.5 | 76.4 |
| NDARAD232HVV | 340.0 | 65.2 |
| NDARAD481FXF | 347.3 | 65.8 |
| NDARAD615WLJ | 353.5 | 67.3 |
| NDARAD653RYE | 639.3 | 122.7 |
| NDARAD774HAZ | 409.5 | 78.6 |
| NDARAE012DGA | 355.1 | 67.9 |
| NDARAE199TDD | 354.8 | 67.6 |
| NDARAE828CML | 349.9 | 66.5 |
| NDARAEZ493ZJ | 424.5 | 82.1 |
| NDARAG143ARJ | 385.5 | 73.9 |
| NDARAG340ERT | 377.6 | 67.4 |

### Availability Notes
- ~26% of subjects in the bucket have RestingState recordings
- Other paradigms available: Video (DM/FF/WK), SAIIT_2AFC, SurroundSupp, WISC_ProcSpeed, vis_learn
- Both raw and preprocessed versions available in CSV and MAT formats
- Behavioral data and eyetracking also available per subject

## Batch Download (2026-04-11)

Expanded from 20 to **138 subjects** (118 new RestingState downloads).

### Download Statistics
| Metric | Value |
|--------|-------|
| New subjects downloaded | 118 |
| Subjects scanned on S3 | ~383 (first portion of alphabet) |
| Subjects skipped (no RestingState) | ~265 |
| Failed downloads | 0 |
| RestingState availability rate | ~31% of scanned subjects |

### Storage After Batch Download
- **Total subjects**: 138
- **Total size**: 11 GB
- **Average per subject**: ~80 MB
- **Disk free after download**: 9 GB
- **Download stopped**: background task timeout (10 min limit); disk approaching 5GB safety margin

### Download Script
Script saved at `scripts/download_hbn_batch.sh` for future use.

## Scaling Up
To download more subjects:
- Need to free disk space first (currently only 9GB free, 5GB safety margin)
- ~4,200 subjects remain unscanned on S3
- Expected ~31% will have RestingState data (~1,300 more subjects available)
- At ~80MB each, full dataset would need ~100GB total

```bash
# Resume downloading (script skips existing subjects automatically)
bash scripts/download_hbn_batch.sh
```

Estimated full dataset (all ~1,200 subjects with RestingState): ~100 GB
