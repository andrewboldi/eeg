# MOABB Dataset Download Status

Updated: 2026-04-11 11:30
MNE data dir: `~/mne_data`
Disk used: ~3.1 GB (mne_data only)
Free space: ~5.7 GB (critically low -- other session using 44GB)

## Download Script

```bash
# Resume downloads (reads status file, skips completed)
uv run python scripts/moabb_download.py
```

## Completed Datasets (50 subjects)

| Dataset | Subj | EEG Ch | Hz | Dur(s) | MB/subj | Paradigm |
|---------|------|--------|----|--------|---------|----------|
| BNCI2014_001 | 9/9 | 22 | 250 | 387 | ~74 | imagery |
| BNCI2014_002 | 14/14 | 15 | 512 | 223 | ~61 | imagery |
| BNCI2014_004 | 9/9 | 3 | 250 | 2419 | ~50 | imagery |
| BNCI2014_008 | 8/8 | 8 | 256 | 1358 | ~21 | p300 |
| BNCI2014_009 | 10/10 | 16 | 256 | 196 | ~9 | p300 |

**Total: 50 subjects, ~2.2 GB**

## In Progress: BNCI2015_001 (12 subjects, ~130MB/subj)

## Queue (62 datasets, ~900 potential subjects)

### Tier 1: BNCI (fast EU server, small files)
BNCI2015_003(10), BNCI2015_004(9), BNCI2015_006(11), BNCI2015_007(16),
BNCI2015_008(13), BNCI2015_009(21), BNCI2015_010(12), BNCI2015_012(10),
BNCI2015_013(6), BNCI2016_002(15), BNCI2019_001(10), BNCI2020_001(45),
BNCI2020_002(18), BNCI2022_001(13), BNCI2024_001(20), BNCI2025_001(20),
BNCI2025_002(10), BNCI2003_004(5)

### Tier 2: PhysionetMI (109 subj, 64ch, 160Hz, ~35MB/subj)
### Tier 3: Cho2017 (52 subj, 64ch, 512Hz, ~350MB/subj -- capped)
### Tier 4: BrainInvaders (BI2014a 64 subj, BI2015a 43 subj, etc.)
### Tier 5: Other MI/P300/SSVEP datasets

## MOABB Full Inventory (149 datasets, 3412 subjects)

Top datasets by subject count:
- PhysionetMI: 109, Liu2022EldBETA: 100, Dreyer2023: 87
- Liu2020BETA: 70, BI2014a: 64, Stieger2021: 62
- Dreyer2023A: 60, Dong2023: 59, Lee2019 (MI/ERP/SSVEP): 54 each
- Cho2017: 52, Yang2025: 51, Liu2024: 50, BNCI2020_001: 45

## Usage

```python
import moabb.datasets as ds
dataset = ds.BNCI2014_001()
data = dataset.get_data(subjects=[1])
# data[subj_id][session_id][run_id] -> mne.io.Raw
raw = data[1]['0train']['0']
print(raw.ch_names, raw.info['sfreq'])
```

## Disk Constraint Notes

- Current budget: ~3GB more (limited by other pretraining session using 44GB in data/)
- Lee2019: ~750MB/subj (too large), Stieger2021: ~2GB/subj (too large)
- PhysionetMI is ideal: 109 subjects at ~35MB each = ~3.8GB total
- Most BNCI datasets are 20-130MB per subject
