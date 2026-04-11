# Mobile BCI Scalp+Ear Dataset — Download & Inspection Status

## Source
- **Figshare**: https://figshare.com/articles/dataset/13604078
- **Paper**: "Mobile BCI dataset of scalp- and ear-EEG with ERP and SSVEP paradigms during standing, walking, and running"
- **API**: `https://api.figshare.com/v2/articles/13604078`

## Dataset Summary

| Property | Value |
|----------|-------|
| Subjects | 18 (s01-s18) |
| Scalp channels | 32 (10-20 system: Fp1, Fp2, AFz, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO7, PO3, POz, PO4, PO8, O1, Oz, O2) |
| Ear channels | 14 cEEGrid (L1, L2, L4, L5, L6, L7, L9, L10, R1, R2, R4, R5, R7, R8) |
| Raw scalp channels | 36 (32 EEG + 4 EOG: HEOGL, HEOGR, VEOGU, VEOGL) |
| Raw sample rate | 500 Hz |
| Preprocessed sample rate | 100 Hz |
| Paradigms | ERP, SSVEP |
| Movement speeds | 0.0, 0.8, 1.6, 2.0 m/sec (standing, slow walk, fast walk, running) |
| Conditions per subject | 16 (2 paradigms x 4 speeds) + training sessions |
| Raw duration per condition | ~460-680 seconds (varies) |
| Preprocessed epochs | 100 trials x channels x timepoints (2-3s each) |
| File format | MATLAB .mat (v5, scipy.io compatible) |

## .mat File Structure

Each file contains:

| Field | Description | Example shape |
|-------|-------------|---------------|
| `raw_x` | Raw continuous EEG | (229637, 36) for scalp; (341693, 14) for ear |
| `raw_fs` | Raw sample rate | 500 Hz |
| `raw_clab` | Raw channel labels | 36 for scalp, 14 for ear |
| `preprocess_x` | Epoched preprocessed EEG | (100, 32, 200) for scalp; (100, 14, 300) for ear |
| `preprocess_fs` | Preprocessed sample rate | 100 Hz |
| `preprocess_clab` | Preprocessed channel labels | 32 for scalp, 14 for ear |
| `event` | Event markers (className, decs, y, time) | Structured array |
| `t` | Trial timing info | (1, 100) |
| `interp_clab` | Interpolated (bad) channels | e.g., ['AFz'] |

## Download Status

- **Local path**: `/home/andrew/eeg/data/raw/mobile_bci_ear/`
- **Files downloaded**: 298 (scalp + ear only, IMU excluded)
- **Total download size**: ~6.2 GB (6,372 MB)
- **IMU files excluded**: 147 files, 2,167 MB (not needed for EEG prediction)

### File counts per type:
- Ear files: 149 (~1,134 MB)
- Scalp files: 149 (~5,238 MB)

### All 18 subjects have BOTH scalp and ear data.

## Channel Mapping

### Scalp EEG (32 channels, 10-20 montage):
```
Fp1, Fp2, AFz, F7, F3, Fz, F4, F8,
FC5, FC1, FC2, FC6, C3, Cz, C4,
CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8,
PO7, PO3, POz, PO4, PO8, O1, Oz, O2
```

### Ear EEG (14 channels, cEEGrid):
```
Left ear:  L1, L2, L4, L5, L6, L7, L9, L10
Right ear: R1, R2, R4, R5, R7, R8
```

Note: These are cEEGrid around-ear electrodes, same type as our Ear-SAAD targets.

## Relevance to Our Project

### Why this dataset is valuable:
1. **Same electrode type**: cEEGrid around-ear EEG (14ch) + scalp EEG (32ch)
2. **Our task analog**: scalp (32ch) -> around-ear (14ch) prediction
3. **18 additional subjects**: combined with our 15 Ear-SAAD subjects = 33 total
4. **500 Hz raw data**: broadband signal, much richer than our 1-9 Hz / 20 Hz
5. **Continuous raw recordings**: can extract arbitrary windows for training

### Key differences from Ear-SAAD:
| | Ear-SAAD | Mobile BCI |
|--|----------|------------|
| Scalp channels | 27 | 32 |
| Ear channels | 12 (in-ear) | 14 (around-ear cEEGrid) |
| Ear type | In-ear | Around-ear (cEEGrid) |
| Sample rate | 256 Hz (raw) / 20 Hz (processed) | 500 Hz (raw) / 100 Hz (processed) |
| Band | 1-9 Hz | Broadband |
| Paradigm | Auditory attention (SAAD) | ERP + SSVEP |
| Subjects | 15 | 18 |
| Simultaneous scalp+ear | Yes | **Separate recordings** |

### CRITICAL NOTE: Scalp and ear are recorded SEPARATELY
The scalp and ear EEG are NOT simultaneous recordings. Each subject did the same
paradigm twice: once with the scalp cap and once with the ear device. This means:
- We CANNOT directly pair scalp and ear signals sample-by-sample
- We CAN use this for: domain adaptation, learning ear-EEG representations,
  transfer learning on spatial patterns, pretraining encoders
- The ERP/SSVEP paradigm creates aligned neural responses across both recordings

### Pretraining strategies:
1. **Self-supervised ear encoder**: Pretrain on 14ch ear data, transfer to 12ch in-ear
2. **Scalp encoder pretraining**: Learn scalp representations on 32ch, fine-tune on 27ch
3. **Contrastive learning**: Match ERP patterns across scalp and ear recordings
4. **Domain adaptation**: Reduce distribution shift between datasets
5. **Scalp spatial filter transfer**: 32ch and 27ch share many electrodes

## Channel Overlap with Ear-SAAD

**25 of 27 Ear-SAAD scalp channels are shared** with Mobile BCI:
```
Shared (25): C3, C4, CP1, CP2, CP5, CP6, Cz, F3, F4, F7, F8, FC1, FC2, FC5, FC6,
             Fp1, Fp2, Fz, O1, O2, P3, P4, P7, P8, Pz
Only in Mobile BCI (7): AFz, Oz, PO3, PO4, PO7, PO8, POz
Only in Ear-SAAD (2): T7, T8
```

This 25/27 overlap is excellent for pretraining scalp spatial filters.
A model trained on the 25 shared channels can directly transfer.

## Next Steps
1. Verify all downloads completed successfully
2. Write preprocessing script to extract raw continuous EEG
3. Design pretraining strategy accounting for non-simultaneous recordings
4. Test scalp encoder pretraining on 25 shared channels
