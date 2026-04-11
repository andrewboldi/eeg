# Around-Ear and In-Ear EEG Public Datasets

**Date**: 2026-04-11
**Goal**: Catalog every publicly available around-ear/in-ear EEG dataset for potential use in our scalp-to-in-ear prediction project.

---

## Summary Table

| # | Dataset | Subjects | Ear Channels | Type | Size | Access | Priority |
|---|---------|----------|-------------|------|------|--------|----------|
| 1 | **NJU cEEGrid AAD** | 98 | 16 (cEEGrid) | Around-ear | ~10 GB (est.) | IEEE DataPort subscription ($40/mo) OR free w/ IEEE Society membership | **HIGHEST** |
| 2 | **Ear-SAAD** (already using) | 15 | 19 around + 12 in-ear | Both | ~2 GB | Free (Zenodo) | Already in use |
| 3 | **Mobile BCI Scalp+Ear** | 24 | 14 (around-ear) | Around-ear | ~5 GB (est.) | Free (Figshare) | **HIGH** |
| 4 | **EESM23** (Ear-EEG Sleep) | 10 | In-ear | In-ear | ~5 GB (est.) | Free (OpenNeuro) | MEDIUM |
| 5 | **EESM19** (Ear-EEG Sleep) | 20 | In-ear | In-ear | ~10 GB (est.) | Free (OpenNeuro) | MEDIUM |
| 6 | **SparrKULee** | 85 | 0 (64-ch scalp only) | Scalp only | >100 GB | Email request (KU Leuven) | LOW (no ear) |
| 7 | **ICASSP 2024 Auditory EEG** | 105 | 0 (64-ch scalp only) | Scalp only | ~50 GB | Email for password (KU Leuven) | LOW (no ear) |
| 8 | **KU Leuven AAD** | 16 | 0 (64-ch scalp only) | Scalp only | ~2 GB | Free (Zenodo) | LOW (no ear) |
| 9 | **AV-GC-AAD KU Leuven** | 16 | 0 (64-ch scalp only) | Scalp only | ~2 GB | Free (Zenodo) | LOW (no ear) |

---

## 1. NJU cEEGrid Auditory Attention Decoding Dataset (HIGHEST PRIORITY)

**This is the 98-subject cEEGrid dataset we want.**

- **Paper**: "Auditory Attention Decoding from Ear-EEG Signals: A Dataset with Dynamic Attention Switching and Rigorous Cross-Validation" (arXiv: 2510.19174)
- **Authors**: Yuanming Zhang, Zeyan Song, Jing Lu, Fei Chen, Zhibin Lin (Nanjing University)
- **DOI**: 10.21227/7QPK-9J22
- **URL**: https://ieee-dataport.org/documents/16-channel-three-speaker-dynamic-switch-ceegrid-auditory-attention-decoding-dataset

### Dataset Details
- **98 participants**, 16-channel cEEGrid electrodes (around-ear)
- 3 concurrent speakers at different spatial locations
- Dynamic attention switching tasks
- 63 trials per participant, 30 seconds each = 31.5 minutes/subject
- Total: ~51.5 hours of cEEGrid data

### Access Requirements
- **IEEE DataPort subscription required** (NOT open access)
- Individual subscription: **$40/month**
- **Free for IEEE Society members** (just log in with IEEE account)
- Alternative: Check if university/institution has an institutional subscription

### How to Get It
```bash
# Option A: IEEE Society membership (free DataPort access)
# 1. Go to https://ieee-dataport.org/subscribe
# 2. Log in with IEEE account (Society members get free access)
# 3. Navigate to: https://ieee-dataport.org/documents/16-channel-three-speaker-dynamic-switch-ceegrid-auditory-attention-decoding-dataset
# 4. Download

# Option B: Pay $40/month subscription
# 1. Go to https://ieee-dataport.org/individual-subscriptions
# 2. Subscribe for 1 month
# 3. Download dataset
# 4. Cancel subscription

# Option C: Contact authors directly
# Yuanming Zhang - Nanjing University
# Paper: https://arxiv.org/abs/2510.19174
# Check paper PDF for author contact emails
```

### Why This is Valuable
- **Largest around-ear EEG dataset** (98 subjects vs our 15)
- cEEGrid electrodes are similar to the around-ear channels in Ear-SAAD
- Auditory attention paradigm (same domain as our work)
- Could massively improve cross-subject generalization

---

## 2. Ear-SAAD Dataset (Already Using)

- **Paper**: Geirnaert et al., "A Direct Comparison of Simultaneously Recorded Scalp, Around-Ear, and In-Ear EEG," Scientific Reports, 2025
- **DOI**: https://zenodo.org/records/16536441
- **Institution**: KU Leuven (ESAT/STADIUS) + Aarhus University (Center for Ear-EEG)

### Dataset Details
- 15 subjects, 60 minutes each
- **27 scalp channels** (our input)
- **19 around-ear channels** (cEEGrid-style)
- **12 in-ear channels** (our prediction target)
- Auditory attention decoding paradigm (2 speakers at +/-60 degrees)

### Access
```bash
# Already downloaded. Free on Zenodo:
wget https://zenodo.org/records/16536441/files/ear_saad_dataset.zip
```

---

## 3. Mobile BCI Scalp + Ear-EEG Dataset (HIGH PRIORITY)

- **Paper**: Lee et al., "Mobile BCI dataset of scalp- and ear-EEGs with ERP and SSVEP paradigms while standing, walking, and running," Scientific Data, 2021
- **DOI**: 10.6084/m9.figshare.13604078
- **URL**: https://figshare.com/articles/dataset/Mobile_BCI_dataset_of_scalp-_and_ear-EEG_with_ERP_and_SSVEP_paradigms_during_standing_walking_and_running/13604078
- **Also on OSF**: https://osf.io/r7s9b/

### Dataset Details
- **24 participants**
- **32-channel scalp EEG** + **14-channel around-ear EEG** (cEEGrid)
- 4 electrooculography channels + 9 inertial measurement unit channels
- Standing, slow walking, fast walking, running conditions
- ERP and SSVEP paradigms

### Access
```bash
# Free download from Figshare (no registration required):
# Visit: https://figshare.com/articles/dataset/13604078
# Or download via API:
curl -L "https://figshare.com/ndownloader/articles/13604078/versions/1" -o mobile_bci_ear_eeg.zip
```

### Why This is Valuable
- Has **simultaneous scalp + around-ear** recordings (like Ear-SAAD)
- 24 subjects adds to our training pool
- Different paradigm (ERP/SSVEP) provides diversity
- Free and immediately downloadable

---

## 4. EESM23 - Ear-EEG Sleep Monitoring 2023 (MEDIUM PRIORITY)

- **Paper**: "Ear-EEG sleep monitoring data sets," Scientific Data, 2025
- **OpenNeuro**: https://openneuro.org/datasets/ds005178
- **GitHub**: https://github.com/OpenNeuroDatasets/ds005178
- **Institution**: Aarhus University (Center for Ear-EEG) -- same group as Ear-SAAD

### Dataset Details
- **10 subjects**, 12 nights each = 120 total nights
- In-ear EEG electrodes (custom Ear-EEG system from Aarhus/Kidmose lab)
- First 2 nights include simultaneous PSG (scalp EEG + EOG + EMG)
- Remaining 10 nights: ear-EEG only
- BIDS format, EEGLAB .set files

### Access
```bash
# Free download from OpenNeuro (no registration):
# Method 1: OpenNeuro CLI
npm install -g @openneuro/cli
openneuro download --snapshot 1.0.0 ds005178 eesm23/

# Method 2: DataLad
pip install datalad
datalad install https://github.com/OpenNeuroDatasets/ds005178.git
cd ds005178
datalad get .

# Method 3: AWS S3
aws s3 sync --no-sign-request s3://openneuro.org/ds005178 eesm23/
```

### Why This is Valuable
- **Same lab** as Ear-SAAD (Kidmose group at Aarhus)
- In-ear EEG with some scalp reference
- Sleep paradigm = very different from auditory attention, adds diversity
- Free and immediately downloadable

---

## 5. EESM19 - Ear-EEG Sleep Monitoring 2019 (MEDIUM PRIORITY)

- **OpenNeuro**: https://openneuro.org/datasets/ds005185
- **Institution**: Aarhus University (Center for Ear-EEG)

### Dataset Details
- **20 subjects**
- Part A: 20 subjects x 4 nights with PSG + ear-EEG + actigraphy
- Part B: 10 subjects x 12 additional nights with ear-EEG only
- Total: ~200 nights, ~320 nights combined with EESM23
- BIDS format

### Access
```bash
# Free download from OpenNeuro:
openneuro download --snapshot 1.0.0 ds005185 eesm19/

# Or via DataLad:
datalad install https://github.com/OpenNeuroDatasets/ds005185.git
```

---

## 6. SparrKULee (LOW - scalp only, but same research group)

- **Paper**: "SparrKULee: A Speech-Evoked Auditory Response Repository from KU Leuven," MDPI Data, 2024
- **DOI**: 10.48804/K3VSND
- **URL**: https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND
- **Institution**: KU Leuven (ExpORL) -- same group as Ear-SAAD

### Dataset Details
- **85 subjects**, 64-channel scalp EEG (BioSemi ActiveTwo)
- 90-150 minutes per subject = 168 total hours
- Single-speaker natural speech stimuli
- **No around-ear or in-ear channels** -- scalp only

### Access
```bash
# Requires email request to sparrkulee@kuleuven.be
# State intended use; some files restricted for privacy
# Download: >100 GB zip file
```

### Why Potentially Useful
- Same research group as Ear-SAAD, similar paradigm
- 85 subjects of scalp EEG for pretraining temporal encoders
- But NO ear channels, so only useful for pretraining, not direct transfer

---

## 7. ICASSP 2024 Auditory EEG Challenge Dataset (LOW - scalp only)

- **URL**: https://exporl.github.io/auditory-eeg-challenge-2024/dataset/
- **Institution**: KU Leuven (ExpORL)

### Dataset Details
- **105 subjects**, 64-channel scalp EEG
- 85 subjects in training set (same as ICASSP 2023 train+test)
- ~200 hours total, single-speaker Flemish speech
- Preprocessed versions available (split, normalized)

### Access
```bash
# Email auditoryeegchallenge@kuleuven.be with:
# - Team member names
# - Email addresses
# - Affiliations
# Receive download password
```

---

## 8. KU Leuven AAD Dataset (LOW - scalp only)

- **Zenodo**: https://zenodo.org/records/4004271
- **Old version**: https://zenodo.org/records/3377911

### Dataset Details
- 16 subjects, 64-channel scalp EEG
- Auditory attention decoding paradigm
- No ear channels

### Access
```bash
# Free on Zenodo:
wget https://zenodo.org/records/4004271/files/auditory_attention_detection_kuleuven.zip
```

---

## 9. AV-GC-AAD KU Leuven (LOW - scalp only)

- **Zenodo**: https://zenodo.org/records/11058711
- **DOI**: 10.5281/zenodo.11058711

### Dataset Details
- 16 subjects, 64-channel scalp EEG + 4 EOG
- Audiovisual gaze-controlled AAD paradigm
- No ear channels

### Access
```bash
# Free on Zenodo:
wget https://zenodo.org/records/11058711/files/av_gc_aad_kuleuven.zip
```

---

## Immediate Action Plan

### Can Download Right Now (Free):
1. **Mobile BCI Scalp+Ear** (24 subjects, scalp+ear) -- Figshare
2. **EESM23** (10 subjects, in-ear) -- OpenNeuro
3. **EESM19** (20 subjects, in-ear) -- OpenNeuro

### Needs Registration/Email:
4. **SparrKULee** (85 subjects, scalp only) -- email sparrkulee@kuleuven.be
5. **ICASSP 2024 Challenge** (105 subjects, scalp only) -- email auditoryeegchallenge@kuleuven.be

### Needs Subscription/Payment:
6. **NJU cEEGrid AAD** (98 subjects, around-ear) -- IEEE DataPort $40/month or free with IEEE Society membership

### Recommended Priority Order:
1. Download Mobile BCI dataset NOW (free, has scalp+ear, 24 subjects)
2. Download EESM23 + EESM19 NOW (free, in-ear, 30 subjects total)
3. Get NJU cEEGrid dataset via IEEE membership or $40 subscription (98 subjects!)
4. Email for SparrKULee (scalp pretraining data, 85 subjects)
5. Email for ICASSP 2024 challenge data (scalp pretraining, 105 subjects)

### Total Around-Ear/In-Ear Subjects Available:
- Ear-SAAD: 15 (already have)
- NJU cEEGrid: 98 (needs IEEE DataPort)
- Mobile BCI: 24 (free now)
- EESM23: 10 (free now)
- EESM19: 20 (free now)
- **Total: 167 subjects with ear electrodes**
