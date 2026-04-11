# Large EEG Datasets for Pretraining — Research Plan

**Goal**: Download thousands of subjects of EEG data for pretraining a temporal encoder,
then fine-tune on our 15-subject Ear-SAAD dataset.

**Date**: 2026-04-10

---

## Executive Summary

| Dataset | Subjects | Hours | Size | Access | Priority |
|---------|----------|-------|------|--------|----------|
| TUH EEG Corpus (TUEG) | 10,874 | 21,000+ | 572 GB (uncompressed) | Registration required (24-48h) | HIGH |
| HBN-EEG (Healthy Brain Network) | 2,600+ | ~5,000+ | ~1 TB (11 releases) | Free, no registration (AWS S3) | **HIGHEST** |
| PhysioNet Motor Imagery | 109 | ~50 | ~3.4 GB | Free, instant | MEDIUM |
| PhysioNet Sleep-EDF | 197 | ~3,900 | ~8 GB | Free, instant | MEDIUM |
| MOABB (all datasets combined) | ~1,000+ | varies | varies | Free via Python API | MEDIUM |
| REVE pretrained weights | (25,000 pretrained) | N/A | ~500 MB | Free (HuggingFace) | **HIGHEST** |
| SingLEM pretrained weights | (9,200 pretrained) | N/A | ~300 MB | Free (GitHub) | HIGH |

**Recommended strategy**: Use REVE or SingLEM pretrained weights directly (they already
absorbed 25K+ subjects of EEG), plus download HBN-EEG for any custom pretraining.

---

## 1. REVE — Pretrained EEG Foundation Model (BEST OPTION)

**Paper**: "REVE: A Foundation Model for EEG — Adapting to Any Setup with Large-Scale
Pretraining on 25,000 Subjects" (arXiv: 2510.21585)

**Key facts**:
- Pretrained on 92 datasets, 25,000 subjects, 60,000+ hours, 19 TB raw data
- Sources: OpenNeuro + MOABB + TUH combined
- Uses masked autoencoding objective
- Novel 4D positional encoding handles arbitrary electrode arrangements
- State-of-the-art on multiple EEG benchmarks
- **Channel-agnostic** — handles any montage via electrode position encoding

**Why this is ideal for us**:
- Our model uses channel-agnostic tokenization — REVE's approach is compatible
- Already pretrained on massive diverse EEG data — no need to download 19 TB
- Fine-tuning on Ear-SAAD would be straightforward

**Download pretrained weights**:
```bash
# Install dependencies
pip install transformers torch

# Load pretrained model
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('brain-bzh/reve-base')
print(model)
"
```

**GitHub**: https://github.com/elouayas/reve_eeg
**HuggingFace**: https://huggingface.co/brain-bzh/reve-base
**Project page**: https://brain-bzh.github.io/reve/

**Fine-tuning approach**:
- Extract temporal encoder from REVE
- Add linear head for 12-channel in-ear prediction
- Fine-tune on Ear-SAAD with frozen/unfrozen encoder layers
- The 4D positional encoding should handle scalp-to-in-ear mapping

---

## 2. SingLEM — Single-Channel EEG Foundation Model

**Paper**: "SingLEM: Single-Channel Large EEG Model" (arXiv: 2509.17920)

**Key facts**:
- Pretrained on 71 public datasets, 9,200+ subjects, 357,000 single-channel hours
- Hybrid architecture: CNN (local features) + hierarchical transformer (temporal)
- Self-supervised pretraining
- **Single-channel design** — inherently hardware-agnostic

**Why relevant**:
- Single-channel approach aligns with channel-agnostic tokenization
- Could apply per-channel, then aggregate for multi-channel prediction
- Smaller model, easier to fine-tune

**Download**:
```bash
git clone https://github.com/ttlabtuat/SingLEM.git
cd SingLEM
# Pretrained models are in the repository
# Training data organized as pickle files with EEG trials + labels per subject
```

**GitHub**: https://github.com/ttlabtuat/SingLEM

---

## 3. HBN-EEG — Healthy Brain Network (Largest Free-Download Dataset)

**Paper**: "HBN-EEG: The FAIR implementation of the Healthy Brain Network (HBN)
electroencephalography dataset" (bioRxiv 2024.10.03.615261)

**Key facts**:
- 2,600+ participants (ages 5-21), 6 cognitive tasks
- 128-channel EEG (EGI HydroCel Geodesic Sensor Net)
- BIDS format, rich HED event annotations
- 11 dataset releases on OpenNeuro (ds005505 through ds005515+)
- Release 1 alone: 136 participants, 103 GB
- **No registration required** — direct download from AWS S3

**Download commands**:
```bash
# Method 1: AWS CLI (fastest, no registration)
pip install awscli

# Download Release 1 (136 subjects, ~103 GB)
aws s3 cp s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1 \
    /data/hbn_eeg/R1 --recursive --no-sign-request

# Download all releases (2,600+ subjects, ~1 TB total)
for i in $(seq 1 11); do
    aws s3 cp s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R${i} \
        /data/hbn_eeg/R${i} --recursive --no-sign-request
done

# Method 2: OpenNeuro CLI
npm install -g @openneuro/cli
openneuro download --snapshot 1.0.0 ds005505 hbn_eeg_R1/

# Method 3: DataLad
pip install datalad
datalad install https://github.com/OpenNeuroDatasets/ds005505.git
cd ds005505
datalad get .
```

**Mini datasets** (for quick testing, 20 subjects each at 100 Hz):
Available as Google Drive directories from the HBN-EEG project page.

**NEMAR**: https://nemar.org/dataexplorer/detail?dataset_id=ds005505
**OpenNeuro**: https://openneuro.org/datasets/ds005505

---

## 4. TUH EEG Corpus (TUEG) — Largest Clinical EEG Dataset

**Key facts**:
- 16,986 sessions from 10,874 unique subjects
- 21,000+ hours of clinical EEG
- 24-36 channels, 250 Hz sampling, 16-bit
- 572 GB uncompressed (330 GB compressed)
- **Requires registration** (email form to help@nedcdata.org, 24-48h turnaround)

**Registration process**:
1. Fill out access form at https://isip.piconepress.com/projects/tuh_eeg/
2. Email signed copy to help@nedcdata.org
3. Wait 24-48 hours for credentials
4. Transmit SSH keys for rsync access

**Download commands** (after registration):
```bash
# Using rsync (official method)
rsync -auxvL nedc_tuh_eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg/v2.0.1/ \
    /data/tuh_eeg/

# Specific subsets:
# TUAB (Abnormal): data/tuh_eeg/tuh_eeg_abnormal/v3.0.1
# TUAR (Artifact):  data/tuh_eeg/tuh_eeg_artifact/v3.0.1
# TUSZ (Seizure):   data/tuh_eeg/tuh_eeg_seizure/v2.0.5

# Using tueg-tools Python library
pip install tueg-tools
python -c "
from tueg_tools import Dataset
ds = Dataset(username='YOUR_USER', password='YOUR_PASS')
ds.download('data/tuh_eeg/tuh_eeg_abnormal/v3.0.1')
"

# Using Braindecode
pip install braindecode
python -c "
from braindecode.datasets import TUH
ds = TUH(path='/data/tuh_eeg/', recording_ids=range(100))
"
```

**URL**: https://isip.piconepress.com/projects/tuh_eeg/

---

## 5. PhysioNet Datasets (Small but Instant Access)

### 5a. EEG Motor Movement/Imagery (EEGMMIDB)

- 109 subjects, 64 channels, 160 Hz
- ~1500 one- and two-minute recordings
- ~3.4 GB total

```bash
# Download with wget
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/

# Or with PhysioNet CLI
pip install wfdb
python -c "
import wfdb
wfdb.dl_database('eegmmidb', dl_dir='/data/physionet/eegmmidb')
"
```

### 5b. Sleep-EDF Database Expanded

- 197 subjects, 2 EEG channels + EOG + EMG, 100 Hz
- Whole-night polysomnographic recordings (~20h each)
- ~8 GB total

```bash
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
```

---

## 6. MOABB — Mother of All BCI Benchmarks

**Key facts**:
- Python library aggregating many BCI datasets
- Programmatic download via `moabb` package
- Estimated ~1,000+ subjects across all datasets combined
- Key large datasets within MOABB:
  - PhysionetMI: 109 subjects
  - BNCI2014-001: 9 subjects
  - Cho2017: 52 subjects
  - Lee2019: 54 subjects
  - Various SSVEP datasets: 357+ subjects total

```bash
pip install moabb

# List all available datasets
python -c "
import moabb
from moabb.datasets import utils
datasets = utils.dataset_list
for d in sorted(datasets, key=lambda x: x.n_subjects, reverse=True):
    print(f'{d.__class__.__name__}: {d.n_subjects} subjects, {d.n_sessions} sessions')
"

# Download a specific dataset
python -c "
from moabb.datasets import PhysionetMI
ds = PhysionetMI()
ds.download()  # Downloads to ~/mne_data/
"
```

**Note**: MOABB is already included in REVE's pretraining corpus, so using REVE
pretrained weights covers this data.

---

## 7. In-Ear / Around-Ear EEG Datasets

### 7a. Ear-SAAD (already using)
- 15 subjects, 27 scalp + 19 around-ear + 12 in-ear channels
- Zenodo: https://zenodo.org/records/16536441

### 7b. cEEGrid Auditory Attention Dataset (NEW, 2025)
- **98 participants**, 16-channel cEEGrid electrodes
- Auditory attention decoding with dynamic attention switching
- Available on IEEE DataPort
- Paper: arXiv 2510.19174

```bash
# Check IEEE DataPort for download link
# URL: https://ieee-dataport.org/keywords/ear-eeg
# May require IEEE account (free)
```

### 7c. Mobile BCI Scalp + Ear-EEG Dataset
- 24 participants, 32-channel scalp + 14-channel ear-EEG
- ERP and SSVEP paradigms (standing, walking, running)
- Paper: Nature Scientific Data (2021)

---

## Recommended Action Plan

### Phase 1: Quick Win (Day 1) — Use Pretrained Models
```bash
# Clone REVE and load pretrained weights
git clone https://github.com/elouayas/reve_eeg.git
cd reve_eeg
pip install -e .
python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('brain-bzh/reve-base'); print(m)"

# Clone SingLEM
git clone https://github.com/ttlabtuat/SingLEM.git
```
**Expected outcome**: Pretrained temporal features from 25K subjects, ready for fine-tuning.

### Phase 2: Custom Pretraining Data (Day 1-2) — HBN-EEG
```bash
# Download HBN-EEG Release 1 (136 subjects, 103 GB)
aws s3 cp s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1 \
    /data/hbn_eeg/R1 --recursive --no-sign-request
```
**Expected outcome**: 136 subjects of high-quality EEG for custom pretraining experiments.

### Phase 3: Register for TUH (Day 1, receive Day 3)
```bash
# Fill out form at https://isip.piconepress.com/projects/tuh_eeg/
# Email to help@nedcdata.org
# Wait 24-48h for credentials
```
**Expected outcome**: Access to 10,874 subjects (largest single source).

### Phase 4: Scale Up (Week 1)
- Download remaining HBN-EEG releases (2,600+ subjects)
- Download TUH EEG once credentials arrive
- Download PhysioNet datasets for diversity
- Total: ~14,000+ unique subjects

### Phase 5: In-Ear Specific Data (Week 1-2)
- Download cEEGrid 98-subject dataset from IEEE DataPort
- Combine with Ear-SAAD for in-ear specific fine-tuning
- Total in-ear/around-ear data: 113 subjects (15 + 98)

---

## Storage Requirements

| Dataset | Size | Cumulative |
|---------|------|------------|
| REVE weights | ~0.5 GB | 0.5 GB |
| SingLEM weights | ~0.3 GB | 0.8 GB |
| HBN-EEG Release 1 | 103 GB | 104 GB |
| HBN-EEG All Releases | ~1 TB | 1.1 TB |
| TUH EEG (compressed) | 330 GB | 1.4 TB |
| PhysioNet MI + Sleep | ~12 GB | 1.4 TB |
| **Total (full plan)** | **~1.4 TB** | |
| **Minimal viable (Phase 1-2)** | **~104 GB** | |

---

## Key Decision: Pretrained Weights vs. Raw Data

**Option A: Use REVE/SingLEM pretrained weights** (recommended first)
- Pros: Instant, no download, already trained on 25K subjects
- Cons: Architecture may not match our needs, less control
- Approach: Extract temporal features, add prediction head, fine-tune on Ear-SAAD

**Option B: Download raw data and pretrain from scratch**
- Pros: Full control over architecture and pretraining objective
- Cons: Needs 1+ TB storage, days of GPU time for pretraining
- Approach: Download HBN-EEG + TUH, pretrain masked autoencoder, fine-tune

**Recommendation**: Start with Option A (REVE weights), benchmark against current best
(r=0.378). If pretrained features help, then consider Option B for further gains.
