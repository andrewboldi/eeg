# cEEGrid 98-Subject Dataset Access Research

## Target Dataset
- **Title**: 16-channel Three-speaker dynamic-switch cEEGrid Auditory Attention Decoding Dataset
- **DOI**: 10.21227/7QPK-9J22
- **Platform**: IEEE DataPort
- **Key stats**: 98 subjects, 16-channel cEEGrid, 3 concurrent speakers, dynamic attention switching
- **Authors**: Yuanming Zhang, Zeyan Song, Jing Lu, Zhibin Lin (Nanjing University, Key Lab of Modern Acoustics / NJU-Horizon Intelligent Audio Lab)
- **Paper**: arXiv:2510.19174 "Auditory Attention Decoding from Ear-EEG Signals: A Dataset with Dynamic Attention Switching and Rigorous Cross-Validation"

## 1. Free Access Pathways to IEEE DataPort

### IEEE DataPort Subscription Model
- **Open Access datasets**: Free to all registered users (just create a free IEEE account)
- **Standard datasets**: Require a paid subscription
- **Individual subscription**: ~$40/month ($480/year value)
- **The 7QPK-9J22 dataset appears to be a Standard (not Open Access) dataset**, meaning it requires a subscription

### Free Access via IEEE Society Membership
- **All IEEE Society members get a free IEEE DataPort subscription** automatically
- This includes IEEE Signal Processing Society, IEEE Computer Society, IEEE Circuits and Systems Society, and others
- Just log in to IEEE DataPort with society member credentials -- subscription activates automatically
- IEEE Society membership costs vary (~$30-50/year for students), but the DataPort subscription ($480 value) is included free
- **This is the cheapest legitimate access path**: ~$30-50 for student IEEE society membership gives full DataPort access

### Student Discount
- No specific "student discount" on IEEE DataPort subscriptions themselves
- However, IEEE student membership + any society membership is heavily discounted (~$30-50 total)
- Student society membership automatically includes free DataPort access
- **Recommendation**: Join IEEE as a student member + Signal Processing Society = cheapest path to full DataPort access

### Institutional Access
- Many universities have institutional IEEE DataPort subscriptions
- Check your university library / institutional access page first
- Some universities provide trial access

### Coupons
- IEEE DataPort FAQ page mentions promotional coupons exist occasionally
- No currently active public coupons found as of this research

## 2. Same Data on Another Platform?

### NJU Auditory Attention Decoding Dataset (Earlier Version)
- **DOI**: 10.21227/31nb-0j75
- **Also on IEEE DataPort**: https://ieee-dataport.org/documents/nju-auditory-attention-decoding-dataset
- Published August 2023 by the same group (Yuanming Zhang, Ziyan Yuan, Jing Lu)
- This appears to be an earlier/different version of the NJU cEEGrid data
- Same access restrictions apply (IEEE DataPort subscription required)

### Other Platforms Checked
- **Zenodo**: NOT found. The NJU cEEGrid dataset is not mirrored on Zenodo
- **Figshare**: NOT found
- **OpenNeuro**: NOT found
- **OSF**: NOT found
- **GitHub**: No data repository found (only code references)
- **SSRN**: Paper preprint available (abstract_id=5800170) but no data download

### Conclusion
The NJU cEEGrid dataset appears to be **exclusively hosted on IEEE DataPort**. The authors did not mirror it on any open-access repository. This is unfortunately common for IEEE DataPort datasets -- the platform's business model incentivizes exclusive hosting.

## 3. Search Results: "NJU cEEGrid auditory attention" on Other Platforms

| Platform | Result |
|----------|--------|
| Zenodo | Not found. Related: KULeuven AAD dataset (zenodo.org/records/4004271), Auditory EEG dataset (zenodo.org/records/1199011) |
| Figshare | Not found |
| OpenNeuro | Not found. Related: EESM23 ear-EEG sleep (ds005178), Surrey cEEGrid sleep (ds005207) |
| OSF | Not found. Related: Mobile BCI scalp+ear dataset (osf.io/r7s9b) |
| GitHub | Not found as data. openlists/ElectrophysiologyData list does not include it |

## 4. Authors and Their Other Publications/Data

### Research Group
- **Affiliation**: Key Lab of Modern Acoustics, Nanjing University (NJU) + NJU-Horizon Intelligent Audio Lab (Horizon Robotics)
- **Key researchers**: Yuanming Zhang, Zeyan Song, Jing Lu, Zhibin Lin

### Known Datasets from This Group
1. **NJU Auditory Attention Decoding Dataset** (2023) -- IEEE DataPort, DOI: 10.21227/31nb-0j75
2. **16-channel Three-speaker cEEGrid Dataset** (the target) -- IEEE DataPort, DOI: 10.21227/7QPK-9J22

### Related Papers
- arXiv:2510.19174 -- "Auditory Attention Decoding from Ear-EEG Signals" (Oct 2025)
- arXiv:2409.08710 -- "Using Ear-EEG to Decode Auditory Attention in Multiple-speaker Environment" (Sep 2024)

### Contact
- Yuanming Zhang: yuanming.zhang@smail.nju.edu.cn
- Could try emailing authors directly to request data sharing outside IEEE DataPort

## 5. Alternative Around-Ear EEG Datasets

### Datasets with Simultaneous Scalp + Around-Ear/In-Ear EEG (Most Relevant)

| Dataset | Subjects | Channels | Task | Access | Link |
|---------|----------|----------|------|--------|------|
| **Ear-SAAD** (Geirnaert 2025) | 15 | 27 scalp + 19 around-ear + 12 in-ear | Auditory attention | **Open** (Zenodo) | zenodo.org/records/16536441 |
| **Mobile BCI** (Lee 2021) | 24 | 32 scalp + 14 ear-EEG | ERP + SSVEP | **Open** (OSF) | osf.io/r7s9b |
| **NJU cEEGrid 98-subj** (Zhang 2025) | 98 | 16 cEEGrid | 3-speaker AAD | **Paywalled** (IEEE DataPort) | 10.21227/7QPK-9J22 |
| **NJU AAD** (Zhang 2023) | ~106 | 16 cEEGrid | AAD | **Paywalled** (IEEE DataPort) | 10.21227/31nb-0j75 |

### Around-Ear / Ear-EEG Only Datasets (Open Access)

| Dataset | Subjects | Channels | Task | Access | Link |
|---------|----------|----------|------|--------|------|
| **EESM23** (Aarhus 2025) | 10 | Ear-EEG | Sleep monitoring | **Open** (OpenNeuro) | openneuro.org/datasets/ds005178 |
| **EESM19** (Aarhus 2025) | Multiple | Ear-EEG + PSG | Sleep monitoring | **Open** (OpenNeuro) | openneuro.org/datasets/ds005185 |
| **Surrey cEEGrid Sleep** (2017) | 20 | cEEGrid + PSG | Sleep monitoring | **Open** (OpenNeuro) | ds005207 |
| **SeizeIT2** (2025) | 125 patients | Behind-ear EEG | Epilepsy | **Open** (Sci Data) | nature.com/articles/s41597-025-05580-x |

### Other Relevant Open Auditory Attention Datasets (Scalp-Only)

| Dataset | Subjects | Channels | Access | Link |
|---------|----------|----------|--------|------|
| **KULeuven AAD** | 16 | 64 scalp | **Open** (Zenodo) | zenodo.org/records/4004271 |
| **DTU AAD** | 18 | 64 scalp | **Open** (Zenodo) | zenodo.org/records/1199011 |
| **ICASSP 2024 Challenge** | 85 | 64 scalp | **Open** | exporl.github.io/auditory-eeg-challenge-2024 |

## Summary and Recommendations

### For Accessing the NJU 98-Subject cEEGrid Dataset
1. **Cheapest path**: IEEE student membership + any society (e.g., Signal Processing Society) = ~$30-50/year, includes free DataPort access
2. **Check institutional access**: Your university may already subscribe to IEEE DataPort
3. **Email authors**: Contact yuanming.zhang@smail.nju.edu.cn to request direct data sharing
4. **Not available elsewhere**: The dataset is exclusively on IEEE DataPort; no mirrors found on Zenodo, Figshare, OpenNeuro, or OSF

### For Our Project (Scalp-to-Ear Prediction)
- The NJU dataset has **cEEGrid only (no simultaneous scalp EEG)**, so it would not be directly useful for scalp-to-in-ear prediction unless we develop a transfer learning approach
- **Ear-SAAD remains the best dataset** for our task (simultaneous scalp + around-ear + in-ear)
- **Mobile BCI** (24 subjects, scalp + ear, open access) could be useful for pretraining or transfer learning
- **SeizeIT2** (125 patients, behind-ear, open) is large but epilepsy-focused

### Key Insight
The NJU cEEGrid dataset records **ear-EEG only** (16 cEEGrid channels, no scalp EEG). For our scalp-to-in-ear prediction task, this dataset would only be useful for:
- Pretraining an ear-EEG decoder
- Learning ear-EEG signal characteristics
- Transfer learning approaches

It would NOT provide the simultaneous scalp+ear pairs we need for direct prediction training.
