# EEG Denoising and Artifact Removal Research

## Motivation

Our subject analysis shows that "hard" subjects (especially Subject 14, consistently ~0.27 r) likely suffer from poor electrode contact and higher artifact contamination. Our current preprocessing is minimal:

- **Narrowband pipeline** (benchmark): Data arrives pre-processed from Ear-SAAD `.mat` files (1-9 Hz bandpass, 20 Hz sampling). We only do NaN interpolation and z-score normalization.
- **Broadband pipeline**: 1-45 Hz Butterworth bandpass (4th order, zero-phase), downsample to 128 Hz, NaN interpolation, z-score per channel per trial.

Neither pipeline performs any artifact detection or removal. Better denoising could improve prediction for artifact-heavy subjects/channels.

---

## 1. Deep Learning Denoising Models

### 1.1 ART: Artifact Removal Transformer (2025)

- **Paper**: [ART: Artifact Removal Transformer](https://arxiv.org/abs/2409.07326) (NeuroImage, March 2025)
- **Code**: [github.com/CNElab-Plus/ArtifactRemovalTransformer](https://github.com/CNElab-Plus/ArtifactRemovalTransformer)
- **Method**: Transformer architecture capturing millisecond-scale transient EEG dynamics. End-to-end denoising that simultaneously handles multiple artifact types (eye, muscle, line noise) in multichannel EEG. Uses enhanced noisy-clean EEG pair generation via ICA decomposition for training data.
- **Performance**: Sets new benchmark in EEG signal processing, surpassing prior deep-learning methods.
- **Applicability to us**: HIGH potential for broadband pipeline. Could denoise raw BIDS data before our spatial filtering. Pretrained models available. Concern: trained on standard scalp montages, may not generalize to in-ear channels.

### 1.2 IC-U-Net: U-Net Denoising Autoencoder (2022)

- **Paper**: [IC-U-Net](https://arxiv.org/abs/2111.10026) (NeuroImage, 2022)
- **Method**: U-Net autoencoder trained on mixtures of brain and non-brain ICA components. Ensemble of loss functions to model complex EEG fluctuations. No parameter tuning or artifact type designation required. No channel number limitations.
- **Applicability to us**: MEDIUM. User-friendly, publicly available, works with any channel count. Could apply to both scalp and in-ear channels. But older model -- ART likely superior.

### 1.3 AnEEG (2024)

- **Paper**: [AnEEG: leveraging deep learning for effective artifact removal](https://www.nature.com/articles/s41598-024-75091-z) (Scientific Reports, Oct 2024)
- **Method**: Deep learning artifact elimination from EEG signals.
- **Applicability to us**: MEDIUM. Recent but less detail available on architecture.

### 1.4 AT-AT: Autoencoder-Targeted Adversarial Transformers (2025)

- **Paper**: [AT-AT](https://arxiv.org/abs/2502.05332) (Feb 2025)
- **Method**: Convolutional autoencoder with batch normalization combined with transformer adversarial training. Lightweight architecture.
- **Applicability to us**: MEDIUM. Novel hybrid approach but very new, less tested.

### 1.5 STFNet: Spatial-Temporal Fusion Network (2024)

- **Paper**: [Integrating spatial and temporal features](https://pubmed.ncbi.nlm.nih.gov/39250956/) (2024)
- **Method**: Stacked multi-dimension feature extractor capturing temporal dependencies and spatial relationships for artifact removal.
- **Applicability to us**: MEDIUM-HIGH. Spatial-temporal integration aligns well with our scalp-to-inear spatial filtering task.

### 1.6 Transformer-Based Autoencoder (2025)

- **Paper**: [Transformer-Based Autoencoder for Denoising EEG and ECoG](https://sciety.org/articles/activity/10.21203/rs.3.rs-6689702/v1)
- **Performance**: Achieves SNR of 15.32 dB for EEG (vs ICA baseline of 8.76 dB) -- nearly 2x improvement.
- **Applicability to us**: HIGH if claims hold up. Dramatic SNR improvement.

---

## 2. Artifact Subspace Reconstruction (ASR)

### 2.1 Overview

ASR is an automated, online, component-based method for removing transient/large-amplitude artifacts. It identifies artifact subspaces via PCA and reconstructs clean data by interpolating from nearby clean data.

- **Key reference**: [ASR: a candidate dream solution](https://pmc.ncbi.nlm.nih.gov/articles/PMC10710985/)
- **Python implementation**: [asrpy](https://github.com/DiGyt/asrpy) -- works directly with MNE-Python Raw objects

### 2.2 Installation and Usage

```python
pip install asrpy
# or: pip install git+https://github.com/DiGyt/asrpy.git

import asrpy
asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=20)
asr.fit(raw)  # calibrate on clean segment
raw = asr.transform(raw)  # remove artifacts
```

### 2.3 Optimal Parameters

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| **cutoff** (k) | 2.5-100 | 20-30 | Primary aggressiveness control. Lower = more aggressive. 2.5 is most aggressive safe value, 5 is conservative default |
| **win_len** | 0.2-2.0s | 0.5s default | For slow EEG (<4 Hz), use 3-6x longer than default |
| **blocksize** | varies | default | Processing block size |
| **max_dropout_fraction** | 0-1 | 0.1 | Max fraction of channels that can be interpolated |

**For our data**: Since we target 1-9 Hz (slow activity), a longer sliding window (1.5-3.0s) would be appropriate. Cutoff of 15-25 seems a good starting range. ASR has ~30 additional parameters, most rarely discussed.

### 2.4 Riemannian ASR

- **Paper**: [Riemannian Modification of ASR](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2019.00141/full)
- Uses Riemannian geometry for covariance estimation -- more robust for low-density EEG.
- Available in [python-meegkit](https://github.com/nbara/python-meegkit): `meegkit.asr`

### 2.5 Low-Density EEG Considerations

- **Paper**: [Optimizing ASR for Low-Density EEG](https://www.academia.edu/89509467/)
- Our setup (27 scalp + 12 in-ear = 39 channels) is moderate density. Standard ASR parameters should work but may benefit from parameter optimization.

---

## 3. ICA-Based Automated Denoising

### 3.1 ICLabel

- **Paper**: [ICLabel](https://pmc.ncbi.nlm.nih.gov/articles/PMC6592775/) (2019)
- **Code**: [github.com/sccn/ICLabel](https://github.com/sccn/ICLabel)
- Classifier trained on 6,000+ datasets. Classifies ICs into: Brain, Eye, Muscle, Heart, Line Noise, Channel Noise, Other.
- Standard in EEGLAB pipelines. Python version available via MNE-Python + mne-icalabel.

### 3.2 MNE-Python ICA Pipeline

```python
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# Fit ICA
ica = ICA(n_components=20, method='infomax', random_state=42)
ica.fit(raw)

# Auto-label components
labels = label_components(raw, ica, method='iclabel')

# Remove non-brain components
exclude = [i for i, label in enumerate(labels['labels'])
           if label not in ('brain', 'other')]
ica.exclude = exclude
raw_clean = ica.apply(raw.copy())
```

### 3.3 AMICA (Adaptive Mixture ICA)

- More robust than standard ICA for mobile/noisy data
- Recent 2024 study optimized decomposition quality using sample rejection criteria
- Better for high-artifact data but computationally expensive

### 3.4 GEDAI (2025)

- **Paper**: [Return of the GEDAI](https://www.biorxiv.org/content/10.1101/2025.10.04.680449v1.full) (bioRxiv, 2025)
- Unsupervised EEG denoising based on leadfield filtering
- Outperformed ASR, MARA, and ICLabel in brain fingerprinting accuracy

---

## 4. Critical Finding: "EEG is better left alone"

### 4.1 The Paper

- **Paper**: [EEG is better left alone](https://www.nature.com/articles/s41598-023-27528-0) (Scientific Reports, 2023)
- **Key argument**: For relatively clean lab EEG, most automated preprocessing techniques have no positive effect or are actively detrimental. Only high-pass filtering and bad channel interpolation reliably helped.
- ICA artifact rejection of eyes/muscles failed to increase performance reliably.
- Rejecting bad segments could not compensate for loss of statistical power.

### 4.2 The Counterpoint for Decoding/Prediction Tasks

- **Paper**: [How EEG preprocessing shapes decoding performance](https://www.nature.com/articles/s42003-025-08464-3) (Communications Biology, 2025)
- **Code**: [github.com/kesslerr/m4d](https://github.com/kesslerr/m4d)
- **Critical finding**: ALL artifact correction steps REDUCED decoding performance across experiments and models.
- Higher high-pass filter cutoffs consistently INCREASED decoding performance.
- Reason: Artifact correction may remove artifacts that covary with experimental conditions, AND may remove portions of the neural signal of interest.

### 4.3 Implications for Our Task

This is the most important finding for our project:

> **For prediction/decoding tasks, aggressive denoising is likely counterproductive.**

Our task is **prediction** (scalp -> in-ear), not source localization or ERP analysis. The in-ear signal contains both neural and non-neural components. If artifacts in scalp and in-ear channels are correlated (e.g., motion artifacts, muscle tension), removing them from the scalp input would remove information that helps predict the in-ear signal.

**However**, there are scenarios where denoising COULD help:
1. **Uncorrelated noise** in in-ear channels (poor contact, electrode drift) -- denoising the target could improve training signal quality
2. **Subject-specific artifact patterns** that confuse cross-subject models
3. **Channel-specific denoising** for known bad channels (ELC, ERT artifacts)

---

## 5. Practical Recommendations for Our Pipeline

### 5.1 What We Should Try (Ordered by Expected Impact)

#### Tier 1: Low-risk, easy to implement

1. **ASR on broadband data before spatial filtering**
   - Use asrpy with cutoff=20, win_len=1.5s
   - Apply to raw BIDS data before bandpass filtering
   - Quick to implement, well-validated, parameter sweep possible
   - Expected: modest improvement on hard subjects, minimal harm to good subjects

2. **Higher high-pass filter cutoff** (from 1 Hz to 2-3 Hz)
   - The decoding performance paper found this consistently helps
   - Very easy to test as a preprocessing parameter
   - Risk: may lose slow neural dynamics relevant to in-ear prediction

3. **Robust channel-level outlier detection**
   - Flag windows where individual channels exceed 4-5 sigma
   - Interpolate or exclude those windows during training
   - Already partially done with NaN handling but could be more aggressive

#### Tier 2: Moderate effort, uncertain benefit

4. **ICA + ICLabel on broadband data**
   - Remove eye/muscle components before spatial filtering
   - Risk: may remove correlated artifacts that help prediction (per "better left alone" findings)
   - Implement via mne-icalabel

5. **Target-side denoising** (denoise in-ear channels only)
   - Hypothesis: noisy targets hurt training more than noisy inputs
   - Apply ASR or wavelet denoising to in-ear channels only
   - Keep scalp channels untouched to preserve predictive information

#### Tier 3: High effort, high potential

6. **ART (Artifact Removal Transformer)**
   - Apply pretrained ART model as preprocessing step
   - Best for broadband pipeline where more artifact types are present
   - Risk: pretrained on standard montages, may not generalize to ear-EEG
   - Would need to test on scalp channels only (in-ear montage too different)

7. **Subject-adaptive denoising**
   - Fit per-subject ASR/ICA during a calibration segment
   - Then apply to that subject's data
   - Could help the hard subjects specifically

### 5.2 What We Should NOT Try

- **Pure correlation loss with denoising** -- already shown to produce degenerate scale
- **InstanceNorm as denoising proxy** -- consistently hurts by -0.003 r
- **Aggressive ICA removal on narrowband 1-9 Hz data** -- too few components to separate reliably at 20 Hz sampling rate
- **Deep learning denoising on the narrowband data** -- not enough bandwidth for these models to work with

### 5.3 Implementation Priority

For maximum impact with minimum risk:

```
Step 1: Test higher HP filter (2 Hz, 3 Hz) on broadband -- 1 hour
Step 2: Add ASR to broadband preprocessing -- 2 hours  
Step 3: Test target-side denoising (in-ear only ASR) -- 2 hours
Step 4: ICA + ICLabel if ASR shows promise -- 3 hours
Step 5: ART if broadband + ASR beats current best -- 4 hours
```

---

## 6. Key Tools and Libraries

| Tool | Install | Purpose |
|------|---------|---------|
| [asrpy](https://pypi.org/project/asrpy/) | `pip install asrpy` | Artifact Subspace Reconstruction for MNE |
| [python-meegkit](https://github.com/nbara/python-meegkit) | `pip install meegkit` | Riemannian ASR + other denoising |
| [mne-icalabel](https://mne.tools/mne-icalabel/) | `pip install mne-icalabel` | Automated ICA component labeling |
| [ART](https://github.com/CNElab-Plus/ArtifactRemovalTransformer) | clone + pip | Transformer-based artifact removal |
| [IC-U-Net](https://arxiv.org/abs/2111.10026) | clone | U-Net denoising autoencoder |
| [DeepSeparator](https://github.com/ncclabsustech/DeepSeparator) | clone | Deep learning artifact separation |

---

## 7. Summary

The literature presents a **paradox for prediction tasks**: aggressive denoising improves signal quality metrics but can reduce prediction/decoding performance by removing correlated artifacts or neural signal components. Our best strategy is:

1. **Start conservative**: Higher HP filter + mild ASR (cutoff=20-30)
2. **Denoise targets more than inputs**: Clean in-ear channels to improve training labels
3. **Per-subject adaptation**: Fit denoising parameters per subject to help hard cases
4. **Validate empirically**: The only way to know is to benchmark each approach

The 0.378 r plateau may be partly due to noise in the in-ear targets (poor electrode contact on hard subjects), not just limitations of our model architecture. Improving target signal quality could unlock gains that no amount of model complexity will achieve.

---

## Sources

- [ART: Artifact Removal Transformer](https://arxiv.org/abs/2409.07326)
- [AnEEG: deep learning artifact removal](https://www.nature.com/articles/s41598-024-75091-z)
- [IC-U-Net](https://arxiv.org/abs/2111.10026)
- [AT-AT: Adversarial Transformers](https://arxiv.org/abs/2502.05332)
- [ASR review paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10710985/)
- [asrpy Python library](https://github.com/DiGyt/asrpy)
- [Riemannian ASR](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2019.00141/full)
- [ICLabel](https://pmc.ncbi.nlm.nih.gov/articles/PMC6592775/)
- [EEG is better left alone](https://www.nature.com/articles/s41598-023-27528-0)
- [How EEG preprocessing shapes decoding performance](https://www.nature.com/articles/s42003-025-08464-3)
- [GEDAI: Unsupervised EEG Denoising](https://www.biorxiv.org/content/10.1101/2025.10.04.680449v1.full)
- [Comprehensive DL denoising review](https://link.springer.com/article/10.1007/s42452-025-07808-2)
- [Novel attention-based artifact removal](https://www.nature.com/articles/s41598-025-98653-1)
- [DeepSeparator](https://github.com/ncclabsustech/DeepSeparator)
- [Optimizing ASR for low-density EEG](https://www.academia.edu/89509467/)
- [A2DM: Artifact-Aware Denoising](https://link.springer.com/article/10.1007/s12559-025-10442-0)
