# Multi-Task Self-Supervised Pretraining for EEG: Literature Review

## Summary

This document surveys recent (2024-2025) multi-task self-supervised pretraining objectives
for EEG and biosignals that go beyond simple masked autoencoding. The goal is to identify
auxiliary objectives that could improve our scalp-to-in-ear EEG prediction task.

---

## 1. Key Papers and Approaches

### 1.1 MTSSRL-MD (ICLR 2025 Workshop)
**Multi-Task Self-Supervised Representation Learning across Multiple Datasets**

Three complementary pretext tasks on heterogeneous EEG datasets:
- **Augmentation contrastive learning**: Standard contrastive loss on augmented views
- **Temporal shuffling discrimination**: Binary classification -- are subsequences in natural temporal order or shuffled?
- **Frequency band masking**: Zero out one of six canonical EEG bands (delta-gamma) in Fourier domain; model must classify which band is missing (cross-entropy loss)

A Channel Alignment Module (CAM) projects heterogeneous montages into shared space.
Outperforms single-task SSRL baselines, especially under low-label conditions on sleep staging.

Source: https://openreview.net/forum?id=qD4N15aL2W

### 1.2 REVE (NeurIPS 2025)
**Foundation Model for EEG with 25,000 Subjects**

Primary objective: Masked autoencoding with spatio-temporal block masking on raw EEG.
Secondary objective: **Global token loss** -- a single [CLS]-like token trained via attention
pooling to reconstruct the full input. This forces the encoder to build a compact global
representation at every depth layer.

Key finding: The global token loss is *particularly effective in frozen-feature scenarios*,
improving zero-shot and few-shot performance. It encourages more balanced use of encoder depth.

Pretrained on 60,000+ hours, 92 datasets. Uses 4D Fourier positional encoding for arbitrary
electrode layouts.

Source: https://arxiv.org/abs/2510.21585

### 1.3 EEGPT (NeurIPS 2024)
**Dual Self-Supervised Learning for Universal EEG**

Two objectives:
- **Mask-based reconstruction**: Standard masked autoencoding of EEG patches
- **Spatio-temporal representation alignment**: Aligns time and spatial views of the same
  signal in embedding space (contrastive-like). Emphasizes high-SNR features over raw input.

Trained on large mixed multi-task EEG corpus. Achieves SOTA on multiple downstream tasks
with linear probing.

Source: https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf

### 1.4 LaBraM (ICLR 2024 Spotlight)
**Large Brain Model with Neural Tokenizer**

Two-stage pretraining:
1. **Neural tokenizer** via vector-quantized neural spectrum prediction: Encodes raw EEG
   patches into discrete codes by predicting the Fourier spectrum (amplitude + phase).
2. **Masked neural code prediction**: Transformer predicts original neural codes for masked
   patches (like BEiT for vision, but with EEG-specific tokenizer).

Symmetric masking strategy improves training efficiency. Three model sizes (5.8M to 369M
params). Trained on ~2,500 hours across 20 datasets.

Source: https://arxiv.org/abs/2405.18765

### 1.5 bioFAME (ICLR 2024, Apple)
**Frequency-Aware Masked Autoencoders for Biosignals**

Key innovations:
- **Frequency-aware transformer**: Uses fixed-size Fourier-based operator for global token
  mixing, independent of input length and sampling rate
- **Frequency-maintain pretraining**: Performs masked autoencoding in latent space while
  preserving frequency structure
- Explicitly designed for robustness to distributional shifts between pretraining and
  inference (different tasks, different modality compositions)

Outperforms prior SOTA by 5.5% average on diverse transfer experiments.

Source: https://arxiv.org/abs/2309.05927 | Code: https://github.com/apple/ml-famae

### 1.6 TF-C (NeurIPS 2022, Harvard)
**Time-Frequency Consistency for Contrastive Pretraining**

Four components: time encoder, frequency encoder, two cross-space projectors.
Core idea: Time-based and frequency-based representations of the same sample should be
closer than representations of different samples in a joint time-frequency space.

Three loss components:
- Time-domain contrastive loss (augmentation invariance)
- Frequency-domain contrastive loss (spectrum augmentation invariance)
- **Cross-space consistency loss** (time and frequency embeddings should agree)

Evaluated on EEG, EMG, ECG, accelerometer. 15.4% average F1 improvement on transfer tasks.

Source: https://arxiv.org/abs/2206.08496 | Code: https://github.com/mims-harvard/TFC-pretraining

### 1.7 EEG2Rep (KDD 2024)
**Self-Prediction with Informative Masking**

Key innovations:
- Predicts masked input in **latent representation space** (not raw signal)
- **Semantic Subsequence Preserving (SSP)**: Instead of random masking, preserves
  semantically meaningful subsequences as context, providing more informative inputs
- Addresses EEG-specific challenges: low SNR, amplitude variability, no explicit segmentation

50% preservation ratio optimal across 6 diverse EEG tasks. Significantly outperforms
standard masking approaches.

Source: https://arxiv.org/abs/2402.17772

### 1.8 MTSL-TimesNet (Knowledge-Based Systems, 2025)
**Multi-Task Self-Supervised Learning for EEG Emotion**

Two auxiliary tasks sharing a feature extractor:
- **Spatial jigsaw**: Shuffle brain-region electrode patches; model reconstructs correct
  spatial arrangement (captures spatial topology)
- **Time-frequency enhanced contrastive learning**: Generates instance-level hard negatives
  from time-frequency augmentations

Achieves 96.3% on DEAP emotion classification. Demonstrates that spatial structure prediction
is a valuable pretext task.

Source: https://www.sciencedirect.com/science/article/abs/pii/S0950705125014030

---

## 2. Taxonomy of Auxiliary Objectives

| Objective | Type | What It Learns | Used By |
|-----------|------|---------------|---------|
| Masked reconstruction (raw) | Generative | Local temporal patterns | REVE, EEGPT, AFTA |
| Masked reconstruction (latent) | Generative | Semantic representations | EEG2Rep, LaBraM |
| Frequency spectrum prediction | Generative | Spectral structure | LaBraM (tokenizer) |
| Global token reconstruction | Generative | Compact global summary | REVE |
| Temporal order prediction | Discriminative | Temporal dynamics | MTSSRL-MD |
| Frequency band identification | Discriminative | Spectral band signatures | MTSSRL-MD |
| Spatial jigsaw | Discriminative | Spatial topology | MTSL-TimesNet |
| Contrastive (augmentation) | Contrastive | Invariant features | MTSSRL-MD, BIOT |
| Contrastive (time-frequency) | Contrastive | Cross-domain consistency | TF-C |
| Spatio-temporal alignment | Contrastive | Space-time coherence | EEGPT |

---

## 3. Relevance to Scalp-to-In-Ear Prediction

### 3.1 Most Promising Auxiliary Tasks

**Frequency band prediction (MTSSRL-MD style)**
- Zero out a frequency band in input; model predicts which band is missing
- For our 1-9 Hz data: delta (1-4 Hz) vs theta (4-8 Hz) vs low-alpha (8-9 Hz)
- Forces encoder to learn discriminative spectral features
- Very cheap to implement as additional classification head
- Risk: Only 3 distinguishable bands in our narrow range

**Global token loss (REVE style)**
- Add a CLS token that must reconstruct the full window from attention pooling
- Encourages encoder to build useful global representations
- Particularly effective for frozen-feature / few-shot transfer (relevant for cross-subject)
- Low implementation overhead: just one extra token + reconstruction head

**Temporal order prediction (MTSSRL-MD style)**
- Split 2s window into sub-segments; predict if order is correct
- Forces model to learn temporal dynamics beyond static spatial mapping
- May help capture the temporal structure our FIR filter already exploits
- Risk: At 20 Hz, 2s windows are only 40 samples -- limited temporal structure to shuffle

**Time-frequency consistency (TF-C style)**
- Enforce that time-domain and frequency-domain encoders agree
- Could regularize the spatial filter to respect spectral structure
- Higher implementation complexity (dual encoder architecture)

### 3.2 Less Promising for Our Task

**Spatial jigsaw**: Our 27 scalp channels have fixed positions and our model is a linear
spatial filter -- shuffling electrode positions would directly conflict with the spatial
mapping we need to learn.

**Contrastive augmentation**: Our dataset is small (12 training subjects) and the mapping
is approximately linear. Contrastive objectives may not provide sufficient signal for
learning a better linear spatial filter.

**Latent-space reconstruction (EEG2Rep)**: Requires a pretrained tokenizer or autoencoder
first, adding significant complexity for marginal gain on a near-linear task.

### 3.3 Practical Recommendation for Our Pipeline

Given our constraints (small dataset, near-linear mapping, 1-9 Hz narrowband, 20 Hz
sampling rate), the most practical multi-task additions are:

1. **Global token loss** (REVE): Add a [CLS] token to our FIR model that must reconstruct
   the average scalp EEG from its internal representation. Minimal overhead, proven to help
   cross-subject transfer.

2. **Frequency band prediction**: Randomly mask delta/theta/alpha bands in input; add a
   3-way classification head. Forces spectral awareness.

3. **Temporal order binary classification**: Split window in half; randomly swap order 50%
   of the time; add binary classification head. Forces temporal awareness.

These can be added as weighted auxiliary losses alongside the primary MSE + correlation
loss, similar to:

```
L_total = L_mse + lambda_corr * L_corr + lambda_global * L_global_recon + lambda_freq * L_freq_cls + lambda_order * L_order_cls
```

Start with lambda values of 0.1 for auxiliary tasks and tune.

---

## 4. References

1. MTSSRL-MD: https://openreview.net/forum?id=qD4N15aL2W
2. REVE: https://arxiv.org/abs/2510.21585
3. EEGPT: https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf
4. LaBraM: https://arxiv.org/abs/2405.18765
5. bioFAME: https://arxiv.org/abs/2309.05927
6. TF-C: https://arxiv.org/abs/2206.08496
7. EEG2Rep: https://arxiv.org/abs/2402.17772
8. MTSL-TimesNet: https://www.sciencedirect.com/science/article/abs/pii/S0950705125014030
9. Self-Supervised EEG Survey (ACM 2025): https://dl.acm.org/doi/10.1145/3736574
10. EEG Foundation Models Review: https://arxiv.org/html/2507.11783v2
