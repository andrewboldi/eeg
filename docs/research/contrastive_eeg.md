# Contrastive Self-Supervised Learning for EEG: Subject-Invariant Representations

Research survey conducted 2026-04-11. Focus: contrastive pretraining methods that learn
subject-invariant EEG features, with applicability to our scalp-to-in-ear prediction task.

---

## 1. Key Papers and Methods

### 1.1 BENDR (2021) -- Foundation Work
- **Paper**: [arXiv 2101.12037](https://arxiv.org/abs/2101.12037)
- **Method**: Adapts wav2vec 2.0 to EEG. Two-stage architecture: (1) convolutional encoder
  compresses raw EEG into latent vectors, (2) transformer contextualizes with contrastive loss
  over masked segments (10 contiguous vectors masked at a time).
- **Pretraining data**: Temple University Hospital EEG corpus (~1500 hours).
- **Key result**: Single pretrained model transfers across different hardware, subjects, and tasks.
  Fine-tuning beats task-specific self-supervision on sleep staging.
- **Relevance to us**: Demonstrated that contrastive pretraining on large EEG corpora produces
  transferable representations. Our 168-subject pretraining pool is much smaller but domain-matched.

### 1.2 CL-SSTER (2024) -- Shared Spatiotemporal Representations
- **Paper**: [NeuroImage 2024](https://www.sciencedirect.com/science/article/pii/S1053811924003872), [arXiv 2402.14213](https://arxiv.org/abs/2402.14213)
- **Method**: Contrastive Learning of Shared SpatioTemporal EEG Representations. Uses spatial
  and temporal convolutions jointly. Positive pairs = same stimulus across different subjects;
  negative pairs = different stimuli.
- **Key insight**: Maximizing inter-subject similarity for identical stimuli forces the network to
  learn subject-invariant spatial and temporal filters. Achieved highest inter-subject correlation
  (ISC) values vs. all baselines.
- **Relevance to us**: Directly applicable paradigm. We can define positive pairs as same-timepoint
  scalp EEG across subjects (since all subjects hear the same audiobook), forcing the encoder to
  learn subject-invariant spatial projections.

### 1.3 CLISA (2022) -- Inter-Subject Alignment
- **Paper**: [IEEE TAFFC 2022](https://ieeexplore.ieee.org/document/9748967/), [arXiv 2109.09559](https://arxiv.org/abs/2109.09559)
- **Method**: Contrastive Learning for Inter-Subject Alignment. Minimizes inter-subject differences
  by maximizing similarity of EEG representations across subjects receiving the same emotional
  stimuli. Uses depthwise spatial convolution + temporal convolution on raw EEG.
- **Key insight**: Sensor dropout (randomly zeroing entire channels) is an effective spatial
  augmentation that improves cross-subject transfer. This aligns with our finding that 15% channel
  dropout helps (iter011).

### 1.4 CLUDA (2024) -- Multi-Source Domain Adaptation
- **Paper**: [Expert Systems with Applications, Oct 2024](https://www.sciencedirect.com/science/article/abs/pii/S0957417424023194)
- **Method**: Contrastive Learning-based Unsupervised multi-source Domain Adaptation. Treats each
  source subject as a separate domain. Pulls same-class features together, pushes different-class
  features apart across all domain pairs.
- **Architecture**: Spatial-temporal convolutional module + self-attention module as feature
  extractor. Aligns conditional distributions per source-target pair.
- **Relevance to us**: Each of our 12 training subjects is a separate "domain." CLUDA's per-pair
  alignment could reduce subject-specific bias before the spatial filter learns.

### 1.5 LaBraM (ICLR 2024 Spotlight) -- Large Brain Model
- **Paper**: [ICLR 2024](https://openreview.net/forum?id=QzTpTRVtrP), [arXiv 2405.18765](https://arxiv.org/abs/2405.18765)
- **Method**: Segments EEG into channel patches. Trains a neural tokenizer via vector-quantized
  neural spectrum prediction, then pretrains transformers to predict masked neural codes.
- **Scale**: ~2,500 hours from ~20 datasets. Validated on abnormal detection, emotion recognition,
  event classification, gait prediction.
- **Key insight**: Channel-patch tokenization handles heterogeneous montages (different electrode
  counts/positions). Our 27-channel scalp input could be treated as 27 channel patches.

### 1.6 EEGPT (NeurIPS 2024) -- Dual Self-Supervised Pretraining
- **Paper**: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html)
- **Code**: [github.com/BINE022/EEGPT](https://github.com/BINE022/EEGPT)
- **Method**: 10M-parameter transformer. Dual SSL: (1) mask-based reconstruction of EEG patches,
  (2) spatio-temporal representation alignment (contrastive). 50% time masking + 80% channel
  masking during pretraining. Patches = 250ms windows at 256 Hz.
- **Key insight**: Reconstructing from representations (not raw signals) avoids the low-SNR problem
  of raw EEG reconstruction. Significantly outperforms BENDR on multiple tasks.
- **Relevance to us**: The dual approach (reconstruction + contrastive) may be better than either
  alone. Heavy channel masking (80%) is an extreme form of our channel dropout finding.

---

## 2. Contrastive vs. Masked Autoencoder Pretraining

### 2.1 Head-to-Head Evidence

| Aspect | Contrastive (SimCLR, BYOL) | Masked Autoencoder (MAE) | Hybrid |
|--------|---------------------------|-------------------------|--------|
| **Positive pairs** | Same-stimulus cross-subject | N/A (reconstruction) | Both |
| **Negatives needed** | SimCLR: yes; BYOL: no | No | Varies |
| **Subject invariance** | Explicitly encouraged | Implicit only | Explicit + reconstruction |
| **EEG-specific finding** | SimCLR > BYOL for EEG (2025 wearable study) | MAEEG: +5% sleep staging with few labels | EEGPT dual SSL beats both |
| **Failure mode** | Collapse without careful neg sampling | Degenerate if mask ratio wrong | More hyperparameters |

### 2.2 Key Findings from Systematic Survey (ACM Computing Surveys 2025)
- [Survey link](https://dl.acm.org/doi/10.1145/3736574)
- Contrastive frameworks with negative samples (SimCLR, Barlow Twins) consistently outperform
  self-distillation methods (BYOL, SimSiam) for EEG.
- Best augmentation combo for EEG contrastive learning:
  - **Weak**: jittering + scaling
  - **Strong**: permutation + jittering
  - **Spatial**: sensor dropout (randomly zero entire channels)
  - **Frequency**: bandstop filtering, phase shifting
- Time shift prediction as a pretext task captures temporal dynamics well.

### 2.3 EEG-DisGCMAE (ICML 2025) -- Best of Both Worlds
- **Paper**: [arXiv 2411.19230](https://arxiv.org/abs/2411.19230), [Code](https://github.com/weixinxu666/EEG_DisGCMAE)
- **Method**: Unified graph self-supervised pretraining that integrates graph contrastive
  pretraining with graph masked autoencoder pretraining. Reconstructs contrastive samples and
  contrasts the reconstructions.
- **Distillation**: Graph topology distillation from high-density teacher to low-density student.
  Directly applicable to our scalp (27ch) -> in-ear (12ch) density reduction.

### 2.4 MAEEG (Apple Research)
- **Paper**: [Apple ML Research](https://machinelearning.apple.com/research/masked-auto-encoder), [arXiv 2211.02625](https://arxiv.org/abs/2211.02625)
- **Architecture**: 6 conv layers -> 64-dim features -> Gaussian noise masking -> 8-layer
  transformer encoder -> 192-dim output.
- **Key finding**: Reconstructing a larger proportion of concentrated masked signal (not random
  sparse masking) gives better downstream performance.

---

## 3. Recommended Augmentation Strategies for EEG Contrastive Learning

Based on the literature consensus:

1. **Temporal jittering**: Add small Gaussian noise to signal amplitude (weak augmentation)
2. **Temporal scaling**: Multiply signal by random scalar per channel (weak augmentation)
3. **Sensor/channel dropout**: Zero entire channels randomly (strong spatial augmentation)
   - Already validated in our iter011 (+0.010 r from 15% channel dropout)
4. **Temporal permutation**: Shuffle temporal segments within a window (strong augmentation)
5. **Bandstop filtering**: Remove random narrow frequency bands
6. **Time shift**: Shift signals by random number of samples; predict the shift as pretext task
7. **Phase perturbation**: Random phase shifts in frequency domain

---

## 4. Applicability to Our Scalp-to-In-Ear Prediction Task

### 4.1 Why Contrastive Pretraining Could Help

Our current bottleneck is **cross-subject variability** (Subject 14 at r=0.27 vs Subject 13 at
r=0.46). A contrastive pretraining phase could:

1. **Learn subject-invariant spatial filters**: By pulling together same-timepoint representations
   across subjects (CL-SSTER paradigm), the encoder would learn projections that are robust to
   individual anatomy differences.
2. **Better initialization than closed-form**: The current CF init assumes a linear mapping. A
   pretrained encoder could capture nonlinear subject-invariant structure.
3. **Leverage unlabeled data**: We have 168 subjects from MOABB/other datasets for pretraining.
   The in-ear channels are only needed for fine-tuning.

### 4.2 Proposed Architecture for Our Task

```
Phase 1: Contrastive Pretraining (168 subjects, scalp EEG only)
  - Encoder: Depthwise spatial conv (27 -> K latent) + temporal FIR (7 taps)
  - Augmentations: channel dropout (15%) + temporal jitter + time shift
  - Contrastive loss: NT-Xent (SimCLR-style)
  - Positive pairs: same timepoint across different subjects
  - Negative pairs: different timepoints

Phase 2: Fine-tuning (12 Ear-SAAD subjects, scalp + in-ear)
  - Freeze or slow-learn the spatial encoder
  - Train output projection: K latent -> 12 in-ear channels
  - Loss: Combined MSE + correlation (our iter017 recipe)
```

### 4.3 Can Contrastive Beat MAE for Our Task?

**Yes, likely**, for three reasons:

1. **Our task is regression, not classification**: Contrastive learning that aligns representations
   across subjects directly addresses our cross-subject transfer problem. MAE reconstruction of
   raw EEG is less aligned with predicting in-ear signals.

2. **Our data is narrowband (1-9 Hz)**: With only ~4 Hz of bandwidth, there is limited spectral
   structure for MAE to reconstruct. Contrastive objectives that focus on spatial invariance are
   better suited.

3. **Small in-ear label set**: We only have 12 subjects with in-ear labels. Contrastive
   pretraining on 168 scalp-only subjects maximizes use of unlabeled data, while MAE would also
   need in-ear signals to learn the mapping.

**However**, a hybrid approach (EEGPT-style dual SSL) could be even better -- pretrain with both
contrastive alignment and masked reconstruction, then fine-tune for the in-ear prediction.

### 4.4 Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Contrastive collapse (all representations identical) | Use SimCLR with large batch + temperature tuning; avoid BYOL |
| Pretraining data mismatch (different montages) | Channel-patch tokenization (LaBraM style) or subset to common channels |
| Overfitting pretrained features to pretraining distribution | Light fine-tuning with low LR; consider linear probe first |
| Computational cost | Start with small encoder (our 7-tap FIR); scale up only if needed |

---

## 5. Concrete Next Steps

1. **iter045**: Implement SimCLR-style contrastive pretraining on 12 Ear-SAAD training subjects
   (proof of concept before scaling to 168 subjects)
2. **iter046**: Add CL-SSTER same-timepoint positive pairs (Ear-SAAD subjects hear same audiobook)
3. **iter047**: Scale pretraining to external datasets (168 subjects)
4. **iter048**: Hybrid contrastive + reconstruction pretraining (EEGPT-inspired dual SSL)
5. **iter049**: Graph-based distillation from scalp to in-ear (EEG-DisGCMAE inspired)

---

## Sources

- [BENDR -- Transformers + Contrastive SSL for EEG](https://arxiv.org/abs/2101.12037)
- [CL-SSTER -- Shared Spatiotemporal EEG Representations (NeuroImage 2024)](https://www.sciencedirect.com/science/article/pii/S1053811924003872)
- [CLISA -- Subject-Invariant EEG for Emotion Recognition](https://arxiv.org/abs/2109.09559)
- [CLUDA -- Multi-Source Domain Adaptation for EEG (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0957417424023194)
- [LaBraM -- Large Brain Model (ICLR 2024 Spotlight)](https://arxiv.org/abs/2405.18765)
- [EEGPT -- Dual SSL Pretrained Transformer (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html)
- [EEG-DisGCMAE -- Graph Contrastive MAE (ICML 2025)](https://arxiv.org/abs/2411.19230)
- [MAEEG -- Apple Masked Autoencoder for EEG](https://arxiv.org/abs/2211.02625)
- [ACM Survey -- Self-supervised Learning for EEG (2025)](https://dl.acm.org/doi/10.1145/3736574)
- [Cross-Subject EEG Emotion Recognition with Contrastive Learning (2025)](https://www.nature.com/articles/s41598-025-13289-5)
- [Wearable EEG SSL Evaluation (2025)](https://arxiv.org/html/2510.07960v1)
- [EEG Challenge 2025 -- Regression Tasks](https://eeg2025.github.io)
- [CET-MAE -- Contrastive EEG-Text Masked Autoencoder (ACL 2024)](https://aclanthology.org/2024.acl-long.393/)
