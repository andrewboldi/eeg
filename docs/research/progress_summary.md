# EEG Scalp-to-In-Ear Prediction: Research Progress Summary

**Project**: Ear-SAAD scalp-to-in-ear EEG prediction (Geirnaert et al. 2025)
**Date**: April 11, 2026
**Iterations completed**: 054 (001-007 synthetic, 007-036 narrowband, 037-054 broadband)

---

## 1. Executive Summary

This research project aims to predict 12-channel in-ear EEG from scalp EEG recordings using the Ear-SAAD dataset (15 subjects). Starting from a simple closed-form linear spatial filter (r=0.366), we systematically explored over 40 model architectures, loss functions, normalization strategies, and data representations. The two largest breakthroughs came not from model complexity but from data engineering: switching from narrowband (1-9 Hz) to broadband (1-45 Hz) data added +0.06 r, and incorporating 19 around-ear cEEGrid channels as additional inputs added +0.15 r. Together these lifted performance from r=0.378 to r=0.638, a 69% improvement.

Despite testing 15+ architecture variants on the broadband 46-channel data -- including ensembles, adversarial training, cross-attention decoders, spectral losses, spatial positional encodings, and test-time adaptation -- none surpassed the simple tiny deep model (r=0.638). A full 15-subject leave-one-subject-out (LOSO) evaluation with the closed-form baseline yielded r=0.645 (95% CI [0.563, 0.728]), confirming the model generalizes but with enormous subject variance (r=0.38 for Subject 8 to r=0.94 for Subject 3). The dominant bottleneck is now clearly physiological: subject difficulty correlates strongly with intrinsic scalp-to-in-ear coupling strength (r=0.77), not data quality or model capacity.

---

## 2. Timeline of Breakthroughs

| Date/Phase | Iteration | Breakthrough | Impact |
|------------|-----------|-------------|--------|
| Synthetic phase | iter001-006 | FIR center-tap CF initialization | Solved gradient training convergence (r=0.887 synthetic) |
| Real data launch | iter007 | First real-data baseline | CF r=0.366 on Ear-SAAD narrowband |
| Narrowband optimization | iter009 | FIR spatio-temporal filter | +0.007 r (0.366 -> 0.373) |
| | iter011 | Channel dropout augmentation | +0.003 r (0.373 -> 0.376) |
| | iter017 | Combined MSE+corr loss + corr validation | +0.002 r (0.376 -> 0.378) |
| | iter020-036 | 16 experiments hit the 0.378 ceiling | Proved narrowband limit |
| Broadband pivot | iter038 | Broadband FIR (1-45 Hz, 256 Hz, 27ch) | +0.06 r over narrowband |
| 46-channel expansion | iter039 | 46-channel input (27 scalp + 19 cEEGrid) | **+0.15 r, new best r=0.638** |
| Architecture search | iter040-054 | 15 architecture variants tested | None beat tiny model |
| Full LOSO | -- | 15-subject closed-form evaluation | r=0.645, huge subject variance confirmed |
| Subject analysis | -- | Coupling strength analysis | Identified physiological bottleneck |

---

## 3. What We Tried and What Worked/Failed

### What Worked

| Technique | Gain | Details |
|-----------|------|---------|
| **Broadband data (1-45 Hz)** | +0.06 r | Raw BIDS download, MNE preprocessing to 256 Hz |
| **Around-ear channels as input** | +0.15 r | 19 cEEGrid channels added (46 total input channels) |
| **FIR spatio-temporal filter** | +0.007 r | 7-tap acausal FIR, CF center-tap initialization |
| **Channel dropout (15%)** | +0.003 r | Regularization during training |
| **Combined MSE+corr loss** | +0.002 r | With correlation-based early stopping |
| **Mixup augmentation** | +0.0003 r | Only without InstanceNorm |
| **Closed-form initialization** | Critical | W* = R_YX @ inv(R_XX) as starting point for gradient methods |

### What Failed (Narrowband Phase, iter013-036)

| Technique | Result | Why It Failed |
|-----------|--------|---------------|
| Band-specific spatial filters | -0.023 r | Splitting bands loses cross-frequency information |
| Euclidean alignment | -0.009 r | Noisy batch covariance estimates |
| InstanceNorm | -0.003 r | Removes useful amplitude dynamics |
| Longer FIR filters (11, 15 taps) | No gain | 7 taps sufficient for 1-9 Hz |
| Causal-only lags | -0.013 r | Acausal filtering is critical |
| Pure correlation loss | -0.003 r, -6 dB SNR | Degenerate scale |
| Ledoit-Wolf shrinkage | -0.009 r | Causal constraint hurt more than shrinkage helped |
| MoE, residual learning, ensembles | No gain | All within noise of baseline |
| PLS regression | No gain | Identical to OLS spatial filters |
| Huber loss | No gain | Outlier robustness not needed |
| Noise augmentation | No gain | Combined with InstanceNorm |
| Cosine loss schedule | No gain | MSE-to-corr annealing ineffective |
| High dropout (25%) | No gain | 15% already optimal |

### What Failed (Broadband 46ch Phase, iter040-054)

| Iteration | Technique | Mean r | vs Best (0.638) |
|-----------|-----------|--------|-----------------|
| iter040 | BIOT fine-tune | -- | Did not converge |
| iter041 | Subject adaptation | 0.587 | -0.051 |
| iter042 | Euclidean alignment | 0.574 | -0.064 |
| iter043 | CF+Deep ensemble | 0.606 | -0.032 |
| iter044 | Residual CF | 0.610 | -0.028 |
| iter045 | Long context (4s windows) | 0.606 | -0.032 |
| iter046 | Calibrated output | 0.581 | -0.057 |
| iter047 | Spatial positional encoding (REVE-inspired) | 0.589 | -0.049 |
| iter048 | NeuroTTT SSL adaptation | 0.597 | -0.041 |
| iter049 | Adversarial subject-invariant | 0.608 | -0.030 |
| iter050 | L1 loss (REVE-style) | 0.605 | -0.033 |
| iter051 | Per-channel output heads | 0.609 | -0.029 |
| iter052 | Cross-attention decoder | 0.596 | -0.042 |
| iter053 | Spectral loss | 0.604 | -0.034 |
| iter054 | Pretrained fine-tune (HBN-EEG) | 0.589 | -0.049 |

### Key Negative Finding: Scaling Law Is Flat

A scaling law experiment (55K to 7M parameters) showed only +0.008 test r across two orders of magnitude in model size. The problem is not model capacity.

---

## 4. Current State

### Best Model
- **Model**: iter039 tiny deep model (46ch broadband)
- **Test r**: 0.638 (subjects 13, 14, 15)
- **Full LOSO r**: 0.645 (95% CI [0.563, 0.728]) with closed-form baseline
- **Architecture**: Small convolutional network with CF initialization
- **Input**: 46 channels (27 scalp + 19 around-ear), 1-45 Hz, 256 Hz sampling

### Data Inventory
| Dataset | Status | Details |
|---------|--------|---------|
| Ear-SAAD narrowband | Processed | 1-9 Hz, 20 Hz, 27ch, 2s windows |
| Ear-SAAD broadband 27ch | Processed | 1-45 Hz, 256 Hz, 27ch scalp only |
| Ear-SAAD broadband 46ch | Processed | 1-45 Hz, 256 Hz, 46ch (scalp + cEEGrid) |
| HBN-EEG (pilot) | Downloaded | 20 subjects, montage mismatch -- did not help |
| MOABB datasets | Not downloaded | Potential for pretraining |

### Infrastructure
- **Hardware**: RTX 4060 (8GB VRAM), 30GB RAM
- **Benchmarking**: Automated LOSO pipeline with leaderboard tracking
- **Preprocessing**: MNE-based pipeline for broadband data (scripts/preprocess_broadband*.py)
- **Analysis tools**: Subject-level analysis, scaling law experiments, per-channel diagnostics

### Subject-Level Performance (Full 15-Subject LOSO, CF Baseline)

| Category | Subjects | Mean r | Range |
|----------|----------|--------|-------|
| Easy (r >= 0.7) | 3, 4, 5, 9, 11, 13 | 0.791 | 0.728 - 0.940 |
| Medium (0.55 <= r < 0.7) | 1, 6, 7, 10, 15 | 0.627 | 0.575 - 0.653 |
| Hard (r < 0.55) | 2, 8, 12, 14 | 0.475 | 0.382 - 0.548 |

---

## 5. Top 5 Next Priorities

### Priority 1: Channel-Weighted Loss Function
**Rationale**: Hard subjects have 2-3 in-ear channels that are fundamentally unpredictable (best scalp correlation < 0.3). Training with equal weight on all channels wastes model capacity on these channels and may distort spatial filters. Weighting loss by estimated channel predictability could improve both hard and easy subjects.
**Expected gain**: +0.01-0.03 r
**Effort**: Low (1 iteration)

### Priority 2: Subject-Specific Fine-Tuning with Calibration Data
**Rationale**: The subject variance (std=0.14) represents the largest source of error. Even a short calibration segment (30-60 seconds) from the target subject could dramatically improve spatial filter alignment. Few-shot adaptation of the trained model's last layers is the most direct path to closing the cross-subject gap.
**Expected gain**: +0.03-0.08 r
**Effort**: Medium (2-3 iterations to get right)

### Priority 3: Large-Scale Pretraining on External EEG Data
**Rationale**: With only 12 training subjects, the model cannot learn robust cross-subject representations. Pretraining on large EEG datasets (TUH EEG Corpus: 30,000+ recordings; Temple University Hospital) with a self-supervised objective, then fine-tuning on Ear-SAAD, could provide better initialization than closed-form. The HBN-EEG pilot failed due to montage mismatch (20 subjects insufficient); a larger dataset with better channel overlap is needed.
**Expected gain**: +0.05-0.10 r
**Effort**: High (5+ iterations, data download, montage alignment)

### Priority 4: Hyperparameter Optimization (Optuna Sweep)
**Rationale**: The tiny deep model was designed by hand. A systematic sweep over learning rate, weight decay, dropout rate, number of layers, hidden dimensions, FIR tap count, and loss weighting could find a better configuration. 50 Optuna trials with 3-fold CV would take approximately 4 hours on the RTX 4060.
**Expected gain**: +0.01-0.03 r
**Effort**: Medium (1 iteration to set up, 4 hours compute)

### Priority 5: Frequency-Domain / Wavelet Approaches
**Rationale**: All current models operate in the time domain. Frequency-domain approaches (STFT, wavelet decomposition) could capture band-specific spatial patterns that the broadband FIR misses. The broadband data (1-45 Hz) contains alpha, beta, and gamma bands that may have distinct spatial transfer functions from scalp to in-ear.
**Expected gain**: +0.01-0.05 r
**Effort**: Medium (2-3 iterations)

---

## 6. Open Questions

### Fundamental / Scientific
1. **What is the theoretical ceiling?** Subject 3 achieves r=0.94 with just a linear filter. If all subjects had Subject-3-level coupling, mean r would be ~0.94. But coupling is physiological -- what fraction of the gap is closable with better algorithms vs. being a hard anatomical limit?

2. **Is the task novel?** The original Ear-SAAD paper performs auditory attention decoding (classification), not scalp-to-in-ear reconstruction (regression). We may be the first to attempt this specific prediction task. How should we frame this for publication?

3. **Why do normalization methods consistently hurt?** InstanceNorm, RevIN, and Euclidean alignment all degrade performance. This suggests that subject-specific amplitude information is a useful signal for prediction, not noise. But this contradicts standard practice in cross-subject EEG. Is the Ear-SAAD dataset unusual, or is this a general finding?

### Technical / Methodological
4. **Is the 3-subject test set representative?** Subjects 13 (r=0.728), 14 (r=0.433), and 15 (r=0.601) span easy-to-hard. But with only 3 test subjects, variance is high. Should we switch to full 15-subject LOSO as the primary metric?

5. **Can we identify hard channels at test time?** If we could detect which in-ear channels have poor coupling during a brief calibration period, we could mask them out or weight them down. This is a realistic deployment scenario.

6. **Why did the scaling law saturate?** 55K to 7M parameters gave only +0.008 r. Is this because (a) the mapping is fundamentally linear, (b) the dataset is too small for large models, or (c) we haven't found the right architecture? The fact that CF (linear) achieves r=0.645 suggests the mapping is mostly linear, but some subjects might benefit from nonlinear components.

7. **Would longer temporal context help with the right architecture?** The 4-second context experiment (iter045) did not improve over 2-second windows. But transformers with proper positional encoding might capture slower dynamics that convolutions miss. This remains untested with the right setup.

### Data / Infrastructure
8. **Which external EEG dataset has the best channel overlap with Ear-SAAD?** The HBN-EEG pilot failed partly due to montage mismatch. We need a dataset with temporal/parietal electrodes in similar locations to the Ear-SAAD 10-20 montage.

9. **Can we use the around-ear channels as an intermediate prediction target?** Multi-task learning (predict around-ear + in-ear jointly) could provide a useful inductive bias, since around-ear channels are geometrically between scalp and in-ear.

10. **Is there value in per-subject model selection?** Different architectures might work better for different subjects. An oracle that picks the best model per subject could reveal whether the single-model approach is leaving performance on the table.
