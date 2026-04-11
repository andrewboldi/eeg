# Domain Adaptation for Regression Tasks

Research survey focused on DA/TTA techniques applicable to **regression** (continuous output prediction), not classification. Compiled April 2026 for the EEG scalp-to-in-ear prediction project.

## Why This Matters

Most domain adaptation literature targets classification (discrete labels). Our EEG prediction task is a **regression** problem: predict 12 continuous in-ear EEG channels from 27 scalp channels. Cross-subject variability is the bottleneck (Subject 14 ~0.27 r vs Subject 13 ~0.46 r), making domain adaptation critical.

---

## 1. TikUDA: Tikhonov-Regularized UDA for Regression

**Paper:** [Efficient Unsupervised Domain Adaptation Regression for Spatial-Temporal Sensor Fusion](https://arxiv.org/abs/2411.06917) (2024, updated Aug 2025)

**Core idea:** Align the inverse Gram matrices (perturbed by a Tikhonov/ridge regularization term) between source and target domain feature representations. This is equivalent to aligning the closed-form ridge regression solutions across domains.

**Method:**
- Extract features via a shared encoder (e.g., Spatial-Temporal Graph Neural Network)
- Compute the Tikhonov-regularized inverse Gram matrix: `(X^T X + lambda I)^{-1}` for both source and target features
- Minimize the distance between source and target inverse Gram matrices
- The Tikhonov regularization ensures numerical stability (full-rank), enabling Cholesky decomposition instead of SVD
- Combined with supervised regression loss on labeled source data

**Why relevant to us:**
- Directly addresses regression (not classification)
- Tested on EEG signal reconstruction tasks
- Our FIR model is essentially a ridge regression -- aligning the Gram matrices across subjects would directly reduce cross-subject shift
- Computationally efficient (Cholesky decomposition)

**Applicability:** HIGH. Could add a TikUDA alignment loss to our FIR training, treating each subject as a domain.

---

## 2. WANN: Adversarial Weighting for DA in Regression

**Paper:** [Adversarial Weighting for Domain Adaptation in Regression](https://arxiv.org/abs/2006.08251) (IEEE ICMLA 2021)

**Core idea:** Learn importance weights for source instances using an adversarial network that estimates the Y-discrepancy (a regression-specific divergence measure).

**Method:**
- Train a weighting network to assign importance weights to source samples
- The weighting network is trained adversarially to maximize the weighted empirical Y-discrepancy
- The task network minimizes the weighted regression loss
- Outperforms KMM and KLIEP (kernel-based importance weighting methods)

**Key insight:** Instance-based approaches (reweighting source samples) outperform feature-based approaches (aligning distributions) for regression under covariate shift.

**Applicability:** MEDIUM. Could weight training subjects by similarity to the test subject's feature distribution. However, requires unlabeled target data during training.

**Code:** https://github.com/antoinedemathelin/wann

---

## 3. TASFAR: Target-Agnostic Source-Free DA for Regression

**Paper:** [Target-agnostic Source-free Domain Adaptation for Regression Tasks](https://arxiv.org/abs/2312.00540) (ICDE 2024)

**Core idea:** Adapt a pre-trained regression model to a new target domain without access to source data or target labels. Uses prediction confidence to estimate the target label distribution, then calibrates the model.

**Method:**
- Train source model normally
- At adaptation time (no source data available):
  1. Run source model on unlabeled target data
  2. Use prediction confidence to estimate a label density map
  3. Calibrate the model using the estimated target label distribution
- First source-free DA method specifically for regression

**Why relevant to us:**
- Directly applicable: train on subjects 1-12, adapt to subject 13/14/15 without their labels
- Source-free: no need to retrain on source data during adaptation
- Preserves privacy (important for clinical EEG)

**Applicability:** HIGH. Could be used as a post-training calibration step for each test subject.

---

## 4. Multi-Source DA for Regression

**Paper:** [Multi-source domain adaptation for regression](https://arxiv.org/abs/2312.05460) (2023)

**Core idea:** Extend single-source DA to multiple sources via outcome-coarsening and ensemble learning.

**Method:**
- Coarsen the continuous outcome into bins (discretize temporarily)
- Apply classification-based DA within each bin
- Combine single-source adapted models via ensemble weighting
- Each source domain contributes proportionally to its relevance

**Why relevant to us:**
- Each training subject is a "source domain"
- Could weight subjects 1-12 differently for each test subject
- Ensemble of per-subject models with adaptive weighting

**Applicability:** MEDIUM. The coarsening step loses information, but the multi-source ensemble idea is sound.

---

## 5. Covariance-Based Alignment for EEG Regression

**Paper:** [Harmonizing and aligning M/EEG datasets with covariance-based techniques to enhance predictive regression modeling](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00040/118217) (Imaging Neuroscience, 2023; Mellot, Collas, Rodrigues, Engemann, Gramfort)

**Core idea:** Align EEG covariance matrices across subjects/sites using Riemannian geometry before fitting regression models.

**Method (3-step alignment):**
1. **Re-centering:** Shift each subject's mean covariance to the identity (Euclidean alignment)
2. **Re-scaling:** Normalize the dispersion of covariance matrices per subject
3. **Rotation correction:** Align principal axes across subjects (Procrustes on the SPD manifold)

**Key results:**
- Riemannian-based models are robust to preprocessing choices and model violations
- Alignment significantly improves cross-domain regression (tested on age prediction from M/EEG)
- Re-centering alone captures most of the benefit
- Works with standard ML pipelines (ridge regression on tangent vectors)

**Why relevant to us:**
- Directly tested on EEG regression (not classification)
- Our iter018 (Euclidean alignment) failed, but used noisy per-batch covariance. This paper suggests per-subject covariance alignment is the correct approach
- The 3-step procedure (re-center, re-scale, rotate) may recover what simple Euclidean alignment missed

**Applicability:** HIGH. Re-implement Euclidean alignment correctly: compute per-subject (not per-batch) covariance matrices, align to geometric mean, then train the FIR model on aligned data.

---

## 6. Test-Time Normalization (TTN)

**Paper:** [TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation](https://arxiv.org/abs/2302.05155) (ICLR 2023)

**Core idea:** Interpolate between source BatchNorm statistics and test-batch statistics, with learnable per-layer weights that reflect domain-shift sensitivity.

**Method:**
- During training: standard BatchNorm with running statistics
- At test time: compute test-batch mean/variance
- Interpolate: `mu_combined = alpha * mu_source + (1-alpha) * mu_test`
- Alpha is learned per BN layer based on domain-shift sensitivity

**Why relevant to us:**
- Simple to implement: just modify BatchNorm behavior at test time
- Could help with subject-specific distribution shifts
- No retraining needed

**Limitation:** Our best model (FIR) has no BatchNorm layers. Would only apply if we add normalization layers, and we already found InstanceNorm hurts.

**Applicability:** LOW for current architecture, but worth noting for deeper models.

---

## 7. Self-Supervised Test-Time Training for Time Series

**Paper:** [Test Time Learning for Time Series Forecasting](https://openreview.net/forum?id=WWymYrA48K)

**Core idea:** Use Test-Time Training (TTT) modules that learn representations via self-supervised objectives during inference.

**Method:**
- TTT modules replace standard sequence modeling layers
- At test time, the model updates internal representations using a self-supervised loss (e.g., reconstruction)
- Captures long-range dependencies specific to the test distribution
- Significant improvements on Electricity, Traffic, and Weather datasets

**Why relevant to us:**
- Could use masked channel reconstruction as a self-supervised objective at test time
- Adapts to each test subject's specific EEG patterns without labels

**Applicability:** MEDIUM. Requires architectural changes (TTT modules) but the idea of self-supervised adaptation at test time is promising.

---

## 8. Deep CORAL for Regression

**Paper:** [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719) (ECCV-W 2016)

**Core idea:** Minimize the Frobenius norm distance between source and target feature covariance matrices as a regularization loss during training.

**Method:**
- CORAL loss: `L_coral = (1/4d^2) * ||C_s - C_t||_F^2`
- Where C_s, C_t are the covariance matrices of source and target features
- Added as a regularizer to the task loss
- "Frustratingly easy" to implement

**Why relevant to us:**
- Can be added to any neural network with minimal code changes
- Aligns second-order statistics of intermediate features across domains
- Originally for classification but the covariance alignment is task-agnostic

**Applicability:** MEDIUM-HIGH. Add CORAL loss between subjects during training to encourage domain-invariant features.

---

## 9. Riemannian Domain Adaptation for EEG

**Paper:** [Physics-informed and Unsupervised Riemannian Domain Adaptation for Machine Learning on Heterogeneous EEG Datasets](https://arxiv.org/abs/2403.15415) (2024)

**Core idea:** Use Riemannian geometry on SPD covariance matrices for source-free, unsupervised DA across EEG datasets with different electrode configurations.

**Method:**
- Represent EEG data as SPD covariance matrices on the Riemannian manifold
- Use physics-informed priors (forward models) to handle different montages
- Align distributions on the SPD manifold without target labels

**Applicability:** LOW for our setup (same montage across subjects), but the Riemannian alignment ideas reinforce approach #5.

---

## Ranked Recommendations for Our Project

### Tier 1: Try First (highest expected impact, easiest to implement)

1. **Correct Euclidean Alignment (from paper #5)**
   - Compute per-subject covariance matrices (not per-batch like iter018)
   - Re-center each subject's covariance to identity
   - This directly addresses our biggest failure mode (cross-subject variability)
   - Implementation: ~30 lines of preprocessing code

2. **CORAL Loss (from paper #8)**
   - Add `||cov(features_subj_i) - cov(features_subj_j)||_F^2` as regularizer
   - Encourages the FIR model to produce subject-invariant representations
   - Implementation: ~10 lines added to training loop

3. **TikUDA Gram Matrix Alignment (from paper #1)**
   - Align `(X^T X + lambda I)^{-1}` across training subjects
   - Natural fit since our model IS a ridge regression
   - Implementation: ~50 lines

### Tier 2: Try Next (moderate complexity)

4. **TASFAR Source-Free Calibration (from paper #3)**
   - Train normally, then calibrate predictions per test subject
   - No architectural changes needed
   - Requires implementing the label density estimation

5. **Subject Importance Weighting (inspired by paper #2)**
   - Weight training subjects by feature similarity to test subject
   - Simple: compute MMD between test subject features and each training subject
   - Use weights in loss function

### Tier 3: Exploratory (higher complexity)

6. **Self-Supervised TTA with Channel Masking (from paper #7)**
   - Mask some scalp channels, predict them from others as auxiliary task
   - Adapt model to test subject using this self-supervised loss
   - Requires architectural changes

7. **Multi-Source Ensemble (from paper #4)**
   - Train per-subject models, ensemble with adaptive weights at test time

---

## Key Takeaway

The most promising direction for our specific problem is **proper covariance-based alignment** (paper #5). Our previous attempt (iter018) failed because it used per-batch covariance estimates, which are noisy. The Mellot et al. paper shows that per-subject alignment with re-centering, re-scaling, and rotation correction significantly improves EEG regression across domains. This should be the next iteration to try.

The second most promising is **CORAL loss** as a training regularizer -- it's trivial to implement and directly encourages domain-invariant features.

---

## Sources

- [TikUDA - Efficient UDA Regression for Sensor Fusion](https://arxiv.org/abs/2411.06917)
- [WANN - Adversarial Weighting for DA in Regression](https://arxiv.org/abs/2006.08251) | [Code](https://github.com/antoinedemathelin/wann)
- [TASFAR - Target-agnostic Source-free DA for Regression](https://arxiv.org/abs/2312.00540)
- [Multi-source DA for Regression](https://arxiv.org/abs/2312.05460)
- [Harmonizing M/EEG with Covariance Alignment for Regression](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00040/118217)
- [TTN - Domain-Shift Aware Batch Normalization](https://arxiv.org/abs/2302.05155)
- [Test Time Learning for Time Series Forecasting](https://openreview.net/forum?id=WWymYrA48K)
- [Deep CORAL](https://arxiv.org/abs/1607.01719)
- [Riemannian DA for Heterogeneous EEG](https://arxiv.org/abs/2403.15415)
- [Revisiting Euclidean Alignment for EEG Transfer Learning](https://arxiv.org/abs/2502.09203)
- [Systematic Evaluation of Euclidean Alignment with Deep Learning for EEG](https://arxiv.org/abs/2401.10746)
- [Self-supervised Learning for EEG Survey](https://arxiv.org/abs/2401.05446)
