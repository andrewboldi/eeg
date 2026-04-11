# Few-Shot Subject Calibration for BCI/EEG

Research survey compiled 2026-04-11. Focus: adapting a cross-subject EEG decoder
to a new subject using minimal calibration data (1 minute ~ 30 windows at 20 Hz, 2s).

## Problem Statement

Our scalp-to-in-ear model achieves mean r=0.378 cross-subject, but per-subject
performance varies wildly (Subject 13: r~0.46, Subject 14: r~0.27). If we could
use even 1 minute of a new subject's paired scalp+in-ear data, we could dramatically
improve predictions for poor-performing subjects.

We have ~3600 windows (2s each = 2 hours) per subject. Even 30 windows (1 minute)
would be a realistic calibration session.

---

## 1. Euclidean Alignment (EA) -- Unsupervised, No Labels Needed

**Source**: He & Wu 2020; revisited in arXiv:2502.09203 (2025)

**Method**: Whiten each subject's EEG so the mean covariance matrix becomes the
identity. For a new subject with trials {X_i}:

1. Compute mean covariance: R = (1/N) * sum(X_i @ X_i^T)
2. Align: X_i_aligned = R^{-1/2} @ X_i

**Key properties**:
- Completely unsupervised -- needs ONLY the input scalp EEG, no labels (in-ear targets)
- Can be computed from as few as ~30 trials (need stable covariance estimate of 27x27 matrix)
- Average improvement: +4.33% accuracy across datasets and architectures
- Already a preprocessing step, compatible with any downstream model

**Relevance to our project**: We tried batch-wise EA in iter018 and it hurt (-0.009 r)
because per-batch covariance was noisy. The fix is to compute EA over ALL available
calibration data for the new subject (not per-batch). With 30+ windows of the new
subject's scalp EEG, we get a stable 27x27 covariance for alignment.

**Implementation plan**:
```python
# At calibration time (input-only, no labels needed):
R_new = np.mean([X @ X.T for X in new_subject_windows], axis=0)
R_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(R_new))
X_aligned = R_inv_sqrt @ X_new  # for each window

# Also align training subjects the same way (precompute per-subject)
```

---

## 2. Test-Time Adaptation (TTA) -- Unsupervised Online

**Source**: arXiv:2311.18520 (2024), arXiv:2509.19403 (2025)

**Methods** (can be combined):

### 2a. Adaptive Batch Normalization
- Replace training-set BN statistics (mean, var) with test-subject statistics
- Requires only a forward pass through calibration data, no backprop
- Tent (BatchNorm Entropy Minimization): adapt only BN layers via entropy loss

### 2b. Entropy Minimization
- During inference, minimize prediction entropy on unlabeled test data
- Updates only normalization layers, not general weights
- Works well when combined with EA preprocessing

### 2c. Dual-Stage Alignment + Self-Supervision (arXiv:2509.19403)
- Stage 1: Euclidean alignment on input data
- Stage 2: Adjust BN statistics on intermediate representations
- Self-supervised loss with soft pseudo-labels (no true labels needed)
- Single-trial online updates possible
- Results: +4.9% for SSVEP, +3.6% for motor imagery

**Relevance**: Our FIR model has no BN layers, but we could add a lightweight
normalization layer specifically for subject adaptation. Or use EA + fine-tuning
the spatial filter weights.

---

## 3. Meta-Learning (MAML / Reptile) -- Supervised, Few-Shot

**Source**: arXiv:2412.19725 (EEG-Reptile, 2024); META-EEG (2024)

**MAML for EEG**:
- Train model to be easily fine-tunable from few examples
- Meta-training: simulate few-shot adaptation across training subjects
- Meta-testing: fine-tune on K examples from new subject
- Typical K: 1, 5, 10, 20 shots per class (for classification)

**Reptile (simpler alternative to MAML)**:
- First-order approximation, no second derivatives
- Train on subject i for N steps, then interpolate weights toward result
- Repeat across subjects -> model sits at a good initialization point
- EEG-Reptile library: supports EEGNet, FBCNet, EEG-Inception architectures
- Automatically filters out subjects too different from others
- Improvement in both zero-shot AND few-shot scenarios

**For our regression task**:
- Instead of K-shot classification, we'd do K-window regression
- Meta-train: for each subject in {1..12}, simulate "adapt on 30 windows, evaluate on rest"
- Meta-test: adapt on 30 windows of new subject, evaluate
- Expected: model learns a good initialization that adapts quickly

**Implementation sketch**:
```python
# Meta-training loop (Reptile-style)
for epoch in range(meta_epochs):
    for subj in train_subjects:
        # Clone model weights
        fast_weights = clone(model.parameters())
        # Inner loop: adapt on K windows from this subject
        for step in range(inner_steps):
            loss = mse(model(X_subj[:K]), Y_subj[:K])
            fast_weights = fast_weights - inner_lr * grad(loss)
        # Outer loop: move toward adapted weights
        model.parameters() += outer_lr * (fast_weights - model.parameters())
```

---

## 4. Subject-Adaptive Transfer via Resting State EEG

**Source**: arXiv:2405.19346 (MICCAI 2024)

**Key insight**: Use resting-state EEG (eyes closed, no task) to extract
subject-specific features, then use these to adapt the model. Resting state
is trivial to collect (subject sits still for 1 minute).

**Method**:
- Disentangle features into task-dependent and subject-dependent components
- Use resting-state EEG to estimate subject-dependent features
- Calibrate the model using these features without task-specific data

**Relevance**: For our use case, we could collect 1 minute of resting scalp EEG
from a new subject to compute their spatial covariance structure, then use this
to adapt the spatial filter. No in-ear electrodes needed during calibration.

---

## 5. Fine-Tuning Strategies with Minimal Labeled Data

### 5a. Freeze-and-Fine-Tune
- Train full model on subjects 1-12
- Freeze temporal (FIR) weights, fine-tune only spatial weights on new subject
- Spatial filter is 27->12 linear map = 324 parameters
- With 30 windows * 40 timepoints = 1200 samples per channel, easily enough

### 5b. Linear Probing
- Freeze entire model, learn only a per-subject scale+bias per channel
- 12 scales + 12 biases = 24 parameters
- Can be fit in closed form with OLS from ~30 windows

### 5c. Low-Rank Adaptation (inspired by LoRA)
- Add a low-rank update to the spatial filter: W_new = W_base + A @ B
- A is 27 x r, B is r x 12, with rank r = 2-4
- Only 27*r + r*12 = ~150 parameters to learn
- Few-shot friendly, regularizes naturally

---

## 6. Semi-Supervised / Unsupervised Calibration (No Labels)

### 6a. Input-Only Covariance Alignment (EA)
See Section 1. Only needs scalp EEG, no in-ear targets.

### 6b. Self-Supervised Contrastive Pre-Adaptation
- Use new subject's unlabeled scalp EEG to adapt encoder
- Contrastive loss: windows close in time should have similar representations
- No in-ear data needed at all

### 6c. Distribution Matching
- Match feature statistics (mean, covariance) of new subject to training distribution
- Optimal transport or moment matching
- Completely unsupervised

### 6d. Online Semi-Supervised (if partial labels available)
- EEGMatch framework: leverage both labeled and unlabeled EEG data
- Mixup-based augmentation + pseudo-label refinement
- Can work with as few as 5-10 labeled windows + many unlabeled

---

## 7. How Many Calibration Samples Are Needed?

| Method | Samples Needed | Labels Needed? | Expected Gain |
|--------|---------------|----------------|---------------|
| Euclidean Alignment | 30+ windows (1 min) | No | +2-5% |
| BN Statistics Update | 30+ windows (1 min) | No | +2-4% |
| Linear probing (scale+bias) | 30+ windows (1 min) | Yes (in-ear) | +5-10% |
| Spatial filter fine-tune | 60+ windows (2 min) | Yes (in-ear) | +10-20% |
| MAML/Reptile few-shot | 10-30 windows | Yes (in-ear) | +5-15% |
| LoRA-style adaptation | 30-60 windows | Yes (in-ear) | +5-15% |
| Full fine-tuning | 300+ windows (10 min) | Yes (in-ear) | +15-25% |

For our setup (27-channel input, 12-channel output, 2s windows at 20 Hz):
- 30 windows = 1 minute of data = 1200 timepoints
- Covariance matrix is 27x27 = 378 unique elements -> 30 windows gives ~3x oversampling
- Spatial filter has 27*12 = 324 params -> 30 windows is tight but feasible with regularization
- Linear probing (24 params) is trivially feasible from 30 windows

---

## 8. Recommended Implementation Priority

### Priority 1: Euclidean Alignment (unsupervised, immediate)
- Compute per-subject EA on training data (subjects 1-12)
- At test time, compute EA from all available test windows
- No architecture changes needed, pure preprocessing
- This fixes our iter018 failure (which used per-batch EA)

### Priority 2: Per-Subject Scale+Bias (supervised, 24 params)
- After EA, learn per-channel affine transform: y_cal = a * y_pred + b
- Closed-form OLS from 30 labeled windows
- Combines with any base model

### Priority 3: Reptile Meta-Learning (supervised, few-shot optimized)
- Restructure training loop for meta-learning
- Each episode: adapt on K windows from one subject, evaluate on rest
- At test time: fine-tune spatial filter on K windows from new subject
- More complex but potentially the biggest gain

### Priority 4: Spatial Filter Fine-Tuning with LoRA
- Add low-rank subject-specific adaptation to spatial filter
- Train base model normally, then meta-learn the LoRA parameters
- Elegant balance of capacity and regularization

---

## 9. Key Papers and Resources

- [Calibration-free online test-time adaptation for EEG MI decoding](https://arxiv.org/abs/2311.18520) -- OTTA methods (EA + BN + entropy)
- [Subject-Adaptive Transfer Learning Using Resting State EEG](https://arxiv.org/abs/2405.19346) -- MICCAI 2024, resting-state adaptation
- [EEG-Reptile: Automatized Reptile-Based Meta-Learning for BCIs](https://arxiv.org/abs/2412.19725) -- Ready-to-use meta-learning library
- [Harnessing Few-Shot Learning for EEG: Survey](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1421922/full) -- Comprehensive 2024 survey
- [Revisiting Euclidean Alignment for Transfer Learning in EEG](https://arxiv.org/abs/2502.09203) -- Systematic evaluation of EA
- [A Systematic Evaluation of Euclidean Alignment with Deep Learning](https://arxiv.org/abs/2401.10746) -- EA + deep learning
- [Harmonizing M/EEG datasets with covariance-based techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007539/) -- Covariance alignment for regression
- [Online Adaptation via Dual-Stage Alignment for Fast-Calibration BCIs](https://arxiv.org/abs/2509.19403) -- Single-trial online adaptation
- [MAML-EEG: Meta-learning for Unseen Subject MI Classification](https://www.researchgate.net/publication/391551953) -- MAML for EEG domain generalization
- [Spatial Filtering of EEG as a Regression Problem](https://ieeexplore.ieee.org/document/7888547/) -- Spatial filters via regression

---

## 10. Concrete Next Steps for Our Project

1. **iter041_subject_adapt**: Implement proper EA (full-subject covariance, not per-batch)
   + per-subject scale+bias calibration. Evaluate in simulated few-shot LOSO:
   train on subjects 1-12, calibrate with 30 windows of test subject, evaluate on rest.

2. **iter042_reptile_meta**: Implement Reptile meta-learning wrapper around our FIR model.
   Meta-train across subjects 1-12, meta-test with K=30 windows on held-out subjects.

3. **iter043_lora_spatial**: Add rank-2 LoRA to spatial filter layer. Meta-learn the
   LoRA parameters for fast adaptation. Compare against full fine-tuning.

4. **Unsupervised baseline**: Test EA-only (no labels) to establish floor for
   unsupervised calibration. If EA alone helps, we can calibrate with just scalp EEG.
