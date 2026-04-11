# Data-Centric AI Strategies for EEG Scalp-to-In-Ear Prediction

## Context

Our bottleneck is **data, not model capacity**. We have only 12 training subjects (3 held out for test), and the 0.378 Pearson r plateau persists across 30+ model iterations. Cross-subject variability dominates: Subject 14 consistently ~0.27 r while Subject 13 reaches ~0.46 r. This document surveys data-centric techniques that could break through this ceiling.

---

## 1. Active Learning: Which New Subjects to Record Next?

### Core Idea
Rather than recording arbitrary new subjects, use model uncertainty to select the *most informative* next subjects to add to the training set.

### Relevant Methods

**Uncertainty Sampling** (Lewis & Gale, 1994; Settles, 2009):
- Query subjects whose data produces maximal prediction uncertainty
- For our regression task: subjects where the model's predicted in-ear EEG has highest variance across bootstrap/ensemble models
- Implementation: train K models on bootstrap samples, measure prediction disagreement on candidate subjects

**Double-Criteria Active Learning for BCI** ([Rezeika et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7091553/)):
- Combines uncertainty and representativeness criteria for EEG
- BvSB (Best vs Second Best) strategy to measure informativeness
- Cosine distance to measure diversity from existing training set

**Enhanced Uncertainty with Category Information** ([PLOS ONE, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12233261/)):
- Incorporates cluster structure into uncertainty estimates
- Could group subjects by scalp topology similarity before selecting diverse representatives

### Practical Application to Our Problem
Since we can't easily record new subjects (dataset is fixed at 15), active learning is most useful for:
1. **Deciding which of the 12 training subjects to weight more heavily** during training
2. **If the dataset expands**: prioritizing subjects with scalp topographies dissimilar to existing training set
3. **Transfer learning**: selecting which public EEG datasets to incorporate based on domain similarity

### Actionable Experiment
- Train ensemble of 5 FIR models on different bootstrap samples of training subjects
- Measure prediction disagreement per test subject
- Use disagreement to identify which *training subject types* are underrepresented

---

## 2. Data Cleaning: Identifying and Removing Bad Training Windows

### Core Idea
Not all 2-second training windows are equally useful. Some contain artifacts, poor signal, or atypical patterns that hurt generalization. Removing the worst windows could improve model quality.

### Relevant Methods

**Autoreject** ([Jas et al., 2017](https://www.sciencedirect.com/science/article/abs/pii/S1053811917305013)):
- Unsupervised algorithm for MEG/EEG artifact rejection
- Learns per-channel amplitude thresholds that minimize cross-validation error
- Removes epochs whose peak-to-peak amplitude exceeds learned thresholds
- Well-established in the EEG community, available in MNE-Python

**EEGEpochNet** ([PubMed, 2025](https://pubmed.ncbi.nlm.nih.gov/41730244/)):
- Self-supervised contrastive learning for automated EEG epoch rejection
- Multi-branch 1D-CNN with U-Net features + bidirectional GRU
- Does NOT require manually labeled artifacts
- State-of-the-art for automated bad epoch detection

**Confident Learning / Cleanlab** ([Northcutt et al., 2021](https://arxiv.org/pdf/1911.00068)):
- Originally for classification, but cleanlab now supports regression
- Identifies mislabeled or noisy examples using model confidence
- No hyperparameters needed; provable guarantees on noise estimation
- Could flag training windows where the target in-ear EEG is corrupted

**Deep Autoencoder Artifact Detection** ([arXiv 2502.08686](https://arxiv.org/html/2502.08686v1)):
- End-to-end pipeline: unsupervised detection in 58D feature space
- Rejected trials train a deep encoder-decoder to correct remaining artifacts
- Two-stage approach: detect then correct

### Practical Application to Our Problem

**Window-level quality scoring:**
1. Train the FIR model on all data
2. Compute per-window residuals (predicted vs actual in-ear EEG)
3. Flag windows with abnormally high residuals as potentially artifactual
4. Retrain excluding the worst N% of windows
5. Sweep N from 5% to 30% and measure impact on held-out subjects

**Channel-level quality scoring:**
- Some in-ear channels (ELC, ERT) have frequent artifacts
- Score each (window, channel) pair by reconstruction error
- Mask or downweight bad (window, channel) pairs during training

**Statistical thresholds:**
- Peak-to-peak amplitude > 3 sigma from mean
- Kurtosis or skewness outliers per window
- Flat-line detection (near-zero variance windows, likely 100% interpolated)

### Actionable Experiment (iter045 candidate)
```python
# Pseudocode for window quality scoring
residuals = model.predict(X_train) - Y_train  # per-window, per-channel
window_scores = np.mean(np.abs(residuals), axis=(1, 2))  # mean abs error per window
threshold = np.percentile(window_scores, 85)  # keep best 85%
clean_mask = window_scores < threshold
model_clean = train(X_train[clean_mask], Y_train[clean_mask])
```

---

## 3. Curriculum Learning: Training on Easy Subjects First

### Core Idea
Present training examples in order of difficulty, starting with "easy" subjects (those with clear scalp-to-in-ear mapping) and gradually introducing harder subjects.

### Relevant Methods

**Classical Curriculum Learning** ([Bengio et al., 2009](https://en.wikipedia.org/wiki/Curriculum_learning)):
- Order training samples from easy to hard
- "Easy" = low loss under a simple model (e.g., closed-form baseline)
- Converges faster and sometimes to better local optima
- Most benefit observed in non-convex optimization (our FIR model is convex, limiting upside)

**Self-Paced Learning** (Kumar et al., 2010):
- Model dynamically selects which examples to train on based on current loss
- Automatically increases difficulty as training progresses
- Avoids fixed curriculum that may become stale

**Self-Paced Curriculum Learning** (combination):
- Start with a fixed ordering, then let the model re-rank during training
- Prevents over-fitting to easy examples while maintaining curriculum structure

**Learning Rate Curriculum (LeRaC)** ([Springer, 2024](https://link.springer.com/article/10.1007/s11263-024-02186-5)):
- Data-agnostic: uses different learning rates per layer instead of data ordering
- Higher learning rates for early layers initially, then equalize
- Could complement data-level curriculum

**Anti-Curriculum** (Braun et al., 2017):
- Train on *hard* examples first
- Rationale: hard examples force the model to learn more general features
- Mixed results in literature; tends to help with noisy data

### Practical Application to Our Problem

**Subject-level difficulty scoring:**
- Easy subjects: high per-subject r under closed-form baseline (Subject 13: ~0.46)
- Hard subjects: low per-subject r (Subject 14-like training subjects: ~0.27)
- Could also score by scalp signal SNR or in-ear channel quality

**Window-level difficulty scoring:**
- Easy windows: low closed-form residual
- Hard windows: high residual (could be noisy OR genuinely complex)

**Implementation for our FIR model:**
1. Compute per-subject r using closed-form baseline on training data
2. Sort subjects by r (easy first)
3. Train first 50 epochs on top-6 easiest subjects
4. Add next 3 subjects for epochs 50-100
5. Add remaining 3 for epochs 100-150

### Risks
- Our model is essentially linear (FIR), so curriculum benefits are limited (curriculum helps most with non-convex optimization)
- With only 12 training subjects, reducing early training set size increases variance
- May work better with deeper models where local optima matter

### Actionable Experiment (iter046 candidate)
```python
# Subject-level curriculum
subject_difficulty = {s: closed_form_r[s] for s in train_subjects}
sorted_subjects = sorted(subject_difficulty, key=lambda s: -subject_difficulty[s])
# Phase 1: easy subjects only
# Phase 2: add medium subjects
# Phase 3: all subjects
```

---

## 4. Data Selection: Which Training Subjects Are Most Informative?

### Core Idea
Not all 12 training subjects contribute equally to test performance. Some may actively *hurt* generalization. Identify and possibly remove or downweight harmful subjects.

### Relevant Methods

**Data Shapley** ([Ghorbani & Zou, 2019](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf)):
- Assigns each training datum a Shapley value measuring its marginal contribution
- Provably fair: satisfies symmetry, null player, and additivity axioms
- Captures complex interactions between data subsets (unlike leave-one-out)
- Expensive: requires training on many subsets; feasible for 12 subjects (2^12 = 4096 subsets)

**LossVal** ([arXiv 2412.04158, 2024](https://arxiv.org/html/2412.04158v2)):
- Efficient data valuation using loss-based utility
- Works for both regression and classification
- Successfully identifies beneficial and detrimental data points
- Lower computational cost than full Shapley

**Leave-One-Subject-Out (LOSO) Valuation:**
- Simplest: train with and without each subject, measure test r difference
- Limitation: doesn't capture interactions (removing subject A might only help if subject B is also removed)
- But with 12 subjects, pairwise interactions are tractable (12*11/2 = 66 pairs)

**Influence Functions** ([Koh & Liang, 2017](https://link.springer.com/article/10.1007/s10994-023-06495-7)):
- Approximate leave-one-out effect using gradients
- Computationally efficient: single pass through training data
- Recent work (DataInf, 2024) extends to modern architectures
- LESS ([arXiv 2402.04333](https://arxiv.org/abs/2402.04333)): training on influence-selected 5% of data can outperform full dataset

**Multisource Transfer Learning for EEG** ([ResearchGate](https://www.researchgate.net/publication/332056102_Multisource_Transfer_Learning_for_Cross-Subject_EEG_Emotion_Recognition)):
- Weights source subjects by domain similarity to target subject
- Uses MMD (Maximum Mean Discrepancy) or KL divergence to measure domain gap
- Subjects similar to the test subject get higher weight

### Practical Application to Our Problem

**Subject Shapley valuation (tractable!):**
With only 12 training subjects, we can exhaustively compute exact Shapley values:
1. For each subset S of training subjects (sample ~500 subsets):
   - Train FIR model on subjects in S
   - Evaluate mean r on validation set (or test subjects)
2. Compute Shapley value for each subject
3. Identify subjects with negative Shapley values (harmful to generalization)
4. Retrain excluding harmful subjects

**LOSO subject valuation (quick experiment):**
1. For each training subject i:
   - Train on all 12 subjects -> r_all
   - Train on 11 subjects (drop i) -> r_minus_i
   - Subject i's value = r_all - r_minus_i
2. If value is negative, subject i is hurting generalization

**Per-test-subject analysis:**
- Subject 14 is always hard. Which training subjects help Subject 14 most?
- May find that certain training subjects are specifically harmful to specific test subjects
- Could lead to test-time subject-adaptive weighting

### Actionable Experiment (iter047 candidate)
```python
# LOSO subject valuation
subject_values = {}
r_all = train_and_eval(all_train_subjects)
for s in train_subjects:
    remaining = [x for x in train_subjects if x != s]
    r_without = train_and_eval(remaining)
    subject_values[s] = r_all - r_without
    print(f"Subject {s}: value = {subject_values[s]:+.4f}")

# Remove subjects with negative value
good_subjects = [s for s, v in subject_values.items() if v > 0]
r_clean = train_and_eval(good_subjects)
```

---

## 5. Cross-Cutting: Meta-Learning and Few-Shot Adaptation

### Relevant Methods for Small EEG Datasets

**META-EEG** (gradient-based meta-learning):
- Intermittent freezing strategy for cross-subject EEG
- Outperforms baselines on LOSO cross-validation
- Could be combined with data selection

**Subject-Adaptive Transfer Learning** ([MICCAI 2024](https://papers.miccai.org/miccai-2024/740-Paper0192.html)):
- Uses resting-state EEG to calibrate for new subjects
- Dual attention relation network for few-shot adaptation
- Disentangles task-dependent and subject-dependent features

**Few-Shot Learning Survey for EEG** ([Frontiers, 2024](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1421922/full)):
- Comprehensive taxonomy: data augmentation, transfer learning, self-supervised learning
- Addresses inter/intra-subject variability specifically
- Recommends combining data augmentation with meta-learning for best results

---

## 6. Prioritized Action Plan

Based on expected impact and implementation effort:

| Priority | Technique | Expected Impact | Effort | Iteration |
|----------|-----------|----------------|--------|-----------|
| **1** | Window quality scoring + pruning | Medium-High | Low | iter045 |
| **2** | LOSO subject valuation | Medium | Low | iter046 |
| **3** | Subject-weighted training | Medium | Low | iter047 |
| **4** | Per-subject difficulty curriculum | Low-Medium | Medium | iter048 |
| **5** | Full Data Shapley valuation | Medium | Medium | iter049 |
| **6** | Influence function analysis | Low-Medium | High | Later |

### Rationale for Prioritization

1. **Window pruning first**: Cheapest intervention. If 10-15% of windows are artifactual, removing them directly improves training signal. No architecture change needed.
2. **Subject valuation next**: With only 12 subjects, LOSO valuation takes 12 extra training runs (~12 minutes). Immediately tells us if any subject is harmful.
3. **Subject weighting**: Instead of binary remove/keep, weight subjects by their value. Smooth version of subject selection.
4. **Curriculum**: Limited benefit expected for linear FIR model, but worth testing if we move to nonlinear models.
5. **Full Shapley**: More principled than LOSO but much more expensive. Do after LOSO confirms subject value varies.

---

## Key References

- [Data-Centric AI to improve performance (Nature Scientific Reports, 2024)](https://www.nature.com/articles/s41598-024-73643-x)
- [Data-Centric AI Survey (ACM Computing Surveys, 2024)](https://dl.acm.org/doi/10.1145/3711118)
- [van der Schaar Lab: Data-Centric AI](https://www.vanderschaar-lab.com/data-centric-ai/)
- [Autoreject for MEG/EEG (Jas et al., 2017)](https://www.sciencedirect.com/science/article/abs/pii/S1053811917305013)
- [Confident Learning / Cleanlab](https://github.com/cleanlab/cleanlab)
- [EEGEpochNet: Self-supervised epoch rejection](https://pubmed.ncbi.nlm.nih.gov/41730244/)
- [Deep Autoencoder EEG Artifact Detection (2025)](https://arxiv.org/html/2502.08686v1)
- [Data Shapley (Ghorbani & Zou, 2019)](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf)
- [LossVal: Efficient Data Valuation (2024)](https://arxiv.org/html/2412.04158v2)
- [Training Data Influence Analysis Survey](https://link.springer.com/article/10.1007/s10994-023-06495-7)
- [LESS: Influential Data Selection (2024)](https://arxiv.org/abs/2402.04333)
- [Few-Shot Learning for EEG Survey (Frontiers, 2024)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1421922/full)
- [Subject-Adaptive Transfer Learning (MICCAI 2024)](https://papers.miccai.org/miccai-2024/740-Paper0192.html)
- [Learning Rate Curriculum (IJCV, 2024)](https://link.springer.com/article/10.1007/s11263-024-02186-5)
- [Double-Criteria Active Learning for BCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC7091553/)
- [MNE: Rejecting bad data](https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html)
