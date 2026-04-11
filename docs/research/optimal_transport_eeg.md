# Optimal Transport for EEG Domain Adaptation

## Research Summary

Survey of optimal transport (OT) methods applied to cross-subject EEG domain adaptation (2023-2025), with focus on applicability to the scalp-to-in-ear prediction problem.

---

## 1. Core Methods

### 1.1 OT for Domain Adaptation (OTDA)

The fundamental idea: estimate a transport map T that pushes source-domain samples onto the target-domain distribution. For cross-subject EEG, each subject is a domain with distinct electrode impedances, skull geometry, and neural dynamics causing distribution shift.

**POT library** (`pip install POT`) provides scikit-learn-style `fit/transform` interface for:
- Sinkhorn transport (entropic regularization)
- Earth Mover's Distance (exact OT)
- Group-lasso and Laplacian regularized OT
- Semi-supervised OT (using label info)

Reference: [POT Documentation](https://pythonot.github.io/)

### 1.2 Sliced-Wasserstein on SPD Matrices (SPDSW)

**Bonet et al., ICML 2023** define a Sliced-Wasserstein distance for measures on the SPD manifold endowed with the Log-Euclidean metric. EEG covariance matrices are SPD by construction, so this provides a principled way to compute OT distances between subject-specific covariance distributions.

- Efficient surrogate to full Wasserstein on SPD manifolds
- Applied to brain-age prediction (MEG) and BCI domain adaptation
- Code: [github.com/clbonet/SPDSW](https://github.com/clbonet/SPDSW)

Reference: [arXiv:2303.05798](https://arxiv.org/abs/2303.05798)

### 1.3 Deep Optimal Transport on SPD Manifolds (DOT)

**Ju & Guan, 2022 (published in AIJ 2025)** combine geometric deep learning with OT on SPD manifolds. A SPDNet-based encoder learns nonlinear push-forward mappings that align source and target covariance distributions by minimizing a composite OT loss while respecting manifold geometry.

- Reduces both marginal and conditional distribution discrepancies
- Validated on 3 cross-session BCI datasets (KU, BNCI2014001, BNCI2015001)
- Consistently outperforms Euclidean alignment baselines
- Code: [github.com/GeometricBCI/Deep-Optimal-Transport-for-Domain-Adaptation-on-SPD-Manifolds](https://github.com/GeometricBCI/Deep-Optimal-Transport-for-Domain-Adaptation-on-SPD-Manifolds)

Reference: [arXiv:2201.05745](https://arxiv.org/abs/2201.05745)

---

## 2. EEG-Specific OT Applications

### 2.1 Multi-Source Feature Transfer Learning (MSTFL)

**Neurocomputing 2024.** Pipeline:
1. Align covariance matrices from different subjects in Riemannian space
2. Extract tangent space features via tangent space mapping
3. Use OT coupling matrix diagonal elements for feature ranking/selection
4. Transfer only the most transportable features across subjects

Key insight: not all features transport equally well. The OT coupling matrix reveals which feature dimensions have strong cross-subject correspondence.

Reference: [Neurocomputing 127944](https://www.sciencedirect.com/science/article/abs/pii/S092523122400715X)

### 2.2 OT + Frequency Mixup for Motor Imagery

**IEEE TNSRE 2022.** Maps EEG into latent space, minimizes Wasserstein distance between source and target, applies frequency-domain mixup augmentation. Joint distribution alignment (not just marginal).

Reference: [IEEE Xplore](https://ieeexplore.ieee.org/document/9910147/)

### 2.3 Wasserstein Domain Adaptation Network

**IEEE TNSRE 2023.** Adversarial domain adaptation using Wasserstein distance as the domain discriminator metric (instead of standard JS divergence). Evaluated on BCI Competition IV datasets 2a and 2b with improved cross-subject motor imagery classification.

Reference: [IEEE Xplore](https://ieeexplore.ieee.org/document/10035017/)

### 2.4 OT for EEG Sleep Stage Classification

**Gramfort et al., 2018.** Early application of OTDA to EEG, showing that optimal transport alignment of sleep EEG features significantly improves cross-subject sleep staging accuracy.

Reference: [HAL](https://hal.science/hal-01814190/document)

### 2.5 Wasserstein Barycenter Transport (WBT)

Multi-source domain adaptation: compute the Wasserstein barycenter of multiple source subjects, then transport all sources to this shared representation before training a unified model.

---

## 3. Regression-Specific: GOPSA (NeurIPS 2024)

**Most relevant to our problem.** Geodesic Optimization for Predictive Shift Adaptation addresses the regression case specifically (not classification).

**Key innovation:** When the output variable (not just features) also shifts across domains, standard OT/DA methods fail. GOPSA handles simultaneous shifts in both features and targets.

**Method:**
- Exploits geodesic structure of Riemannian manifold of SPD matrices
- Jointly learns domain-specific re-centering (geodesic intercept) and shared regression model
- Riemannian mixed-effects model: domain-specific intercept + shared slopes
- Evaluated on cross-site age prediction from resting-state EEG (HarMNqEEG, 14 sites, 1500+ subjects)

**Results:** Significantly higher R-squared, lower MAE, higher Spearman rho vs. baselines across multiple source-target site combinations.

**Code:** [github.com/apmellot/GOPSA](https://github.com/apmellot/GOPSA)

**Reference:** [arXiv:2407.03878](https://arxiv.org/abs/2407.03878), [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/93495)

---

## 4. Can OT Align EEG Features Across Subjects?

**Yes, with caveats:**

### Evidence FOR applicability:
- Cross-subject EEG distribution shift is well-documented and is the primary bottleneck (our iter013-036 confirm this)
- OT provides principled alignment that respects geometry (vs. heuristic Euclidean alignment which failed in our iter018)
- SPDSW and DOT specifically handle SPD covariance matrices, which is exactly how spatial EEG filters are parameterized
- GOPSA handles the regression case with output shift, directly relevant to our scalp-to-in-ear prediction
- Multiple papers report 5-15% accuracy improvements from OT alignment in classification settings

### Caveats for our problem:
- **Most OT-EEG work is classification, not regression.** Only GOPSA targets regression directly.
- **Our problem is channel prediction, not class prediction.** We predict 12 continuous in-ear channels, not discrete labels. The OT alignment would need to operate on spatial covariance structure.
- **Small dataset (12 training subjects).** OT methods need enough samples to estimate transport plans reliably.
- **Narrowband data (1-9 Hz at 20 Hz).** Limited spectral diversity may reduce the benefit of frequency-domain OT methods.
- **Our Euclidean alignment (iter018) already failed** with r=0.369 (worse than baseline). However, Euclidean alignment is a crude heuristic; proper OT alignment could perform differently.

### Recommended approach for this project:
1. **Start with POT's `SinkhornTransport`** on tangent-space features (simple, fast)
2. **Try GOPSA** if regression-specific alignment is needed (handles output shift)
3. **Avoid full DOT/SPDSW** unless simple methods show promise (too complex for 12 subjects)

---

## 5. Practical Implementation Plan

```python
# Minimal OT alignment using POT library
import ot
import numpy as np

def ot_align_subjects(X_source, X_target):
    """Align source subject features to target distribution using OT."""
    # Compute cost matrix (squared Euclidean)
    M = ot.dist(X_source, X_target, metric='sqeuclidean')
    M /= M.max()
    
    # Sinkhorn transport (entropic regularization)
    ot_model = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_model.fit(Xs=X_source, Xt=X_target)
    
    # Transport source samples to target domain
    X_transported = ot_model.transform(Xs=X_source)
    return X_transported
```

**Dependencies:** `pip install POT` (pure Python, numpy backend, ~50KB)

---

## 6. Key Papers Reference List

| Paper | Year | Venue | Method | Task | Relevance |
|-------|------|-------|--------|------|-----------|
| GOPSA (Mellot et al.) | 2024 | NeurIPS | Riemannian geodesic OT | EEG regression | HIGH - regression + cross-site |
| SPDSW (Bonet et al.) | 2023 | ICML | Sliced-Wasserstein on SPD | BCI DA | MEDIUM - principled SPD distance |
| DOT (Ju & Guan) | 2022/2025 | AIJ | Deep OT on SPD manifolds | BCI classification | MEDIUM - geometric deep learning |
| MSTFL | 2024 | Neurocomputing | OT feature ranking | EEG classification | MEDIUM - feature selection |
| OT+FreqMixup | 2022 | IEEE TNSRE | Wasserstein + mixup | Motor imagery | LOW - classification only |
| Wasserstein DA Net | 2023 | IEEE TNSRE | Adversarial Wasserstein | Motor imagery | LOW - classification only |
| OT Sleep Staging | 2018 | PRNI | OTDA | Sleep EEG | LOW - early work, classification |

---

## 7. Bottom Line for This Project

The 0.378 plateau in our LOSO benchmark is driven by cross-subject variability (Subject 14 ~0.27 vs Subject 13 ~0.46). OT-based alignment could help by:

1. **Aligning spatial covariance structure** across training subjects before fitting the FIR filter
2. **Learning a transport map** from each test subject's distribution to the training distribution (semi-supervised with unlabeled test data)
3. **GOPSA-style mixed-effects** approach: shared FIR filter + per-subject geodesic intercept

The strongest candidate is a simple Sinkhorn transport on tangent-space covariance features, combined with our existing FIR architecture. If that fails, GOPSA's regression-specific approach is the next step.
