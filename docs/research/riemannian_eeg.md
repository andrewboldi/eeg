# Riemannian Geometry for EEG Cross-Subject Transfer

## Summary

Riemannian geometry treats EEG covariance matrices as points on the manifold of
Symmetric Positive Definite (SPD) matrices, enabling principled alignment across
subjects. This is the mathematically correct framework for the cross-subject
variability problem that dominates our error budget.

**Key insight for our project**: Our iter018 (Euclidean alignment) and iter042
both used Euclidean whitening (R^{-1/2} applied to data). Riemannian alignment
goes further by operating on the SPD manifold with geodesic-aware operations,
which better preserves the geometric structure of EEG covariance.

---

## 1. Core Concepts

### SPD Manifold

EEG spatial covariance matrices are Symmetric Positive Definite (SPD). The space
of SPD matrices is NOT a vector space -- it is a Riemannian manifold (curved
space). Euclidean operations (mean, distance, interpolation) are geometrically
incorrect on this manifold.

**Three key metrics on the SPD manifold:**

| Metric | Formula for distance | Properties |
|--------|---------------------|------------|
| Affine-Invariant (AIRM) | d(A,B) = ||log(A^{-1/2} B A^{-1/2})||_F | Gold standard, congruence-invariant, expensive |
| Log-Euclidean | d(A,B) = ||log(A) - log(B)||_F | Fast, closed-form, good approximation |
| Euclidean | d(A,B) = ||A - B||_F | Ignores manifold structure, swelling effect |

### Geodesics and Frechet Mean

- **Geodesic**: Shortest path between two SPD matrices on the manifold (NOT a
  straight line in matrix space)
- **Frechet/geometric mean**: The point minimizing sum of squared geodesic
  distances to all samples. For N=2 matrices A, B: the geometric mean is
  A^{1/2} (A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}
- **Tangent space**: At any point on the manifold, there is a local Euclidean
  approximation (tangent space) where standard ML can be applied

### Tangent Space Mapping

Map SPD matrices to a Euclidean tangent space at a reference point (typically the
geometric mean), then apply standard linear methods (Ridge, SVM, etc.):

```
S_i -> log_M(S_i) = M^{1/2} log(M^{-1/2} S_i M^{-1/2}) M^{1/2}
```

where M is the reference point (geometric mean) and log is matrix logarithm.

---

## 2. Riemannian Alignment for Cross-Subject Transfer

### The Problem

Each subject has a different "center" on the SPD manifold due to:
- Different skull thickness / conductivity
- Different cortical folding
- Different electrode impedances
- Different neural source orientations

### Euclidean Alignment (EA) -- What We Tried

Reference: He & Wu 2020

```python
# Per-subject: compute mean covariance R_bar, then whiten
R_bar = mean(X_i @ X_i.T / T for all windows)
X_aligned = R_bar^{-1/2} @ X  # Whiten each subject's data
```

**What EA does**: Makes the marginal distribution of each subject's data have
identity covariance. Operates in Euclidean space.

**Our results**: iter018 used per-batch whitening (too noisy, r=0.369). iter042
used full-session covariance (proper EA). Both underperformed the FIR baseline
(r=0.378).

### Riemannian Alignment (RA) -- The Upgrade

Reference: Zanini et al. 2018, Rodrigues et al. 2019

```python
# Per-subject: compute GEOMETRIC mean M_s on SPD manifold
M_s = geometric_mean(Cov_1, Cov_2, ..., Cov_N)  # Iterative algorithm
# Re-center to identity using parallel transport
Cov_aligned = M_s^{-1/2} @ Cov_i @ M_s^{-1/2}
# For raw data: X_aligned = M_s^{-1/2} @ X
```

**What RA does**: Re-centers each subject's distribution to the Identity on the
SPD manifold using the geometric (not arithmetic) mean. This is the Riemannian
equivalent of z-scoring.

### Key Differences: RA vs EA

| Aspect | Euclidean Alignment | Riemannian Alignment |
|--------|-------------------|---------------------|
| Reference point | Arithmetic mean of covariances | Geometric (Frechet) mean on SPD manifold |
| Whitening | R_bar^{-1/2} @ X | M_geo^{-1/2} @ X |
| Geometric correctness | Ignores curvature | Respects manifold structure |
| Computational cost | O(C^3) per subject | O(C^3 * n_iter) per subject |
| Downstream classifier | Any | Any (but Riemannian classifiers benefit more) |
| Cross-session stability | Good | Better (congruence-invariant) |

**Practical difference**: For well-conditioned covariance matrices, EA and RA
give similar results. RA helps more when covariances are ill-conditioned or when
subjects are very different (large inter-subject variability).

### Riemannian Procrustes Analysis (RPA) -- Full Pipeline

Three alignment steps (Rodrigues et al. 2019):

1. **Re-centering**: Transport each subject's geometric mean to the Identity
2. **Stretching**: Equalize dispersion (variance of geodesic distances to center)
3. **Rotation**: Align principal directions using Procrustes in tangent space

This is the most complete Riemannian alignment and has shown the best cross-subject
transfer results in BCI classification.

---

## 3. Relevant Recent Papers (2024-2025)

### Riemannian Geometry-Based EEG Approaches: A Literature Review
- **arXiv**: 2407.20250 (July 2024)
- Comprehensive review of Riemannian methods in BCI
- Notes increasing integration of deep learning with Riemannian geometry
- Identifies cross-subject distribution shift as the key open problem
- Highlights that Riemannian methods are robust to volume conduction effects

### Cross-Subject and Cross-Montage EEG Transfer Learning (ITSA)
- **arXiv**: 2508.08216 (August 2025)
- Individual Tangent Space Alignment (ITSA): subject-specific recentering +
  distribution matching + supervised rotational alignment
- Fuses Regularized CSP with Riemannian geometry features
- Significant improvement in LOSO cross-validation
- Uses parallel fusion of spatial and Riemannian features

### Riemannian Transfer Learning Based on Log-Euclidean Metric
- **Frontiers in Neuroscience**, May 2024 (10.3389/fnins.2024.1381572)
- Extends Procrustes Analysis to Log-Euclidean metric (faster than AIRM)
- Shows log-Euclidean metric gives good approximation with lower compute cost
- Addresses inter-individual variability in BCI systems

### Harmonizing M/EEG Datasets with Covariance-Based Techniques
- **Imaging Neuroscience / MIT Press**, 2025 (PMC12007539)
- Directly applicable: uses covariance alignment for **regression** (not just classification)
- Pipeline: covariance estimation -> Riemannian alignment -> tangent space -> Ridge regression
- Uses coffeine library (builds on pyriemann) for regression pipelines
- Shows alignment improves cross-dataset prediction of continuous outcomes

### SPD Learning for Covariance-Based Neuroimaging Analysis
- **arXiv**: 2504.18882 (April 2025)
- SPDLearn: geometric deep learning library for SPD matrices
- Provides differentiable Riemannian layers for neural networks

### Source-Free Unsupervised Domain Adaptation on SPD Manifolds (SPDIM)
- **ICLR 2025** (arXiv: 2411.07249)
- Source-free adaptation: no access to source data at test time
- Particularly relevant for deploying models on new subjects without retraining

---

## 4. pyRiemann Library

### Overview

- **Repository**: https://github.com/pyRiemann/pyRiemann
- **License**: BSD-3
- **Documentation**: https://pyriemann.readthedocs.io/
- **Current version**: 0.8+ (actively maintained)
- Scikit-learn compatible API (fit/transform/predict)

### Key Components for Our Task

```python
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# Classification pipeline (standard example)
clf = make_pipeline(
    Covariances(estimator='lwf'),   # Ledoit-Wolf shrinkage
    TangentSpace(metric='riemann'),  # Project to tangent space
    LogisticRegression()
)

# REGRESSION pipeline (what we need)
reg = make_pipeline(
    Covariances(estimator='lwf'),
    TangentSpace(metric='riemann'),
    Ridge(alpha=1.0)
)
```

### Alignment Functions

```python
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import sqrtm, inv

def riemannian_align(X_subjects):
    """
    X_subjects: dict of {subj_id: array (n_windows, n_channels, n_times)}
    Returns: aligned data dict
    """
    aligned = {}
    for subj, X in X_subjects.items():
        # Compute per-window covariances
        covs = np.array([x @ x.T / x.shape[1] for x in X])
        # Geometric mean on SPD manifold
        M = mean_riemann(covs)
        # Whitening with geometric mean
        M_isqrt = inv(sqrtm(M)).real
        # Apply to raw data
        aligned[subj] = np.array([M_isqrt @ x for x in X])
    return aligned
```

### Can We Use It?

**Yes, but with caveats for our specific task:**

1. **Our task is regression, not classification**: pyriemann is primarily designed
   for classification. However, TangentSpace + Ridge regression works for
   continuous prediction.

2. **Our task is sample-level prediction**: We predict in-ear EEG at each time
   point, not epoch-level labels. We would need to:
   - Compute covariance per window (2s segments)
   - Use alignment as a preprocessing step before our FIR model
   - OR: use tangent space features + Ridge as a standalone approach

3. **Channel count**: We have 27 scalp channels -> 27x27 covariance matrices.
   This is manageable (not too large for matrix operations).

4. **Installation**: `pip install pyriemann` or `uv add pyriemann`

---

## 5. Applicability to Our Scalp-to-In-Ear Task

### What Could Work

1. **Riemannian alignment as preprocessing** (most promising):
   - Compute geometric mean covariance per subject
   - Re-center each subject to identity
   - Then train our existing FIR model on aligned data
   - This directly addresses the cross-subject variability bottleneck

2. **Augmented covariance approach**:
   - Build covariance matrices that include BOTH scalp and in-ear channels
   - [X_scalp; X_inear] -> 39x39 covariance matrix
   - Project to tangent space, then use Ridge to predict in-ear from tangent features
   - This captures cross-covariance structure between scalp and in-ear

3. **Geodesic interpolation for data augmentation**:
   - Interpolate between subjects' covariance matrices along geodesics
   - Generate synthetic "intermediate" subjects
   - Could help with our small N=12 training set

### What Probably Won't Work

1. **Pure Riemannian classification** (MDM, etc.) -- our task is regression
2. **Full RPA with rotation alignment** -- may overfit with only 12 training subjects
3. **Per-window covariance classification** -- too coarse for sample-level prediction

### Recommended Experiment Plan

**iter045: Riemannian Alignment + FIR**
- Use `mean_riemann()` from pyriemann to compute per-subject geometric mean
- Re-center all subjects to identity (proper Riemannian whitening)
- Train the proven FIR model (iter017/030) on the aligned data
- Compare against: iter042 (Euclidean alignment) and iter030 (no alignment)

**iter046: Tangent Space + Ridge Regression**
- Compute augmented covariances [scalp; inear] per window
- Project to tangent space at geometric mean
- Ridge regression from tangent features to in-ear channels
- This is the coffeine/Engemann et al. approach

**iter047: Geodesic Data Augmentation**
- Interpolate along geodesics between training subjects
- Generate K synthetic subjects per pair
- Train FIR on augmented dataset

---

## 6. Why Riemannian Might Beat Euclidean for Us

Our iter042 (Euclidean alignment) failed to improve over the FIR baseline. The
Riemannian approach may help because:

1. **Geometric mean vs arithmetic mean**: The arithmetic mean of covariance
   matrices suffers from "swelling effect" (determinant of mean > mean of
   determinants). The geometric mean avoids this, giving a more representative
   center point.

2. **Better conditioning**: Riemannian operations naturally handle
   ill-conditioned matrices better than Euclidean operations.

3. **The alignment is the Riemannian equivalent of z-scoring**: Just as z-scoring
   removes mean/variance differences in Euclidean space, Riemannian alignment
   removes "geometric mean" differences on the SPD manifold.

4. **Our main bottleneck is cross-subject variability**: Subject 14 consistently
   at r=0.27 while Subject 13 at r=0.46. Proper alignment could reduce this gap.

However, the practical difference between EA and RA may be small for
well-conditioned 27x27 covariance matrices. The real win may come from combining
alignment with the augmented covariance / tangent space regression approach.

---

## Sources

- [Riemannian Geometry-Based EEG Approaches: A Literature Review (2024)](https://arxiv.org/abs/2407.20250)
- [Cross-Subject and Cross-Montage EEG Transfer Learning via ITSA (2025)](https://arxiv.org/abs/2508.08216)
- [Riemannian Transfer Learning Based on Log-Euclidean Metric (2024)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1381572/full)
- [Harmonizing M/EEG Datasets with Covariance-Based Techniques (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007539/)
- [Revisiting Euclidean Alignment for Transfer Learning in EEG-Based BCI (2025)](https://arxiv.org/abs/2502.09203)
- [A Systematic Evaluation of Euclidean Alignment with Deep Learning (2024)](https://arxiv.org/abs/2401.10746)
- [SPD Learning for Covariance-Based Neuroimaging (2025)](https://arxiv.org/abs/2504.18882)
- [SPDIM: Source-Free Unsupervised DA on SPD Manifolds, ICLR 2025](https://arxiv.org/abs/2411.07249)
- [Transfer Learning: A Riemannian Geometry Framework (Zanini et al. 2018)](https://ieeexplore.ieee.org/iel7/10/4359967/08013808.pdf)
- [pyRiemann GitHub](https://github.com/pyRiemann/pyRiemann)
- [pyRiemann Documentation](https://pyriemann.readthedocs.io/)
- [coffeine: Covariance Data Frames for Predictive M/EEG Pipelines](https://github.com/coffeine-labs/coffeine)
- [Predictive Regression Modeling with MEG/EEG (Engemann et al. 2020)](https://www.biorxiv.org/content/10.1101/845016v1.full)
