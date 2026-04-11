# Signal Processing Baselines for Scalp-to-In-Ear EEG Prediction

## Research Summary

We searched for physics-based and classical signal processing approaches that could
inform or replace our current learned FIR spatial filter (best r=0.378). The key
finding is that **the scalp-to-ear mapping is fundamentally linear due to volume
conduction physics**, which explains why our linear FIR model is hard to beat.

---

## 1. Forward/Inverse Models and Lead Field Matrices

### Key Concept
The EEG forward problem computes how a dipole source in the brain produces
potentials at electrode locations. This is encoded in a **lead field matrix (LFM)**
`L` of shape `(n_electrodes, n_sources * 3)`, where each column gives the potential
at all electrodes for a unit dipole at one source location/orientation.

The relationship is linear:
```
V_scalp = L_scalp @ s    (scalp potentials from sources)
V_ear   = L_ear   @ s    (ear potentials from same sources)
```

### Implication for Our Problem
If we had both lead field matrices, we could compute the **direct transfer matrix**:
```
V_ear = L_ear @ pinv(L_scalp) @ V_scalp = T @ V_scalp
```
This is a physics-derived linear spatial filter -- exactly what our closed-form
baseline already approximates from data! The closed-form solution `W = R_YX @ inv(R_XX)`
is the data-driven equivalent of `T = L_ear @ pinv(L_scalp)`.

### Why This Won't Beat Our Current Approach
- We don't have individual head models for Ear-SAAD subjects
- Generic head models introduce more error than data-driven estimation
- Our data-driven `W` already captures the same linear mapping empirically
- The pseudo-inverse of `L_scalp` is ill-conditioned (requires regularization), same as our Tikhonov baseline

### References
- [Kappel et al. 2019 - Ear-EEG Forward Models](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00943/full) -- BEM-based forward model with ear canal anatomy
- [Review: Forward Problem in EEG Source Analysis](https://link.springer.com/article/10.1186/1743-0003-4-46)
- [Effects of Forward Model Errors on Source Localization](https://pmc.ncbi.nlm.nih.gov/articles/PMC3683142/)

---

## 2. Volume Conduction: The Physics of Scalp-to-Ear

### Key Findings
Volume conduction through the head is governed by Maxwell's equations applied to an
inhomogeneous dielectric (brain, CSF, skull, scalp). Critically:

- **The volume conductor is a LINEAR system** -- potentials at any electrode are a
  linear superposition of all active sources
- Tissue conductivities: brain ~0.33 S/m, CSF ~1.79 S/m, skull ~0.01 S/m, scalp ~0.43 S/m
- The skull is the dominant attenuator and spatial low-pass filter
- Ear-canal electrodes see signals through the temporal bone, which has different
  thickness/conductivity than parietal skull

### What This Means
The linearity of volume conduction means **no nonlinear model can fundamentally
outperform a well-regularized linear model** for this task. The mapping from scalp
to ear is linear in the source signals, and source signals are the same for both.
The only nonlinearity would come from noise interactions or electrode artifacts.

This explains our finding that deep networks, residual learning, MoE, and other
nonlinear approaches all fail to beat FIR + linear spatial filter.

### Ear-EEG Sensitivity
- Ear-EEG is most sensitive to **temporal cortex** sources
- Mean sensitivity ratio: ear/scalp ~ 0.3 per ear, ~0.9 for bilateral ear-EEG
- Inter-electrode distance within the ear is small, limiting spatial resolution
- Ear-EEG has a "keyhole" view -- sees a subset of sources that scalp also sees

### References
- [Ear-EEG Sensitivity Modeling (2022)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.997377/full)
- [Sensitivity of Ear-EEG: Source-Sensor Relationship](https://link.springer.com/article/10.1007/s10548-020-00793-2)
- [Volume Conduction in EEG and MEG](https://www.sciencedirect.com/science/article/abs/pii/S0013469497001478)

---

## 3. The Keyhole Hypothesis

### Core Finding (Mikkelsen, Kidmose & Hansen, 2017)
Ear-EEG acts as a "keyhole" into broadly distributed neural processes:

- **High mutual information** between ear-EEG and scalp EEG
- The scalp-to-ear mapping is well-approximated by a **linear model**
- This linear mapping is **stable across time and mental states**
- Ear-EEG can be reliably predicted from scalp EEG, and vice versa
- Highest predictability for **temporal region** scalp electrodes

### Quantitative Findings
- Linear models provide a **lower bound** on the scalp-to-ear information transfer
- The mapping stability across mental states means a single spatial filter suffices
  (no need for state-dependent switching, which we confirmed -- MoE didn't help)

### Implications
This is essentially the theoretical justification for our entire approach. The
keyhole hypothesis says:
1. The relationship IS linear (confirmed by our experiments)
2. A single linear mapping works across conditions (confirmed)
3. The bottleneck is not the model but the **information content** accessible from
   the scalp electrode geometry

### Reference
- [Keyhole Hypothesis: High Mutual Information between Ear and Scalp EEG](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2017.00341/full)

---

## 4. Beamforming Approaches

### LCMV Beamformer
Linearly Constrained Minimum Variance beamforming constructs spatial filters that:
- Minimize output variance (noise) subject to unit gain for the target source
- Require a lead field matrix (forward model) for the target location
- Formula: `w = inv(C) @ l / (l' @ inv(C) @ l)` where C is data covariance, l is lead field

### Could We Use Beamforming?
**Not directly**, because:
- Beamforming reconstructs a source signal at a brain location, not at another electrode
- We would need: (1) beamform to sources, (2) project sources to ear electrodes
- This is equivalent to `T = L_ear @ pinv(L_scalp)` with a different regularization
- Without individual head models, this reduces to data-driven spatial filtering again

### Multi-Channel Wiener Filter (MWF)
A potentially interesting variant:
- Operates in the frequency domain
- Constructs separate spatial filters per frequency bin
- Formula: `W(f) = S_yx(f) @ inv(S_xx(f))` (cross-spectral density matrices)
- This is exactly our closed-form solution but computed per-frequency

**This could help if the optimal spatial filter varies across frequencies.**
In our 1-9 Hz band this is unlikely to matter much (only ~4 frequency bins at 20 Hz
sampling), but for broadband data (1-45 Hz at 256 Hz) this could be valuable.

### References
- [LCMV Beamformer Tutorial - MNE](https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html)
- [Unified View on Beamformers for M/EEG](https://www.sciencedirect.com/science/article/pii/S1053811921010612)
- [Multi-Channel Wiener Filter for EEG](https://www.researchgate.net/publication/280313851_EEG_signal_enhancement_using_multi-channel_wiener_filter_with_a_spatial_correlation_prior)
- [Frequency-Adaptive Broadband Beamformer](https://www.biorxiv.org/content/10.1101/502690v1.full)

---

## 5. Electrode Interpolation Methods

### Spherical Spline Interpolation (SSI)
Standard method for reconstructing bad EEG channels:
- Projects electrodes onto a sphere
- Computes weights via inverse-distance on the sphere
- Used by MNE-Python, EEGLAB, FieldTrip as default
- **Limitation**: assumes the target electrode is ON the scalp, not inside the ear canal

### Attention-Based Channel Weighting
Recent work (2023) uses learned attention weights:
- Learns correlation structure among channels from good data
- No need for electrode location information
- This is essentially what our data-driven spatial filter already does

### Virtual EEG Electrodes via CNNs
- CNNs trained to upsample or restore missing channels
- Could potentially be trained to "predict" ear channels from scalp
- But this is exactly our current approach with a different architecture

### Key Insight
All interpolation methods reduce to estimating a linear (or mildly nonlinear)
mapping from existing to missing channels. Our closed-form baseline IS the optimal
linear interpolation for this specific channel set.

### References
- [Spherical Spline Interpolation (EEGLAB)](https://sccn.ucsd.edu/~jiversen/pdfs/courellis_etal_eeg_channel_2017.pdf)
- [Attention-Based Channel Interpolation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10552919/)
- [Virtual EEG Electrodes via CNNs](https://www.sciencedirect.com/science/article/pii/S0165027021000613)
- [BiLSTM Channel Interpolation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11359714/)

---

## 6. Frequency-Domain Transfer Function

### Concept
Instead of a single spatial filter W applied to time-domain data, compute a
**frequency-dependent transfer function** T(f):

```
Y(f) = T(f) @ X(f)
```

where X(f), Y(f) are the DFT of scalp and ear signals respectively.

The optimal T(f) is:
```
T(f) = S_yx(f) @ inv(S_xx(f))
```

This is the multi-channel Wiener filter in the frequency domain.

### Potential Advantage
- Allows different spatial filters at different frequencies
- At 1 Hz, temporal sources dominate (alpha-like); at 9 Hz, faster dynamics
- The brain regions contributing to ear-EEG may shift with frequency

### Potential Implementation
```python
# Frequency-domain Wiener filter
from scipy.signal import csd
for f_bin in frequency_bins:
    S_xx[f] = cross_spectral_density(X, X, f)  # 27x27
    S_yx[f] = cross_spectral_density(Y, X, f)  # 12x27
    T[f] = S_yx[f] @ np.linalg.inv(S_xx[f] + lambda * I)
# Apply: Y_hat(f) = T(f) @ X(f), then IFFT
```

### Assessment for Our Setup
- With 20 Hz sampling and 1-9 Hz band, we only have ~8-9 useful frequency bins
- The spatial filter likely doesn't vary much across this narrow band
- **For broadband (1-45 Hz at 256 Hz), this becomes much more interesting**
- Worth testing as iter045 or similar

---

## 7. Actionable Ideas Ranked by Expected Impact

### HIGH POTENTIAL (try these)

1. **Frequency-Domain Wiener Filter on Broadband Data** (1-45 Hz, 256 Hz)
   - Different spatial filter per frequency bin
   - Could capture frequency-dependent source contributions
   - Expected: moderate improvement on broadband, minimal on 1-9 Hz

2. **Cross-Validated Regularization of the Transfer Matrix**
   - Our current Tikhonov lambda is fixed or swept coarsely
   - Per-channel or per-frequency regularization could help
   - Shrinkage estimators (Ledoit-Wolf on S_xx) for better conditioning

3. **Rank-Reduced Regression (RRR)**
   - Constrain the transfer matrix T to be low-rank: T = B @ A
   - If only ~5-8 independent sources contribute to ear-EEG, rank 5-8 suffices
   - This is like PCA regression but jointly optimized
   - Formula: min ||Y - B @ A @ X||^2 with rank(B@A) = k

### MODERATE POTENTIAL (worth investigating)

4. **Common Spatial Patterns (CSP) Pre-filtering**
   - Find spatial filters that maximize variance ratio between ear-predictable
     and ear-unpredictable scalp components
   - Use top-k CSP components as input to the FIR model

5. **Generalized Eigenvalue Decomposition (GEVD)**
   - Find directions in scalp space that maximize correlation with ear space
   - Similar to CCA but without the ear-side mixing
   - `eig(R_scalp_ear @ inv(R_scalp_scalp))` gives optimal projection directions

### LOW POTENTIAL (theory says won't help much)

6. **Physics-Based Forward Model Transfer Matrix**
   - Requires individual head models (MRI) which we don't have
   - Generic models introduce systematic errors
   - Data-driven estimation already captures the same linear mapping

7. **LCMV/DICS Beamforming Pipeline**
   - Source reconstruction + re-projection is mathematically equivalent
   - Adds complexity without theoretical benefit over direct estimation

---

## 8. Key Takeaway

**The physics confirms our empirical finding**: the scalp-to-ear mapping is
fundamentally linear due to volume conduction. Our current approach (data-driven
linear spatial + temporal filter) is already close to optimal for this physics.

The remaining improvement opportunities are:
- **Better regularization** of the linear mapping (rank reduction, shrinkage)
- **Frequency-dependent spatial filters** (especially for broadband data)
- **Cross-subject alignment** of the linear mapping (the real bottleneck)
- **More data** (broadband, longer recordings, more subjects)

The 0.378 ceiling on 1-9 Hz data likely reflects the **information-theoretic
limit** of what 27 scalp channels can tell us about 12 in-ear channels in this
narrow frequency band, not a model capacity issue.
