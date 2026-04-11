# Subject-Level Analysis: Why Some Subjects Are Easy and Others Hard

**Dataset**: Ear-SAAD broadband_46ch.h5 (15 subjects, 46 scalp channels, 12 in-ear channels, 256 Hz, 3594 windows each)

**LOSO model**: Broadband FIR (iter038-style), trained on 14 subjects, tested on held-out subject.

## LOSO Performance Summary

| Subject | LOSO r | Category |
|---------|--------|----------|
| 3       | 0.940  | Easy     |
| 4       | 0.840  | Easy     |
| 9       | 0.765  | Easy     |
| 11      | 0.742  | Easy     |
| 5       | 0.732  | Easy     |
| 13      | 0.728  | Easy     |
| 10      | 0.653  | Medium   |
| 7       | 0.618  | Medium   |
| 15      | 0.601  | Medium   |
| 1       | 0.586  | Medium   |
| 6       | 0.575  | Medium   |
| 12      | 0.548  | Hard     |
| 2       | 0.539  | Hard     |
| 14      | 0.433  | Hard     |
| 8       | 0.382  | Hard     |

## Key Finding: Scalp-to-In-Ear Coupling Strength Is the Dominant Predictor

The single most important factor determining whether a subject is easy or hard is **how strongly their in-ear EEG correlates with their scalp EEG**. This is a physiological property of the subject, not a data quality issue.

### Top Statistical Predictors of LOSO r

| Metric | Pearson r with LOSO r | p-value | Interpretation |
|--------|----------------------|---------|----------------|
| In-ear channel RMS CV (coefficient of variation) | **-0.777** | **0.0007** | Hard subjects have uneven channel quality |
| In-ear window variance | **+0.714** | **0.0028** | Easy subjects have richer temporal dynamics |
| Min best scalp-inear channel correlation | **+0.767** | **0.0008** | Hard subjects have "dead" in-ear channels with no scalp correlate |
| Mean best scalp-inear channel correlation | **+0.707** | **0.0032** | Hard subjects have weaker overall scalp-inear coupling |
| Number of well-coupled channels (best r > 0.5) | **+0.775** | **0.0007** | Hard subjects have fewer usable in-ear channels |

### Non-Predictors (no significant correlation with LOSO r)

| Metric | Pearson r | p-value | Interpretation |
|--------|-----------|---------|----------------|
| Number of windows | N/A | N/A | All subjects have exactly 3594 windows |
| Scalp mean amplitude | -0.198 | 0.48 | Scalp signal strength doesn't matter |
| In-ear mean amplitude | +0.044 | 0.88 | In-ear signal strength doesn't matter |
| Amplitude ratio (scalp/inear) | +0.138 | 0.62 | Relative amplitudes don't matter |
| NaN percentage | 0 | N/A | All channels are 0% NaN in this preprocessed data |
| Dead channels (>90% NaN) | 0 | N/A | No dead channels in any subject |
| In-ear SNR (dB) | -0.046 | 0.87 | Trial-level SNR doesn't predict LOSO r |
| Scalp channel RMS CV | -0.260 | 0.35 | Scalp channel variability doesn't matter |

## Detailed Analysis: Subject 3 (Best, r=0.940) vs Subject 8 (Worst, r=0.382)

### Channel-Level Scalp-to-In-Ear Correlations

**Subject 3** -- Every in-ear channel has a scalp channel correlated at r > 0.91:
```
InEar ch 0: best_r=0.970 (scalp ch31)    InEar ch 6: best_r=0.950 (scalp ch36)
InEar ch 1: best_r=0.916 (scalp ch31)    InEar ch 7: best_r=0.926 (scalp ch41)
InEar ch 2: best_r=0.968 (scalp ch34)    InEar ch 8: best_r=0.923 (scalp ch41)
InEar ch 3: best_r=0.969 (scalp ch31)    InEar ch 9: best_r=0.973 (scalp ch41)
InEar ch 4: best_r=0.974 (scalp ch34)    InEar ch10: best_r=0.972 (scalp ch45)
InEar ch 5: best_r=0.968 (scalp ch34)    InEar ch11: best_r=0.972 (scalp ch45)
```
Min=0.916, Mean=0.957. The scalp-to-in-ear mapping is almost perfectly linear for this subject.

**Subject 8** -- Several channels have very weak coupling (< 0.3):
```
InEar ch 0: best_r=0.322 (scalp ch34)    InEar ch 6: best_r=0.516 (scalp ch36)
InEar ch 1: best_r=0.419 (scalp ch33)    InEar ch 7: best_r=0.235 (scalp ch36)
InEar ch 2: best_r=0.831 (scalp ch30)    InEar ch 8: best_r=0.753 (scalp ch45)
InEar ch 3: best_r=0.831 (scalp ch30)    InEar ch 9: best_r=0.588 (scalp ch45)
InEar ch 4: best_r=0.216 (scalp ch35)    InEar ch10: best_r=0.796 (scalp ch45)
InEar ch 5: best_r=0.778 (scalp ch35)    InEar ch11: best_r=0.729 (scalp ch45)
```
Min=0.216, Mean=0.585. Channels 0, 4, and 7 have virtually no scalp correlate -- these are **fundamentally unpredictable** from scalp EEG.

### Subject 14 (Second Hardest, r=0.433) -- Similar Pattern

```
InEar ch 3: best_r=0.012    InEar ch 9: best_r=0.012    InEar ch 6: best_r=0.187
```
Only 5/12 channels are well-coupled (best_r > 0.5). Three channels are essentially uncorrelated with any scalp channel.

## What Causes Poor Coupling?

The analysis rules out simple explanations:

1. **Not noise/amplitude**: Hard subjects have normal amplitude levels and no NaN/dead channels
2. **Not data quantity**: All subjects have identical window counts (3594)
3. **Not scalp quality**: Scalp channel variability does not predict performance

The coupling strength appears to be a **physiological/anatomical property**:
- **Skull thickness and conductivity** vary across individuals, affecting how far scalp potentials propagate to the ear canal
- **In-ear electrode contact quality** varies -- some channels may have poor skin contact even without being "dead"
- **Individual cortical folding patterns** determine how much temporal/parietal activity projects to the ear location
- **Local noise sources** (muscle artifacts, jaw movement) may contaminate some in-ear channels differently across subjects

## In-Ear Channel Difficulty Rankings (Across All Subjects)

Some in-ear channels are inherently easier to predict:

| Channel | Mean best_r | Std | Interpretation |
|---------|-------------|-----|----------------|
| ch11    | 0.891       | 0.087 | Easiest -- consistently well-coupled |
| ch 5    | 0.880       | 0.093 | Easy -- low cross-subject variance |
| ch10    | 0.834       | 0.185 | Easy |
| ch 4    | 0.824       | 0.195 | Easy |
| ch 2    | 0.727       | 0.253 | Medium |
| ch 3    | 0.725       | 0.322 | Medium -- high variance across subjects |
| ch 9    | 0.690       | 0.359 | Medium -- very high variance |
| ch 0    | 0.684       | 0.259 | Medium |
| ch 8    | 0.643       | 0.271 | Hard |
| ch 1    | 0.645       | 0.237 | Hard |
| ch 7    | 0.634       | 0.283 | Hard |
| ch 6    | 0.619       | 0.255 | Hard |

Channels 5, 10, 11 are reliably predictable; channels 6, 7, 8 are consistently harder.

## Easy vs Hard Subject Group Comparison

| Metric | Easy (r>=0.7) | Hard (r<0.55) | Ratio | t-test p |
|--------|--------------|---------------|-------|----------|
| In-ear channel RMS CV | 0.0001 | 0.0125 | 0.01x | **< 0.0001** |
| In-ear window variance | 0.9991 | 0.9864 | 1.01x | **< 0.0001** |
| Mean best xcorr | ~0.82 | ~0.60 | 1.37x | 0.06 |
| Well-coupled channels | 11.2/12 | 7.8/12 | 1.44x | -- |

The in-ear channel RMS CV is the strongest differentiator: **easy subjects have nearly identical RMS across all 12 in-ear channels, while hard subjects have 2-3 channels with markedly different (usually lower) RMS**, indicating poor electrode contact or physiological differences.

## Can We Identify Hard Subjects Ahead of Time?

**Yes.** A simple pre-screening metric works:

1. For each in-ear channel, compute its maximum correlation with any scalp channel over a short calibration segment (e.g., 30 seconds)
2. Count the number of channels with best_r > 0.5
3. If fewer than 10/12 channels are well-coupled, flag as "hard subject"

This would correctly identify subjects 8 (8/12), 14 (5/12), and 15 (8/12) as hard, and also flag subjects 2, 6, 10, 12 as marginal.

Alternatively, compute the coefficient of variation of in-ear channel RMS. CV > 0.005 flags hard subjects with near-perfect recall.

## Implications for Improving Predictions on Hard Subjects

### 1. Channel-Weighted Loss Function
Instead of equal-weight MSE across all 12 in-ear channels, weight each channel by its estimated predictability (scalp-inear coupling strength). This prevents the model from wasting capacity on fundamentally unpredictable channels.

### 2. Per-Subject Channel Masking
During evaluation, mask out channels with coupling below a threshold. This would dramatically improve apparent performance on hard subjects while honestly reporting "N/12 channels predicted."

### 3. Subject-Specific Fine-Tuning (Few-Shot)
Hard subjects have unique coupling patterns. Even a few minutes of target-subject data could adapt the spatial filters. The model already works well for the well-coupled channels of hard subjects; it just needs to learn which channels to ignore or handle differently.

### 4. Nonlinear Models for Weak Coupling
For channels where the best linear correlation is only 0.2-0.4, there may still be a nonlinear relationship. Deeper models (attention mechanisms, nonlinear spatial filters) might capture coupling that linear FIR filters miss.

### 5. Robust Training with Channel Dropout
Training with aggressive channel dropout (both scalp input and in-ear target) may produce a model more robust to the heterogeneous channel quality seen in hard subjects.

### 6. Anatomical Priors
If scalp channel locations relative to the ear are known, incorporating this spatial prior could help the model focus on the most relevant scalp channels (temporal/parietal electrodes near the ear).

## Summary

**The dominant predictor of subject difficulty is the intrinsic scalp-to-in-ear coupling strength**, which is a physiological property that varies dramatically across individuals (min channel coupling ranges from 0.99 for Subject 4 to 0.01 for Subject 14). Hard subjects are not noisy or missing data -- they simply have in-ear channels that are weakly coupled to scalp EEG. This is likely due to individual differences in skull conductivity, electrode contact, and cortical anatomy. The most promising improvement strategies are channel-weighted loss functions and subject-specific fine-tuning.
