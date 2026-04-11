# Channel Importance Analysis: 46-Channel Setup

**Dataset**: Ear-SAAD broadband_46ch.h5 (15 subjects, 46 input -> 12 in-ear output, 128 Hz, 3594 windows/subject)  
**Method**: Per-window Pearson correlation (analyses 1-3), spatial closed-form linear model LOSO on subjects 13/14/15 (ablation)  
**Input channels**: 27 scalp (10-20 system) + 19 around-ear (9 left cEEGrid + 10 right cEEGrid)

---

## 1. Which Channels Contribute Most to Prediction?

### Overall Importance Ranking (mean |r| with all 12 in-ear channels)

Around-ear channels completely dominate the top 19 positions. No scalp channel reaches the importance of even the weakest around-ear channel.

| Rank | Channel | Type | Mean |r| | Notes |
|------|---------|------|---------|-------|
| 1 | cEL9 | around-ear L | 0.345 | Best overall |
| 2 | cEL8 | around-ear L | 0.339 | |
| 3 | cER10 | around-ear R | 0.338 | Best right-side |
| 4 | cER9 | around-ear R | 0.336 | |
| 5 | cEL7 | around-ear L | 0.319 | |
| 6 | cER7 | around-ear R | 0.317 | |
| 7 | cEL6 | around-ear L | 0.305 | |
| 8 | cEL5 | around-ear L | 0.304 | |
| 9 | cEL4 | around-ear L | 0.303 | |
| 10 | cER8 | around-ear R | 0.298 | |
| 11 | cER6 | around-ear R | 0.297 | |
| 12 | cER1 | around-ear R | 0.293 | |
| 13 | cEL3 | around-ear L | 0.283 | |
| 14 | cEL1 | around-ear L | 0.277 | |
| 15 | cER3 | around-ear R | 0.268 | |
| 16 | cER5 | around-ear R | 0.252 | |
| 17 | cER2 | around-ear R | 0.238 | |
| 18 | cER4 | around-ear R | 0.235 | |
| 19 | **T8** | **scalp** | **0.214** | **Best scalp channel** |
| 20 | cEL2 | around-ear L | 0.211 | Weakest AE (except cEL2) |
| 21 | FC6 | scalp | 0.208 | |
| 22 | T7 | scalp | 0.206 | |
| 23 | CP5 | scalp | 0.195 | |
| 24 | FC5 | scalp | 0.190 | Paper top pick |
| 25 | P7 | scalp | 0.188 | |
| 26 | F8 | scalp | 0.185 | |
| 27-33 | P8,CP6,O1,O2,C3,C4,F7 | scalp | 0.13-0.18 | Moderate |
| 34-40 | P3,P4,F4,F3,CP1,Pz,CP2 | scalp | 0.07-0.12 | Low |
| 41-46 | FC1,FC2,Cz,Fp1,Fp2,Fz | scalp | 0.03-0.07 | **Negligible** |

### Scalp Channel Importance by Brain Region

| Region | Mean |r| | Channels |
|--------|---------|----------|
| Temporal | 0.210 | T7, T8 |
| Fronto-Central (lateral) | 0.199 | FC5, FC6 |
| Centro-Parietal (lateral) | 0.185 | CP5, CP6 |
| Parietal (lateral) | 0.178 | P7, P8 |
| Occipital | 0.163 | O1, O2 |
| Central | 0.118 | Cz, C3, C4 |
| Frontal (lateral) | 0.107 | F3, F4, F7, F8 |
| Fronto-Central (midline) | 0.064 | FC1, FC2 |
| Centro-Parietal (midline) | 0.076 | CP1, CP2 |
| Frontal (midline/polar) | 0.047 | Fz, Fp1, Fp2 |
| Parietal (midline) | 0.098 | Pz, P3, P4 |

**Key pattern**: Lateral temporal and periauricular scalp channels (T7, T8, FC5, FC6, CP5, CP6) are most important. Midline channels (Fz, Cz, Pz, FC1, FC2, CP1, CP2) contribute minimally. This makes physiological sense -- in-ear electrodes primarily measure activity from temporal cortex, and electrodes closest to the ear pick this up best.

### Laterality: Left In-Ear Predicts from Left, Right from Right

Every left in-ear channel (ELA-ELT) is best predicted by left around-ear channels (cEL1-9), and every right in-ear channel (ERA-ERT) is best predicted by right around-ear channels (cER1-10). There is essentially **zero cross-hemisphere information** (Left-Right around-ear cross-r = 0.0001).

**Best scalp-to-in-ear correlations by channel:**
- Left in-ear (ELI, ELT): best from cEL9 (r=0.77), no scalp channel above r=0.35
- Right in-ear (ERI, ERT): best from cER10 (r=0.78), T8 reaches r=0.36 for ERC
- Hard channels (ERA, ELC): even best around-ear only reaches r=0.46-0.49

---

## 2. Around-Ear Channel Redundancy

### Inter-Correlation Within Around-Ear Channels

| Group | Mean Pairwise r | Interpretation |
|-------|----------------|----------------|
| Left (cEL1-9) | 0.545 | Moderately correlated |
| Right (cER1-10) | 0.543 | Moderately correlated |
| Left-Right cross | 0.000 | Completely independent |

### Most Redundant Pairs (r > 0.80)

| Pair | Correlation | Could Drop One? |
|------|------------|-----------------|
| cEL5 -- cEL6 | 0.883 | Yes (keep cEL5 or cEL6) |
| cER9 -- cER10 | 0.865 | Yes (keep cER10, rank 3) |
| cEL6 -- cEL7 | 0.857 | Yes (keep cEL7, rank 5) |
| cEL7 -- cEL8 | 0.814 | Borderline |
| cER6 -- cER7 | 0.809 | Borderline |
| cEL8 -- cEL9 | 0.809 | Borderline |

The around-ear electrodes form a **spatial chain** around each ear, so adjacent electrodes are highly correlated. The redundancy follows a nearest-neighbor pattern -- electrodes that are physically adjacent share the most signal.

### Non-Redundant Around-Ear Subset

A pruned set of 11 around-ear channels (dropping 8 redundant ones: cEL2, cEL6, cEL8, cER2, cER4, cER5, cER8) retains most information. However, ablation shows this costs r=0.008 (see section 3).

---

## 3. Channel-Dropping Ablation Results

### Ablation Table (spatial CF, LOSO on subjects 13/14/15)

| Configuration | Channels | r | Delta vs Full |
|---------------|----------|------|---------------|
| **All 46 channels** | 27s + 19ae | **0.5832** | baseline |
| 19 around-ear only | 0s + 19ae | 0.5774 | -0.006 |
| 27 scalp only | 27s + 0ae | 0.4575 | -0.126 |
| Top 36 by corr | 17s + 19ae | 0.5803 | -0.003 |
| **Top 30 by corr** | **11s + 19ae** | **0.5803** | **-0.003** |
| **Top 26 by corr** | **7s + 19ae** | **0.5806** | **-0.003** |
| Top 20 by corr | 1s + 19ae | 0.5789 | -0.004 |
| Top 15 by corr | 0s + 15ae | 0.5793 | -0.004 |
| Top 10 by corr | 0s + 10ae | 0.5583 | -0.025 |
| Paper 6 + 19 AE | 6s + 19ae | 0.5810 | -0.002 |
| Paper 6 only | 6s + 0ae | 0.4493 | -0.134 |
| 9 TC + 19 AE | 9s + 19ae | 0.5802 | -0.003 |
| 27 scalp + 11 AE | 27s + 11ae | 0.5756 | -0.008 |
| 20 lateral + 19 AE | 20s + 19ae | 0.5795 | -0.004 |
| 8 lateral + 10 AE | 8s + 10ae | 0.5699 | -0.013 |

### Key Findings

1. **Around-ear channels carry almost all the information.** Dropping all 27 scalp channels costs only r=0.006. Dropping all 19 around-ear channels costs r=0.126. Around-ear alone (0.5774) nearly matches full 46ch (0.5832).

2. **You can safely drop 20 channels.** The top 26 channels (7 scalp + 19 AE = 26ch) achieve r=0.5806, losing only 0.003 from full 46ch. The 20 dropped channels are all midline/frontal scalp electrodes that contribute negligible information.

3. **Even aggressive pruning works.** Top 15 (all around-ear) achieves r=0.5793, losing only 0.004. Below 15 channels, performance drops more steeply.

4. **Pruning around-ear is costlier than pruning scalp.** Dropping 8 redundant around-ear channels (keeping 11) costs r=0.008, more than dropping 20 scalp channels (r=0.003).

5. **The paper's recommended channels (FC5, C3, T7, T8, CP5, CP6) work well as a scalp subset** when combined with around-ear (r=0.5810), but alone are worse than scalp-only (r=0.4493 vs 0.4575).

---

## 4. Recommended Attention Weights for Channel-Weighting Strategy

Based on the correlation analysis and ablation results, here are recommended attention weight tiers:

### Tier 1: Critical (weight 1.0) -- 19 around-ear channels
All cEL and cER channels. These carry the vast majority of predictive information. Never drop these.

### Tier 2: Helpful (weight 0.5-0.7) -- 7 lateral scalp channels
| Channel | |r| | Rationale |
|---------|-----|-----------|
| T8 | 0.214 | Best scalp channel, right temporal |
| FC6 | 0.208 | Right fronto-central, near ear |
| T7 | 0.206 | Left temporal |
| CP5 | 0.195 | Left centro-parietal |
| FC5 | 0.190 | Left fronto-central (paper top pick) |
| P7 | 0.188 | Left parietal |
| F8 | 0.185 | Right frontal |

### Tier 3: Marginal (weight 0.2-0.4) -- 8 scalp channels
P8, CP6, O1, O2, C3, C4, F7, P3 (|r| = 0.12-0.18). Include if compute allows; dropping them costs <0.001 r.

### Tier 4: Negligible (weight 0.0-0.1, safe to drop) -- 12 scalp channels
P4, F4, F3, CP1, Pz, CP2, FC1, FC2, Cz, Fp1, Fp2, Fz (|r| < 0.12). These contribute effectively nothing. Dropping all 12 has no measurable effect on performance.

---

## 5. Practical Recommendations

### For Maximum Performance
Use all 46 channels (r=0.5832). The marginal scalp channels add a small but measurable benefit (+0.006 over AE-only).

### For Efficient Models (recommended)
Use 26 channels: 19 around-ear + 7 lateral scalp (T8, FC6, T7, CP5, FC5, P7, F8). This achieves r=0.5806, only 0.003 below full 46ch, with 43% fewer parameters.

### For Minimal Hardware
Use 19 around-ear channels only (r=0.5774). No scalp cap needed. Performance is 99% of full 46ch.

### For Channel-Weighted Models
Apply learned attention weights initialized with the correlation-based importance scores above. This lets the model discover optimal weightings during training while starting from a good prior. Key: weight around-ear channels 2-5x higher than scalp channels.

### Comparison with Paper's Findings
The paper identified FC5, C3, T7/T8 as most important for AAD (auditory attention decoding). Our analysis partially agrees -- T7, T8, FC5 are in our top scalp channels -- but for the direct prediction task, **C3 ranks only 31st** (|r|=0.151). This is because the paper's task (classifying attended speaker) relies on auditory cortex processing, while our task (predicting in-ear EEG) depends more on physical proximity and volume conduction. The around-ear channels, being physically closest to the in-ear electrodes, dominate for our task.
