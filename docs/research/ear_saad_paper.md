# Ear-SAAD Paper: Methods and Findings

**Paper**: "A Direct Comparison of Simultaneously Recorded Scalp, Around-Ear and In-Ear EEG for Neural Selective Auditory Attention Decoding to Speech"
**Authors**: Simon Geirnaert, Simon L. Kappel, Preben Kidmose (2025)
**Source**: `docs/external/ear_saad/geirnaert2025direct.tex`

## Paper's Task vs. Our Task

**Important distinction**: The paper's task is **auditory attention decoding (AAD)** -- classifying which of two speakers the listener is attending to. Their metric is **AAD accuracy (%)** on decision windows.

**Our task** is **scalp-to-in-ear EEG prediction** -- reconstructing in-ear EEG signals from scalp EEG. Our metric is **Pearson r** between predicted and actual in-ear channels.

These are fundamentally different tasks. The paper does NOT do scalp-to-in-ear prediction. However, the paper's methods for linear spatio-temporal filtering are directly relevant.

## Methods Used in the Paper

### Algorithm: Linear Stimulus Reconstruction (Backward Model)
- Reconstruct attended speech envelope from EEG using a linear spatio-temporal decoder
- Decoder: `d_hat = inv(X'X + lambda*I) * X' * s_a`
- Where X is time-lagged EEG (block Hankel matrix), s_a is attended speech envelope
- This is a **backward model** (EEG -> stimulus), not a forward model

### Temporal Filter
- **Filter span: [0, 400] ms post-stimulus** (causal lags only)
- At 20 Hz sampling rate, this is **L = 9 time lags** (0, 50, 100, ..., 400 ms)
- Only causal (post-stimulus) lags used -- neural response follows stimulus

### Regularization: Ledoit-Wolf Shrinkage
- **Yes, they use Ledoit-Wolf analytical shrinkage** for the regularization parameter lambda
- Reference: Ledoit and Wolf (2004) "A well-conditioned estimator for large-dimensional covariance matrices"
- They note this "has shown good performance for this AAD decoder"
- No cross-validation needed for lambda -- it's computed analytically

### Preprocessing Pipeline
1. Bandpass filter 1-9 Hz (zero-phase, 4th-order Butterworth)
2. Split into 10-minute trials
3. EOG regression (scalp EEG only)
4. Bad channel removal (EEGLAB clean_channels, r < 0.45 threshold)
5. Artifact Subspace Reconstruction (ASR, cutoff 5)
6. High-power artifact rejection (per-channel, 1s windows)
7. Bad channels/segments replaced with zeros
8. Downsample to 20 Hz (no anti-aliasing filter)
9. Normalize per trial (Frobenius norm = 1 across time and channels)
10. Common Average Referencing (CAR) per EEG setup

### Speech Feature Extraction
- Gammatone filterbank (19 bands, 50 Hz - 5 kHz)
- Powerlaw compression (exponent 0.6)
- Sum across bands -> single envelope
- Bandpass 1-9 Hz, downsample to 20 Hz
- Normalize to zero mean, unit variance per trial

### Evaluation
- **Leave-one-trial-out CV** for participant-specific decoding (6 folds per participant)
- **Leave-one-participant-out CV** for participant-independent decoding
- Decision windows: 1, 5, 10, 30, 60, 120, 300, 600 seconds
- Primary metric: AAD accuracy on 60s decision windows

## Key Results

### AAD Accuracy (60s windows, participant-specific)
| EEG Setup | AAD Accuracy |
|-----------|-------------|
| Scalp (27 ch) | 83.44% |
| Around-ear (19 ch) | 67.22% |
| In-ear (12 ch) | 61.11% |

### Participant-Independent Decoding
| EEG Setup | AAD Accuracy | Drop from specific |
|-----------|-------------|-------------------|
| Scalp | 75.77% | -7.67 pp |
| Around-ear | 65.55% | -1.67 pp |
| In-ear | 49.44% | -11.67 pp (below significance!) |

### Key Finding on In-Ear Cross-Subject
- Participant-independent decoding with in-ear EEG **fails** (below significance)
- This aligns with our finding that cross-subject variability is the bottleneck
- Reasons: noise variability, anatomical differences in earpiece fit, variable electrode orientations

## Tricks We Haven't Tried

### 1. Ledoit-Wolf Shrinkage (PROPERLY)
We tried this in iter035 but combined it with **causal-only lags**, which hurt performance.
**Action**: Try Ledoit-Wolf with our current acausal FIR setup (7 taps centered).

### 2. Frobenius Norm Normalization
Paper normalizes each trial so Frobenius norm across time and channels = 1.
We may be using different normalization. Worth checking.

### 3. Per-Trial Normalization
Paper normalizes EEG per trial (Frobenius norm) and envelope per trial (zero mean, unit variance).
Our benchmark may handle normalization differently.

### 4. Artifact Subspace Reconstruction (ASR)
Paper uses ASR with cutoff=5. We may not be applying this level of artifact rejection.
Could help with noisy subjects (especially Subject 14).

### 5. Bad Channel Detection + Zero Replacement
Paper explicitly detects bad channels (r < 0.45 with other channels) and replaces with zeros.
We interpolate NaN values linearly -- the paper's approach of letting the decoder handle zeros may be better.

### 6. Greedy Forward Node Selection
Paper shows that adding 3 strategically chosen scalp electrodes (FC5, C3, T8) to in-ear EEG achieves near-full-scalp AAD accuracy. This suggests certain scalp channels are far more informative than others.
**Action**: Try channel selection/weighting in our prediction model.

### 7. Important Scalp Electrodes
The most important scalp electrodes for in-ear EEG are in left-lateralized fronto-central areas:
- FC5, C3, T8 (top 3 for in-ear)
- FC5, C3, T7 (top 3 for around-ear)
- These are near auditory cortex -- makes physiological sense

## Comparison to Our Approach

| Aspect | Paper | Our Best (iter030) |
|--------|-------|-------------------|
| Task | AAD (classify attention) | Predict in-ear from scalp |
| Temporal filter | 9 causal lags (0-400ms) | 7 acausal taps (FIR) |
| Regularization | Ledoit-Wolf analytical | SGD with combined loss |
| Normalization | Frobenius norm per trial | (check benchmark.py) |
| Cross-subject | Leave-one-participant-out | LOSO (train 1-12, test 13-15) |
| In-ear cross-subject | FAILS (below significance) | r = 0.378 (weak but present) |

## Implications for Our Work

1. **The paper confirms cross-subject in-ear prediction is extremely hard** -- even the original authors couldn't get significant participant-independent decoding with in-ear EEG alone.

2. **Our r = 0.378 is likely near the ceiling** for cross-subject linear methods on this narrowband (1-9 Hz, 20 Hz) data. The paper's failure at participant-independent in-ear decoding supports this.

3. **Broadband data (1-45 Hz at 256 Hz)** is the most promising path forward -- more frequency content means more information for the decoder.

4. **Subject-specific fine-tuning** could help enormously -- the paper shows participant-specific decoding works much better than participant-independent.

5. **Channel selection** could help -- not all 27 scalp channels contribute equally. FC5, C3, T7/T8 are most important.
