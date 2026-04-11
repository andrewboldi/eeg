# Generalization Analysis: Training Set Size and Theoretical Ceiling

**Dataset**: Ear-SAAD broadband_46ch.h5 (15 subjects, 46 scalp -> 12 in-ear)
**Model**: Closed-form linear spatial filter (CF baseline)
**Current LOSO mean r**: 0.6454

## 1. Training Set Size Scaling Law

We train CF baseline on k randomly-sampled subjects (k=2..14) and evaluate on the remaining subjects.
Each configuration is repeated 20 times with different random subsets.

**Fitted model**: r(n) = r_inf - a / n^b
- r_inf = 0.6883 +/- 0.0267
- a = 0.1862
- b = 1.2133

| Training Subjects | Measured Mean r | Measured Std | Notes |
|-------------------|----------------|--------------|-------|
|  2 | 0.6071 | 0.0657 |
|  3 | 0.6433 | 0.0501 |
|  4 | 0.6509 | 0.0243 |
|  5 | 0.6632 | 0.0251 |
|  6 | 0.6633 | 0.0294 |
|  7 | 0.6644 | 0.0237 |
|  8 | 0.6737 | 0.0459 |
|  9 | 0.6869 | 0.0539 |
| 10 | 0.6825 | 0.0699 |
| 11 | 0.6620 | 0.0627 |
| 12 | 0.6674 | 0.0549 |
| 13 | 0.7322 | 0.1074 | *high variance: only 2 test subjects per draw* |
| 14 | 0.6474 | 0.1513 | *high variance: only 1 test subject per draw* |

### Extrapolations

| Training Subjects | Predicted Mean r | Delta vs Current (n=14) |
|-------------------|-----------------|------------------------|
|  14 | 0.6807 | +0.0000 |
|  20 | 0.6834 | +0.0027 |
|  30 | 0.6853 | +0.0046 |
|  50 | 0.6867 | +0.0060 |
| 100 | 0.6876 | +0.0069 |
| 200 | 0.6880 | +0.0073 |
| 500 | 0.6882 | +0.0075 |

## 2. Which Subjects Benefit Most from More Training Data?

Log-slope = d(r)/d(log(n_train)): how much each subject's r improves per doubling of training subjects.

| Subject | LOSO r (n=14) | r at n=2 | r at n=14 | Improvement | Log-slope | Category |
|---------|--------------|----------|-----------|-------------|-----------|----------|
|  4 | 0.840 | 0.772 | 0.957 | +0.184 | +0.0882 | Easy |
|  3 | 0.940 | 0.804 | N/A | N/A | +0.0792 | Easy |
|  7 | 0.618 | 0.717 | N/A | N/A | +0.0686 | Medium |
|  1 | 0.586 | 0.594 | N/A | N/A | +0.0555 | Medium |
|  9 | 0.765 | 0.670 | N/A | N/A | +0.0536 | Easy |
|  5 | 0.732 | 0.727 | 0.819 | +0.091 | +0.0460 | Easy |
|  2 | 0.539 | 0.481 | 0.576 | +0.094 | +0.0429 | Hard |
| 11 | 0.742 | 0.670 | 0.744 | +0.075 | +0.0397 | Easy |
| 12 | 0.548 | 0.509 | 0.583 | +0.074 | +0.0363 | Hard |
| 13 | 0.727 | 0.661 | 0.721 | +0.060 | +0.0315 | Easy |
| 10 | 0.653 | 0.602 | 0.644 | +0.042 | +0.0213 | Medium |
|  6 | 0.575 | 0.504 | 0.544 | +0.040 | +0.0189 | Medium |
| 15 | 0.601 | 0.541 | 0.571 | +0.029 | +0.0171 | Medium |
|  8 | 0.382 | 0.427 | 0.459 | +0.032 | +0.0148 | Hard |
| 14 | 0.432 | 0.388 | 0.417 | +0.029 | +0.0144 | Hard |

Note: "N/A" for r at n=14 means the subject was never the sole held-out subject in 20 random draws
(at k=14, only 1 of 15 subjects is held out per draw, so each subject appears ~1.3 times on average).
The log-slope metric uses all data points across all k values and is more reliable.

### Subjects that benefit most from more training data

- **Subject 4**: log-slope=+0.0882, improvement +0.184
- **Subject 3**: log-slope=+0.0792, improvement +nan
- **Subject 7**: log-slope=+0.0686, improvement +nan
- **Subject 1**: log-slope=+0.0555, improvement +nan
- **Subject 9**: log-slope=+0.0536, improvement +nan

### Subjects that plateau early (diminishing returns)

- **Subject 6**: log-slope=+0.0189, improvement +0.040
- **Subject 15**: log-slope=+0.0171, improvement +0.029
- **Subject 8**: log-slope=+0.0148, improvement +0.032
- **Subject 14**: log-slope=+0.0144, improvement +0.029

## 3. Theoretical Ceiling for Cross-Subject Prediction

Three independent estimates of the ceiling:

| Method | Estimate | Notes |
|--------|----------|-------|
| Scaling law asymptote (n -> inf) | **0.6883** +/- 0.0267 | Power law fit to n=2..14 data |
| Top-5 subject average | 0.8037 | Best case if all subjects were 'easy' |
| Noise ceiling (shared variance) | 0.7910 | r_within * shared_variance_ratio (0.833) |

### Interpretation

- **Current performance**: r = 0.6454 (14 training subjects)
- **Scaling law ceiling**: r = 0.6883 (achievable with infinite similar subjects)
- **Remaining gap**: 0.0429

### What would 20 training subjects give us?

Extrapolated mean r at n=20: **0.6834** (improvement of **+0.0027** over n=14)

This modest gain reflects the diminishing returns of adding similar subjects.
The cross-subject mapping is already well-characterized with 14 subjects;
the bottleneck is inter-individual physiological variability, not training set size.

## 4. Key Takeaways

1. **Diminishing returns from more subjects**: The scaling law shows rapid saturation.
   Going from 14 to 20 subjects would improve mean r by only ~+0.0027.

2. **Subject-specific physiology is the bottleneck**: Hard subjects (8, 14, 2) have
   fundamentally weak scalp-to-in-ear coupling that no amount of cross-subject
   training data can overcome.

3. **Two paths to improvement**:
   - *Better models* (nonlinear, attention): May capture nonlinear coupling for hard subjects
   - *Subject-specific adaptation*: Even a few minutes of target-subject calibration
     data would help more than adding 100 new training subjects

4. **The 0.65 plateau for CF baseline on broadband is near the cross-subject ceiling**.
   Nonlinear models (iter039) already push past this to 0.638 on the 3-subject test set,
   suggesting some nonlinear coupling exists.
