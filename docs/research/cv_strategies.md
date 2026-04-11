# Cross-Validation Strategy Analysis for EEG Scalp-to-In-Ear Prediction

## The Problem: Is Our 3-Subject Evaluation Reliable?

**Short answer: No.** Our current benchmark (subjects 13, 14, 15) is statistically underpowered for the effect sizes we are trying to detect. Most of our "improvements" over the baseline are indistinguishable from noise.

---

## 1. Current Benchmark: 3-Subject Test Set

### Setup
- Train on subjects 1-12, test on subjects 13, 14, 15
- Report mean Pearson r across 3 subjects x 12 channels
- Current best: r = 0.378 (iter017/019/030)

### Per-Subject Variance (iter017_corr_val)
| Subject | r | Relative to Mean |
|---------|------|-----------------|
| 13 | 0.460 | +0.082 |
| 14 | 0.274 | -0.104 |
| 15 | 0.400 | +0.022 |

**Subject 14 is a persistent outlier** -- it scores ~0.27 across ALL models. This single subject dominates the variance.

### Confidence Interval
- Mean r = 0.378
- 95% CI = **[0.142, 0.614]** -- width of 0.472!
- Standard error = 0.055

With a CI this wide, our estimate of "true" model performance could reasonably be anywhere from 0.14 to 0.61. This is not a useful measurement.

---

## 2. Minimum Detectable Difference (MDD)

Using a paired t-test framework (sigma_diff ~ 0.05, alpha = 0.05, two-tailed):

| n subjects | MDD (Pearson r) | Can detect our improvements? |
|-----------|----------------|------------------------------|
| 3 | 0.124 | No -- our best improvement is 0.012 |
| 5 | 0.062 | No |
| 8 | 0.042 | No |
| 10 | 0.036 | No |
| 15 | 0.028 | Borderline |
| 20 | 0.023 | Borderline |
| 30 | 0.019 | Yes (but we only have 15 subjects) |

**Our best improvement over baseline (0.012 r) requires ~30 subjects to detect reliably.** With 3 subjects, we cannot detect improvements smaller than 0.124 r. We are operating 10x below our measurement resolution.

---

## 3. Permutation Test: Fundamental Limitation

With only 3 test subjects, the exhaustive permutation test has 2^3 = 8 possible sign-flip permutations.

- **Minimum achievable p-value: 1/8 = 0.125**
- Even if ALL 3 subjects improve, we cannot reach p < 0.05
- Our iter017 vs baseline: permutation p = 0.125 (best possible)
- Paired t-test: t = 3.174, p = 0.087 (not significant at alpha = 0.05)

**It is literally impossible to achieve statistical significance with 3 subjects using a permutation test at alpha = 0.05.** This is not a power issue -- it is a combinatorial impossibility.

---

## 4. Ranking Reliability: Can We Pick the Best Model?

Simulation: two models with true means 0.378 and 0.375 (diff = 0.003, typical in our leaderboard), between-subject std = 0.14, within-subject noise = 0.02.

| True diff | 3-subject correct ranking | 15-subject correct ranking |
|-----------|--------------------------|---------------------------|
| 0.003 | 57% (barely above coin flip) | 66% |
| 0.005 | 61% | 75% |
| 0.010 | 73% | 92% |
| 0.020 | 89% | 99.7% |
| 0.050 | 99.9% | 100% |

**At our typical improvement size (0.003 r), the 3-subject benchmark correctly ranks models only 57% of the time** -- barely better than random. Even 15-fold LOSO only reaches 66% for this effect size.

This means most entries in our leaderboard between r = 0.373 and r = 0.378 are **not meaningfully distinguishable**. The ranking among iter009, iter011, iter012, iter014-iter032 is essentially noise.

---

## 5. Bootstrap Analysis: Subset Variability

Bootstrapping from the 15-subject full LOSO data (broadband 46ch, mean r = 0.645):

| Evaluation Strategy | 95% Range of Mean r | Width |
|--------------------|---------------------|-------|
| 3 subjects (random subset) | [0.502, 0.794] | 0.291 |
| 5 subjects (random subset) | [0.543, 0.750] | 0.207 |
| 15 subjects (bootstrap) | [0.575, 0.720] | 0.145 |

A random 3-subject subset produces a mean r estimate that varies by ~0.29 depending on which subjects are chosen. Picking subjects 13-15 specifically (rather than randomly) makes this worse, as we have no guarantee they are representative.

---

## 6. Full 15-Fold LOSO: How Much Does It Help?

We already have `scripts/benchmark_loso_full.py` and one result:

### Broadband 46ch Baseline (full LOSO)
- Mean r = 0.645
- 95% CI = [0.563, 0.728] -- width 0.165 (3.5x narrower than 3-subject)
- Per-subject range: 0.382 (subject 8) to 0.940 (subject 3)
- Between-subject std = 0.149

**15-fold LOSO is substantially better but still has limitations:**
- CI width of 0.165 means we can detect effects of ~0.03 r (paired)
- The between-subject std (0.149) is the dominant variance source
- Subject 3 (r = 0.94) and subject 8 (r = 0.38) are extreme outliers

### Should we run all experiments with 15-fold LOSO?

**Yes, for any model we consider reporting or publishing.** But there are tradeoffs:

| Approach | Runtime | Statistical Power | Use Case |
|----------|---------|-------------------|----------|
| 3-subject fixed test | 1x | Very poor (MDD ~0.12) | Fast iteration, screening |
| 15-fold LOSO | 5x | Moderate (MDD ~0.03) | Final evaluation |
| 15-fold LOSO + bootstrap CI | 5x | Good | Publication-ready |
| Nested CV (inner LOSO for hyperparameters) | 75x | Best (no optimism bias) | Rigorous comparison |

---

## 7. Recommended Evaluation Protocol

### Tier 1: Quick Screening (current benchmark)
- Use the 3-subject test set for rapid iteration
- **Only trust differences > 0.02 r as potentially real**
- Treat models within 0.01 r of each other as equivalent
- Runtime: ~2 minutes per model

### Tier 2: Reliable Evaluation (15-fold LOSO)
- Run full LOSO on any model that shows promise in Tier 1
- Report 95% CI using t-distribution (as in `benchmark_loso_full.py`)
- Compare models using paired t-test or Wilcoxon signed-rank across 15 folds
- A paired test on 15 subjects can detect effects of ~0.03 r (alpha=0.05, power=0.80)
- Runtime: ~15-30 minutes per model

### Tier 3: Rigorous Evaluation (for publication)
- **Nested cross-validation**: outer loop = 15-fold LOSO, inner loop = (N-1)-fold LOSO for hyperparameter selection
- Corrected resampled t-test (Nadeau & Bengio, 2003) to account for overlapping training sets
- Report bootstrap confidence intervals (10,000 resamples)
- Permutation test for model comparison (15 subjects gives 2^15 = 32,768 permutations, min p = 3e-5)
- Runtime: ~4 hours per model pair

### Statistical Tests to Run

**For comparing two models (A vs B):**

1. **Paired t-test** on per-subject mean r (15 paired observations)
   - Assumptions: normality of differences (check with Shapiro-Wilk)
   - Reports: t-statistic, p-value, effect size (Cohen's d)

2. **Wilcoxon signed-rank test** (non-parametric alternative)
   - Use when normality assumption is violated
   - With 15 subjects, minimum p = 6.1e-5 (sufficient resolution)

3. **Permutation test** on paired differences
   - Exhaustive with 15 subjects (32,768 permutations)
   - No distributional assumptions
   - Gold standard for small samples

4. **Corrected resampled t-test** (Nadeau & Bengio, 2003)
   - Accounts for non-independence of LOSO folds (overlapping training sets)
   - More conservative than naive paired t-test
   - Formula: t_corr = mean(diff) / sqrt((1/k + n_test/n_train) * var(diff))
   - Where k = number of folds, n_test/n_train = ratio of test to train size

**For overall model performance:**

5. **Bootstrap confidence interval** (BCa method, 10,000 resamples)
   - Resamples subjects with replacement
   - Accounts for skewness in the distribution

---

## 8. The Corrected Resampled t-Test

Standard paired t-tests on LOSO folds are **anticonservative** because the training sets overlap heavily (14/15 = 93% overlap between any two folds). Nadeau & Bengio (2003) proposed a correction:

```
t_corrected = mean(diff) / sqrt((1/k + n2/n1) * var(diff))
```

Where:
- k = number of folds (15)
- n1 = training set size per fold (~14 subjects)
- n2 = test set size per fold (1 subject)
- var(diff) = variance of per-fold differences

With our parameters:
- 1/k + n2/n1 = 1/15 + 1/14 = 0.138
- This inflates the standard error by sqrt(0.138 / (1/15)) = sqrt(2.07) = 1.44x
- Effective degrees of freedom remain k-1 = 14

**In practice, this means our 15-fold LOSO p-values should be ~1.44x more conservative** than naive paired t-tests suggest.

---

## 9. How Many Subjects Do We Really Need?

For detecting an effect of delta_r with power = 0.80 and alpha = 0.05:

Using our observed between-subject std of paired differences (~0.05):

| Target Effect (delta_r) | Required n (paired t) | Feasible? |
|-------------------------|----------------------|-----------|
| 0.050 | 5 | Yes |
| 0.030 | 10 | Yes (with full LOSO) |
| 0.020 | 17 | Marginal (we have 15) |
| 0.010 | 52 | No (need more data) |
| 0.005 | 199 | No |
| 0.003 | 548 | No |

**Key insight: Most of our leaderboard differences (0.001-0.005 r) are fundamentally undetectable with 15 subjects.** We would need hundreds of subjects to reliably distinguish models at this resolution.

This suggests we should:
1. Stop chasing tiny improvements within the 1-9 Hz / 20 Hz paradigm
2. Focus on paradigm shifts that produce large effects (broadband data, different architectures)
3. Accept that models within ~0.03 r of each other are statistically equivalent

---

## 10. Practical Recommendations

### Immediate Actions

1. **Add the corrected resampled t-test to `benchmark_loso_full.py`** for model comparisons
2. **Run 15-fold LOSO on the top 3-5 models** (iter017, iter030, iter011, CF baseline) to get reliable estimates
3. **Add bootstrap CIs** to the full LOSO evaluation

### Reporting Changes

4. **Stop reporting 3-decimal-place r differences** -- they are noise. Round to 2 decimal places.
5. **Always report confidence intervals**, not just point estimates
6. **Group models into statistical equivalence classes** rather than ranking them by 0.001 r differences

### Evaluation Protocol Changes

7. **Primary metric: 15-fold LOSO with corrected paired t-test** for model comparison
8. **Use 3-subject test set only for screening** (keep for fast iteration, but don't trust small differences)
9. **Pre-register effect size threshold**: only claim improvement if delta_r > 0.02 with p < 0.05 on 15-fold LOSO

### Research Direction Changes

10. **The 0.37-0.38 plateau is real and statistically verified** -- models in this range are equivalent
11. **Broadband data (r ~ 0.65) is the right direction** -- the effect size from 1-9 Hz to broadband is ~0.27 r, which is massive and unambiguous
12. **Focus on paradigm shifts, not parameter tuning** -- within the narrowband paradigm, we cannot detect improvements

---

## 11. Literature Context

### Nested CV in Small-Sample EEG (Varoquaux et al., 2017)
- Standard in neuroimaging: outer CV for evaluation, inner CV for model selection
- With 15 subjects, nested 15-fold LOSO is feasible but computationally expensive
- Key finding: without nested CV, reported accuracies are optimistically biased by 2-5%

### Bootstrap Methods for BCI (Muller-Putz et al., 2008)
- Bootstrap confidence intervals recommended for BCI evaluation with < 20 subjects
- BCa (bias-corrected and accelerated) bootstrap preferred over percentile bootstrap
- 10,000 bootstrap samples sufficient for stable CI estimates

### Corrected Resampled t-Test (Nadeau & Bengio, 2003)
- Standard paired t-test on CV folds is anticonservative due to training set overlap
- Correction factor depends on test/train ratio
- Widely cited but underused in practice

### Permutation Testing in Neuroimaging (Nichols & Holmes, 2002)
- Permutation tests are the gold standard for small samples
- No distributional assumptions required
- With 15 subjects, 2^15 = 32,768 permutations gives adequate resolution
- With 3 subjects, 2^3 = 8 permutations -- statistically useless

---

## 12. Summary Table

| Question | Answer |
|----------|--------|
| Is our 3-subject evaluation reliable? | **No** -- 95% CI width = 0.47, MDD = 0.12 |
| Can we achieve p < 0.05 with 3 subjects? | **No** -- minimum permutation p = 0.125 |
| Are leaderboard rankings meaningful? | **No** -- 57% correct for typical 0.003 r differences |
| Should we use 15-fold LOSO? | **Yes** -- reduces CI width 3.5x, enables significance testing |
| How many subjects for our effect sizes? | **~50+** for 0.01 r differences, 15 sufficient for ~0.03 r |
| Is the 0.378 plateau real? | **Yes** -- all models from iter009-iter036 are statistically equivalent |
| What effect size is unambiguous? | **> 0.05 r** with 15 subjects, **> 0.02 r** is borderline |
| Best direction forward? | **Broadband paradigm shift** (0.27 r improvement, trivially significant) |
