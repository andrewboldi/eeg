# Test-Time Adaptation & Domain Adaptation for EEG: Research Summary

## Context
Our task: predict 12-channel in-ear EEG from 46-channel scalp+around-ear EEG. Trained on 12 subjects, tested on 3 held-out subjects. Cross-subject variability is the main bottleneck (Subject 8: r=0.38, Subject 3: r=0.94). We have access to test subjects' INPUT scalp EEG at test time, but NOT their in-ear EEG labels.

## Papers Downloaded

1. **NeuroTTT** (arXiv 2509.26301, ICLR 2026) — `docs/external/neurottt/`
2. **MDM-Tent** (arXiv 2509.24700, ICASSP 2026) — `docs/external/seeg_tta/`
3. **SFDA-SSVEP** (arXiv 2305.17403, JNE) — `docs/external/sfda_ssvep/`
4. **Revisiting Euclidean Alignment** (arXiv 2502.09203, JNE 2025) — `docs/external/euclidean_alignment/`

---

## Method 1: Tent — Entropy Minimization (Wang et al., ICLR 2021)

**Core idea:** At test time, minimize the entropy of model predictions by updating ONLY BatchNorm parameters (affine scale gamma, shift beta, and optionally running mean/variance).

**Algorithm:**
```
For each test batch x_test:
    1. Forward pass: y_hat = model(x_test)
    2. Compute entropy: H = -sum(p * log(p))
    3. Backprop through H, update only BN params:
       theta_BN <- theta_BN - alpha * grad(H, theta_BN)
    4. Use adapted model for final prediction
```

**Key properties:**
- Source-free: no access to training data needed
- Updates only BN parameters (~0.1% of model params)
- One forward + one backward pass per batch
- Extremely lightweight, suitable for real-time

**Relevance to our task:** Our current FIR model has NO BatchNorm layers — we would need to add them. However, for a regression task, we cannot compute entropy of predictions directly (no class probabilities). We would need an alternative unsupervised objective.

**Adaptation for regression:**
- Instead of entropy minimization, we could minimize a self-supervised objective on the INPUT data (e.g., reconstruction loss, temporal consistency)
- Or: add a classification auxiliary head (predict which subject/segment) and use its entropy

---

## Method 2: NeuroTTT — Self-Supervised Test-Time Training (Wang et al., ICLR 2026)

**Two-stage approach:**

### Stage I: Domain-Specific Self-Supervised Fine-tuning
During training, jointly optimize:
```
L_finetune = L_main(g(f(x)), y) + sum_j(w_j * L_ssl_j(h_j(f(x_tilde_j))))
```
Where:
- `f` = shared backbone (feature extractor)
- `g` = main task head (regression in our case)
- `h_j` = self-supervised task heads (lightweight classifiers)
- `x_tilde_j` = transformed input for SSL task j

**SSL tasks they use (domain-specific):**
1. **Stopped-Band Prediction**: Zero out a frequency band, predict which band was removed (classification over bands)
2. **Amplitude Scaling Prediction**: Scale signal amplitude, predict scale factor
3. **Anterior-Posterior Flip Detection**: Swap frontal/parietal channels, predict if flipped
4. **Temporal Jigsaw**: Shuffle temporal segments, predict permutation

### Stage II: Test-Time Training
For each test sample:
```
1. Compute SSL loss on x_test (no labels needed):
   L_ssl = CrossEntropy(h(f(x_test_transformed)), known_transformation)
2. One gradient step:
   theta' = theta - alpha * grad(L_ssl, theta)
3. Predict with adapted model:
   y_hat = g(f_theta'(x_test))
4. Optionally reset or carry forward theta'
```

**Key finding:** Tent (BN-only entropy minimization) often OUTPERFORMS full-parameter SSL-based TTT for cross-subject transfer. Reason: full-model TTT on single noisy samples can destabilize the model, while Tent's BN-only updates provide gentler nudging.

**Relevance to our task:**
- We can design EEG-specific SSL tasks for our scalp-to-in-ear problem:
  - **Channel dropout prediction**: Zero out a channel, predict which one
  - **Temporal reversal detection**: Reverse time, predict if reversed
  - **Frequency band masking**: Mask a band, predict which
- Train the model with both regression loss AND SSL losses
- At test time, use SSL on unlabeled scalp EEG to adapt

---

## Method 3: Euclidean Alignment (He & Wu, IEEE TBME 2020; Wu, JNE 2025)

**Core idea:** Align the marginal distributions of EEG data across subjects by whitening each subject's covariance matrix to the identity.

**Algorithm:**
```
For each subject/domain:
    1. Compute reference covariance matrix:
       R_bar = (1/N) * sum_n(X_n @ X_n.T)
    2. Compute transformation matrix:
       T = R_bar^(-1/2)  (matrix square root inverse)
    3. Align all trials:
       X_tilde_n = T @ X_n,  for all n
```

After alignment, the mean covariance of each subject equals the identity matrix I.

**Properties:**
- Unsupervised (no labels needed)
- Two closed-form formulas, extremely efficient
- Flexible — works with any downstream method
- Should be placed BETWEEN temporal filtering and spatial filtering

**Results across 13 BCI paradigms:**
- Motor imagery: +1% to +26% accuracy improvement
- Event-related potential: +36-40% improvement
- Works with both traditional ML and deep learning

**Relevance to our task:**
- We already tried EA in iter018 and iter042 — it HURT performance
- Possible reasons:
  - Our batch-level EA was noisy (batch covariance unreliable)
  - EA on broadband data may need per-subject computation, not per-batch
  - We should compute R_bar over ALL windows of each test subject, then align
- **Critical insight from paper**: EA should be computed over the ENTIRE subject's data, not small batches. With ~3600 windows per test subject, we have plenty of data for robust covariance estimation.

---

## Method 4: SFDA for BCI (Guney et al., JNE 2023)

**Core idea:** Adapt a pre-trained DNN to a new subject using only unlabeled target data, via pseudo-labeling + local-regularity.

**Two-term loss function:**
```
L_total = lambda * L_sl + (1-lambda) * L_ll + beta * ||w||^2
```

1. **Self-Adaptation Loss (L_sl)**: Pseudo-label strategy
   - Use model predictions as pseudo-labels
   - Minimize cross-entropy against pseudo-labels
   - EM-style iterations: predict -> adapt -> predict -> ...

2. **Local-Regularity Loss (L_ll)**: Force similar inputs to get similar predictions
   - Compute correlation between test instances
   - For each instance, find k nearest neighbors
   - Force the model to assign the neighbor's pseudo-label to the instance

3. **Dynamic lambda selection**: Try multiple lambda values, select based on silhouette clustering score

**Relevance to our task:**
- Pseudo-labeling is for classification; not directly applicable to regression
- However, the local-regularity idea IS applicable: similar input EEG windows should produce similar output predictions
- We could use this as a consistency regularizer at test time

---

## Practical Adaptation Strategies for Our Setup

Given our specific setup (46ch input -> 12ch output, regression, ~3600 windows per test subject), here are the most promising approaches ranked by feasibility:

### 1. **Proper Euclidean Alignment** (Easiest, try first)
```python
# At test time, for each test subject:
# Compute covariance over ALL their input windows
X_all = all_windows_for_subject  # shape: (N, C, T)
R_bar = np.mean([x @ x.T for x in X_all], axis=0)  # (C, C)
R_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(R_bar))
# Also compute R_bar for training subjects (pooled or per-subject)
# Align: X_aligned = R_inv_sqrt @ X
```
- Use subject-level covariance (not batch-level)
- Also align training subjects during training
- Place after bandpass filter, before the model

### 2. **Input Statistics Normalization** (Simple)
```python
# Per-subject z-scoring of input channels
# At test time:
for each test subject:
    mu = mean(all_windows, axis=0)  # per-channel mean
    sigma = std(all_windows, axis=0)  # per-channel std
    X_normalized = (X - mu) / sigma
```
- Simpler than EA, addresses amplitude/offset differences
- Should be computed over entire test subject's data

### 3. **Tent-style BN Adaptation** (Requires model changes)
```python
# Add BatchNorm layers to model
# At test time:
model.eval()
# But keep BN in train mode to update running stats
for bn_layer in model.batch_norm_layers:
    bn_layer.train()

# Process all test subject's data in batches
# BN stats automatically adapt to test distribution
for batch in test_subject_batches:
    _ = model(batch)  # forward pass updates BN stats

# Now BN stats reflect test subject's distribution
# Make final predictions
model.eval()
predictions = model(test_data)
```

### 4. **Self-Supervised TTT** (Most complex, highest potential)
```python
# During training, add SSL head:
class Model(nn.Module):
    def __init__(self):
        self.backbone = FIRBackbone()
        self.regression_head = RegressionHead()
        self.ssl_head = SSLHead()  # e.g., predict masked channel

# SSL task: Channel masking prediction
# Randomly zero out one input channel, predict which one
# This is computable on INPUT data alone (no labels needed)

# At test time:
for test_batch in test_subject_data:
    # Create SSL task: mask random channel
    masked_batch, mask_label = random_channel_mask(test_batch)
    ssl_loss = cross_entropy(model.ssl_head(model.backbone(masked_batch)), mask_label)
    ssl_loss.backward()
    optimizer.step()  # update backbone + ssl_head only

# Then predict with adapted backbone
predictions = model.regression_head(model.backbone(test_data))
```

### 5. **Test-Time Consistency Regularization** (Novel for regression)
Since we have a regression task, we can use prediction CONSISTENCY as the adaptation signal:
```python
# Key insight: augmented versions of the same input should produce
# similar outputs. At test time:
for test_batch in test_subject_data:
    # Original prediction
    pred1 = model(test_batch)
    # Augmented prediction (add small noise, shift time, etc.)
    pred2 = model(augment(test_batch))
    # Consistency loss
    consistency_loss = MSE(pred1, pred2)
    consistency_loss.backward()
    # Update only specific layers (BN, or last layer)
    optimizer.step()
```

---

## Key Takeaways

1. **Tent (BN-only adaptation) is the simplest and often most effective** — but requires BN layers in the model, and an adaptation objective suitable for regression.

2. **Euclidean Alignment works when computed properly** — use entire subject's data, not small batches. We should retry with proper subject-level EA.

3. **Self-supervised TTT is powerful but complex** — needs auxiliary SSL tasks designed for EEG. Channel masking and temporal jigsaw are natural choices.

4. **For regression tasks**, entropy minimization doesn't apply directly. Alternatives:
   - Prediction consistency under augmentation
   - Self-supervised auxiliary tasks on input data
   - Input statistics normalization (simplest)
   - Reconstruction-based objectives

5. **Amount of data needed**: With ~3600 windows per test subject, we have MORE than enough for reliable adaptation. Most methods need only 10-100 samples for BN adaptation.

6. **What to adapt**: Papers consistently find that adapting FEWER parameters (just BN) is more robust than adapting the entire model, especially with limited test data.

7. **Our previous EA failure (iter018, iter042)** was likely due to batch-level covariance estimation. Subject-level EA should be retried.

---

## Recommended Iteration Plan

| Priority | Method | Complexity | Expected Impact |
|----------|--------|-----------|----------------|
| 1 | Subject-level EA (proper implementation) | Low | Medium |
| 2 | Per-subject input z-scoring | Very low | Low-Medium |
| 3 | Add BN + Tent-style stats adaptation | Medium | Medium-High |
| 4 | SSL TTT with channel masking | High | High |
| 5 | Consistency regularization at test time | Medium | Medium |

## Sources

- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726) (ICLR 2021)
- [NeuroTTT: Test-Time Training for EEG Foundation Models](https://arxiv.org/abs/2509.26301) (ICLR 2026)
- [MDM-Tent: Multi-Scale Framework with TTA for sEEG](https://arxiv.org/abs/2509.24700) (ICASSP 2026)
- [Source-Free Domain Adaptation for SSVEP-based BCIs](https://arxiv.org/abs/2305.17403) (JNE 2023)
- [Revisiting Euclidean Alignment for EEG Transfer Learning](https://arxiv.org/abs/2502.09203) (JNE 2025)
- [Lightweight SFDA with Adaptive Euclidean Alignment](https://pubmed.ncbi.nlm.nih.gov/39292591/) (IEEE TNSRE 2024)
- [Awesome Test-Time Adaptation List](https://github.com/tim-learn/awesome-test-time-adaptation)
