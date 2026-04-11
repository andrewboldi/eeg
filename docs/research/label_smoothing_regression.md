# Label Smoothing / Target Regularization for Regression

## Problem Statement

Our in-ear EEG targets are inherently noisy due to poor electrode contact (especially on hard subjects like Subject 14, r~0.27). Standard MSE training treats these noisy targets as ground truth, potentially overfitting to measurement noise rather than learning the true scalp-to-in-ear mapping.

Three candidate techniques are evaluated below.

---

## 1. Target Noise Injection (DisturbValue)

**Concept**: Add Gaussian noise to target values during training, analogous to label smoothing for classification.

**Key paper**: Alla & Szegedy, "Disturbing Target Values for Neural Network Regularization" ([arXiv:2110.05003](https://arxiv.org/abs/2110.05003))

- **DisturbValue (DV)**: Inject Gaussian noise N(0, sigma) to a random fraction p of target values each batch.
- **DisturbError (DE)**: Only disturb targets where the model is confident (low error), leaving uncertain predictions alone.
- Validated on 8 regression datasets; competitive with or outperforms L2 regularization and Dropout.

**Theoretical basis**: Bishop (1995) showed training with noise on targets is equivalent to Tikhonov regularization with an extra penalty term ([Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf)). The noise level sigma controls the implicit regularization strength.

**Applicability to our problem**:
- **HIGH relevance**. Our targets have genuine measurement noise (poor electrode contact, motion artifacts). Adding controlled noise during training would prevent the FIR model from fitting to per-sample noise patterns.
- Easy to implement: just add `y_noisy = y + sigma * torch.randn_like(y)` during training.
- Hyperparameter: sigma should be proportional to target noise level. For our data (targets roughly unit-variance after preprocessing), try sigma in {0.01, 0.05, 0.1}.
- Risk: too much noise degrades signal; too little has no effect. Unlike classification label smoothing, there is no natural "uniform distribution" to mix toward.

**Implementation sketch**:
```python
# In training loop, after loading batch:
if target_noise_sigma > 0:
    noise = target_noise_sigma * torch.randn_like(inear)
    inear_train = inear + noise
# Use inear_train for loss, but inear for correlation validation
```

**Expected impact**: +0.001 to +0.003 r. Modest because our model is already simple (7-tap FIR) and unlikely to severely overfit to target noise. Most benefit would come from hard subjects where target noise is highest.

---

## 2. Teacher-Student Distillation with CF Soft Targets

**Concept**: Use the closed-form (CF) linear model as a "teacher" to provide soft regression targets. Train the FIR student on a weighted mix of ground truth and teacher predictions.

**Key references**:
- Hinton et al., "Distilling the Knowledge in a Neural Network" ([arXiv:1503.02531](https://arxiv.org/pdf/1503.02531)) -- foundational work on soft targets
- "Knowledge distillation with insufficient training data for regression" ([ScienceDirect, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0952197624001593)) -- regression-specific distillation achieving 43.9% RMSE reduction with limited data

**How it works for regression**:
- Teacher provides predictions y_teacher = CF(x)
- Student loss = alpha * MSE(y_pred, y_true) + (1-alpha) * MSE(y_pred, y_teacher)
- The teacher's predictions are "denoised" versions of the targets -- they capture the learned linear mapping without per-sample noise.

**Applicability to our problem**:
- **MEDIUM relevance**. The CF model and the FIR model learn very similar mappings (the center tap of FIR converges to CF weights). The "dark knowledge" from CF is limited because it's a simpler model than the student.
- However, CF predictions ARE smoother than raw in-ear recordings, so they could act as denoised targets.
- For hard subjects, CF predictions may be more reliable than the noisy ground truth.
- Could combine: use CF predictions as soft targets only for channels/subjects where ground truth is noisy.

**Implementation sketch**:
```python
# Before training: get CF predictions on training data
cf = ClosedFormLinear(C_scalp, C_inear)
cf.fit(train_ds.scalp.numpy(), train_ds.inear.numpy())
with torch.no_grad():
    cf_preds = cf(train_ds.scalp.to(device))  # (N, C_inear, T)

# In training loop:
soft_target = alpha * inear + (1 - alpha) * cf_preds[batch_idx]
loss = loss_fn(pred, soft_target)
```

**Expected impact**: +0.000 to +0.002 r. Limited because teacher and student are so similar. The CF model captures ~96% of what FIR learns. The remaining 4% is exactly the temporal dynamics FIR adds, which CF cannot teach.

**Variant -- Multi-subject teacher**: Train per-subject CF models, then use ensemble of per-subject predictions as soft targets. This could help cross-subject transfer by smoothing subject-specific noise.

---

## 3. Target Mixup (Temporal Neighbor Smoothing)

**Concept**: Instead of standard mixup (which mixes random samples), mix neighboring time windows' targets to smooth temporal discontinuities.

**Key references**:
- Zhang et al., "mixup: Beyond Empirical Risk Minimization" -- original mixup paper
- "Enhanced mixup for improved time series analysis" ([IJAIN](https://ijain.org/index.php/IJAIN/article/view/1592/0)) -- time-series-specific mixup strategies
- "Diffusion models-based motor imagery EEG sample augmentation via mixup strategy" ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417024024527)) -- EEG-specific temporal/spatial mixup

**How it works**:
- Standard mixup (already in iter030): mix random pairs of (x, y) with lambda ~ Beta(alpha, alpha)
- **Temporal neighbor mixup**: mix adjacent or nearby windows specifically, exploiting temporal continuity of EEG
- **Target-only smoothing**: apply a small moving average to targets across time windows within a subject

**Applicability to our problem**:
- **LOW-MEDIUM relevance**. We already use standard mixup in iter030 (alpha=0.4), which achieved the current best r=0.378. Temporal neighbor mixup is a refinement.
- Our 2-second windows at 20 Hz are already fairly smooth. Adjacent windows overlap significantly if using sliding windows.
- Risk: over-smoothing targets removes real high-frequency dynamics.

**Implementation sketch (temporal neighbor mixup)**:
```python
# During data loading, for each subject's windows:
# Sort windows by time, then mix adjacent pairs
for i in range(0, len(windows) - 1, 2):
    lam = np.random.beta(0.2, 0.2)  # lighter mixing
    x_mix = lam * x[i] + (1-lam) * x[i+1]
    y_mix = lam * y[i] + (1-lam) * y[i+1]
```

**Expected impact**: +0.000 to +0.001 r. Marginal over standard mixup. The temporal structure is already smooth at 1-9 Hz.

---

## Recommendation: Priority Ordering

| Rank | Method | Expected r gain | Effort | Risk |
|------|--------|----------------|--------|------|
| 1 | **Target noise (DisturbValue)** | +0.001 to +0.003 | Low (5 lines) | Low |
| 2 | **CF soft targets** | +0.000 to +0.002 | Medium (20 lines) | Low |
| 3 | **Temporal neighbor mixup** | +0.000 to +0.001 | Medium (15 lines) | Medium (over-smoothing) |

### Suggested Iteration Plan

**iter045_target_noise**: Add DisturbValue to the iter030 recipe. Sweep sigma in {0.02, 0.05, 0.1}. This is the simplest and most theoretically grounded approach. Can be combined with existing mixup.

**iter046_cf_soft_targets**: Use CF predictions as soft targets with alpha=0.8 (80% real, 20% CF). Test whether denoised targets help on hard subjects (14, 15).

**iter047_combined**: If either helps, combine target noise + CF soft targets + mixup.

### Important Caveats

1. **Our model is already very simple** (7-tap FIR, ~2K parameters). Target regularization helps most when models can memorize noise, which is less of a risk with such a constrained model.
2. **The 0.378 plateau** may be a fundamental limit of the 1-9 Hz narrowband data, not a regularization problem. Target smoothing cannot recover information that was filtered out.
3. **Validation must use clean targets** -- never add noise or smooth the validation/test targets.
4. **Per-subject analysis** is critical: check if gains come from hard subjects (14) or are uniform.

---

## Sources

- [DisturbValue: Disturbing Target Values for Neural Network Regularization](https://arxiv.org/abs/2110.05003)
- [Training with Noise is Equivalent to Tikhonov Regularization (Bishop, 1995)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf)
- [Hinton et al. -- Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531)
- [Knowledge distillation with insufficient training data for regression (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0952197624001593)
- [Improving Time Series Classification with Representation Soft Label Smoothing](https://arxiv.org/abs/2408.17010)
- [Does Label Smoothing Mitigate Label Noise? (ICML 2020)](https://proceedings.mlr.press/v119/lukasik20a/lukasik20a.pdf)
- [Enhanced mixup for improved time series analysis](https://ijain.org/index.php/IJAIN/article/view/1592/0)
- [ICLR 2025: Towards Understanding Why Label Smoothing Degrades Selective Classification](https://proceedings.iclr.cc/paper_files/paper/2025/file/9dc5accb1e4f4a9798eae145f2e4869b-Paper-Conference.pdf)
- [Train Neural Networks With Noise to Reduce Overfitting](https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/)
- [Diffusion models-based EEG sample augmentation via mixup strategy](https://www.sciencedirect.com/science/article/abs/pii/S0957417024024527)
