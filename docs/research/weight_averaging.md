# Weight Averaging Strategies for EEG Model Improvement

## Context

Current best: r = 0.378 (iter030: Mixup + FIR + combined loss + corr val).
Model: SpatioTemporalFIR with 7-tap acausal filter, ~2,268 parameters (27x12x7).
The 0.378 plateau appears to be a fundamental limit of SGD on 1-9 Hz data.
We already tried SWA in iter019 (matched 0.378, did not beat it).

This document evaluates three weight-averaging strategies for breaking through
the plateau.

---

## 1. Model Soup (Multi-Seed Weight Averaging)

### Method
Train the same architecture multiple times with different random seeds, then
average the resulting weights. Three recipes exist:
- **Uniform soup**: Simple average of all model weights.
- **Greedy soup**: Sequentially add models only if validation improves.
- **Learned soup**: Learn per-model coefficients (overkill for our scale).

### Key findings from literature
- Wortsman et al. (2022) showed that fine-tuned models from different seeds
  lie in a single low-error basin, enabling effective weight averaging.
- Even averaging just 2 seeds gave consistent gains of up to 1% on ImageNet.
- Greedy soup always matches or beats the best single model on held-out val.
- **Caveat**: Original results were on large pretrained models (CLIP, ViT).
  Gains were "less substantial" for smaller models trained from scratch.
- Model Stock (ECCV 2024) showed that just 2-3 fine-tuned models suffice
  if they are wisely selected.

### Applicability to our model
- **Pros**: Zero inference cost. Easy to implement. Our model is tiny (2K params)
  so training 5-10 seeds is cheap (~5 min total). Different seeds produce
  different local minima in the loss landscape, and averaging may find a
  flatter region.
- **Cons**: Our model starts from a closed-form (CF) initialization, which
  already places all seeds near the same basin. The diversity between seeds
  may be very low, limiting the benefit. Also, our model is nearly linear --
  weight averaging is exact for linear models (average of linear models = linear
  model of averaged weights), so the "implicit ensemble" effect is weaker.
- **Expected gain**: +0.000 to +0.002. Low diversity from CF init is the
  main concern. Could help if combined with different hyperparams per seed
  (learning rate, dropout rate, mixup alpha).

### Implementation plan
```python
# Train N models with different seeds
models = []
for seed in range(10):
    torch.manual_seed(seed)
    model = train_single(train_ds, val_ds, ...)
    models.append(model)

# Greedy soup
soup = models[0].state_dict()
best_r = validate(soup)
for m in models[1:]:
    candidate = average(soup, m.state_dict())
    r = validate(candidate)
    if r > best_r:
        soup = candidate
        best_r = r
```

---

## 2. Exponential Moving Average (EMA) During Training

### Method
Maintain a shadow copy of model weights that is an exponential moving average
of the training iterates:
```
ema_weights = decay * ema_weights + (1 - decay) * current_weights
```
Use the EMA weights for evaluation/inference.

### Key findings from literature
- Busbridge et al. (2024, TMLR) provided the first systematic study of EMA
  in deep learning. Key results:
  - EMA generalizes better than SGD, on par with SWA.
  - EMA improves robustness to noisy labels, calibration, and consistency.
  - EMA requires less learning rate decay (implicit regularization via averaging).
  - EMA timescale should be between 1 epoch and total training epochs.
- Typical decay values: 0.99-0.999 depending on training length and batch size.
- For our setup (150 epochs, ~50 batches/epoch = ~7500 steps):
  - decay=0.999 -> effective window ~1000 steps (~20 epochs). Reasonable.
  - decay=0.9999 -> effective window ~10000 steps (entire training). Too broad.
  - decay=0.99 -> effective window ~100 steps (~2 epochs). Too narrow.

### Applicability to our model
- **Pros**: Trivially cheap (one extra copy of 2K params). Provides implicit
  regularization that could help with cross-subject generalization. We already
  tried SWA (uniform average of last 50 epochs) and it matched but didn't beat
  best checkpoint. EMA with exponential weighting emphasizes more recent
  (better) iterates, which could be strictly better than SWA's uniform average.
- **Cons**: Our model converges fast (best checkpoint often at epoch 60-100).
  EMA may not help if the model barely moves after convergence. Also, our
  cosine LR schedule already reduces noise in later epochs, partially
  overlapping with EMA's benefit.
- **Expected gain**: +0.000 to +0.002. Marginal improvement possible. The
  SWA result (iter019 = 0.378) suggests weight averaging during training has
  limited headroom.

### Implementation plan
```python
ema_decay = 0.999
ema_state = {k: v.clone() for k, v in model.state_dict().items()}

for epoch in range(1, 151):
    train_one_epoch(...)
    # Update EMA
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k] = ema_decay * ema_state[k] + (1 - ema_decay) * v

# Use EMA weights for inference
model.load_state_dict(ema_state)
```

---

## 3. Subject-Subset Soup (Train on Different Subject Subsets, Average)

### Method
Instead of varying random seeds, vary the *training data*. Train separate
models on different subsets of the 12 training subjects, then average weights.
This creates genuine diversity in the learned spatial filters.

### Variants
- **Leave-K-out soup**: Train on subsets of 10/12 subjects (C(12,2) = 66 combos,
  pick ~10 diverse ones), average weights.
- **Bootstrap soup**: Sample 12 subjects with replacement, train each, average.
- **Domain-specific soup**: Train one model per training subject (12 models),
  greedy-average.

### Key findings from literature
- Cross-subject variability is the main bottleneck (Subject 14 ~0.27, Subject
  13 ~0.46). Models trained on different subject subsets will learn different
  spatial filter compromises.
- MEERNet (emotion recognition) uses per-domain classifiers with shared
  feature extractors, averaged at inference. Similar concept.
- Domain generalization literature shows that averaging models trained on
  different source domains improves out-of-domain performance.

### Applicability to our model
- **Pros**: Creates genuine weight diversity (unlike multi-seed which starts
  from same CF init on same data). Different subject subsets emphasize
  different spatial patterns, and averaging may find a more universal filter.
  Each model is cheap to train. Could combine with greedy selection.
- **Cons**: Training on fewer subjects (10 vs 12) means less data per model.
  The CF initialization is data-dependent, so each model starts at a
  different point -- good for diversity but the CF solutions on subsets
  may be worse individually. Also, our test subjects (13-15) may need
  specific spatial patterns not captured by any subset average.
- **Expected gain**: +0.001 to +0.005. This is the most promising of the
  three strategies because it addresses the actual bottleneck (cross-subject
  variability) rather than just optimizer noise. The diversity is real and
  meaningful.

### Implementation plan
```python
import itertools

# Train on multiple subject subsets
all_subjects = list(range(1, 13))
soups = []
for held_out in itertools.combinations(all_subjects, 2):
    subset = [s for s in all_subjects if s not in held_out]
    train_ds_subset = make_dataset(subset)
    model = train_single(train_ds_subset, val_ds, ...)
    soups.append(model)

# Greedy soup on validation set
final = greedy_soup(soups, val_loader)
```

---

## Recommendation

### Priority order

1. **Subject-Subset Soup** (highest expected impact, +0.001 to +0.005)
   - Addresses the actual bottleneck (cross-subject variability).
   - Creates real diversity in spatial filters.
   - Combine with greedy selection on validation set.
   - Implementation: iter044 or next available iteration.

2. **EMA during training** (+0.000 to +0.002)
   - Cheap to add to any existing training loop.
   - Can combine with subject-subset soup.
   - Try decay=0.999 first, sweep 0.99-0.9999.
   - Could be added as a "free" enhancement to any future iteration.

3. **Multi-Seed Soup** (+0.000 to +0.002)
   - Lowest expected impact due to CF initialization reducing seed diversity.
   - Worth trying with varied hyperparameters instead of just seeds.
   - E.g., soup of: (lr=1e-3, dropout=0.15), (lr=5e-4, dropout=0.20),
     (lr=2e-3, dropout=0.10) -- this is closer to original model soup paper.

### Combined approach (recommended for next iteration)

Train 10 models with different subject subsets, each using EMA (decay=0.999),
then greedy-soup the EMA checkpoints. This stacks all three strategies:
- Data diversity (subject subsets)
- Temporal smoothing (EMA within each run)
- Weight averaging (greedy soup across runs)

Total training time: ~10x single model = ~15 min. Acceptable.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| CF init dominates, averaging just returns ~CF solution | Use longer training (200 epochs) to move further from init |
| Subject-subset models are individually weaker | Greedy soup never performs worse than best single model |
| EMA decay too high/low | Sweep 0.99, 0.999, 0.9999 on validation |
| Overfitting to validation set via greedy selection | Use held-out portion of val set for final selection |
| 10 model runs exceed time budget | Start with 5 models, scale up if promising |

---

## References

- Wortsman et al. (2022). [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482). ICML 2022.
- Jang et al. (2024). [Model Stock: All we need is just a few fine-tuned models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06044.pdf). ECCV 2024.
- Busbridge et al. (2024). [Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits](https://arxiv.org/abs/2411.18704). TMLR 2024.
- Izmailov et al. (2018). [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407). UAI 2018.
- Li et al. (2023). [Stochastic weight averaging enhanced temporal convolution network for EEG-based emotion recognition](https://www.sciencedirect.com/science/article/abs/pii/S1746809423000940). Biomedical Signal Processing and Control.
- Apple ML Research. [How to Scale Your EMA](https://machinelearning.apple.com/research/scale-em).
