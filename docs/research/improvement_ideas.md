# Top Improvement Ideas — Critical Evaluation

## Evaluation Criteria
- **Impact**: Expected gain in mean r
- **Feasibility**: Can implement in <2 hours
- **Risk**: Chance it hurts performance
- **Confidence**: 0-100% it actually improves mean r

---

## PASSED (Top 10 — implementing these)

### 1. Ensemble of CF + Deep Models (Confidence: 90%)
**What**: Average predictions from CF baseline, FIR model, and tiny deep model. Each captures different aspects — CF captures linear spatial mapping, FIR captures temporal dynamics, deep model captures nonlinear interactions.
**Why**: Ensembles almost always help when base models are diverse. CF and deep models make very different errors. Zero implementation risk — just average at inference.
**Downside**: Slower inference (3x). Minor.
**Expected gain**: +0.02-0.04 r

### 2. Optuna HPO Sweep (Confidence: 85%)
**What**: Systematic search over LR (1e-5 to 1e-3), weight decay (1e-4 to 0.1), dropout (0.05-0.3), H (32-128), n_blocks (1-4), loss alpha (0.3-0.7), mixup alpha (0.1-0.8), channel dropout (0.05-0.25). Use TPESampler + MedianPruner.
**Why**: We've been hand-tuning. Even small HP improvements compound. The scaling law showed tiny model is near-optimal — but HP tuning within that architecture could push further.
**Downside**: Compute time (~50 trials × 10 min = 8 hours). Worth it overnight.
**Expected gain**: +0.01-0.03 r

### 3. Residual-from-CF Architecture (Confidence: 75%)
**What**: Instead of skip connection from CF, make CF the PRIMARY prediction and model only learns a small residual correction. Architecture: `output = CF(x) + alpha * model(x)` where alpha is learned (initialized to 0.1). This prevents the model from catastrophically forgetting the strong CF baseline during training.
**Why**: CF is already r=0.577. The deep model should focus compute on what CF misses (nonlinear dynamics, temporal patterns) instead of re-learning spatial mapping.
**Downside**: May limit model's ability to learn fully nonlinear mappings.
**Expected gain**: +0.01-0.02 r

### 4. Flash Attention + SDPA (Confidence: 80%)
**What**: Replace `nn.MultiheadAttention` with `F.scaled_dot_product_attention` which auto-selects FlashAttention-2 on RTX 4060. ~2x faster attention, enabling more training iterations or larger batch sizes in same time.
**Why**: Pure speedup, no accuracy change. Enables more Optuna trials.
**Downside**: None. Drop-in replacement.
**Expected gain**: 2x training speed → enables other improvements

### 5. REVE Foundation Model Fine-tuning (Confidence: 65%)
**What**: Load REVE pretrained encoder (25K subjects), add regression decoder, fine-tune on Ear-SAAD. REVE uses 4D positional encoding that handles arbitrary electrode positions.
**Why**: 25K subjects of pretraining captures EEG universals we can't learn from 15 subjects. This is the most promising path for breaking through the cross-subject barrier.
**Downside**: Architecture mismatch (REVE designed for classification), 200Hz pretrained vs our 128Hz. May need significant adaptation.
**Expected gain**: +0.05-0.15 r (high variance)

### 6. Test-Time Fine-Tuning (Confidence: 60%)
**What**: After training on 12 subjects, fine-tune the last 2 layers on each test subject's data for 10-20 steps using self-supervised objective (reconstruct masked time segments). This adapts the model to subject-specific anatomy without using labels.
**Why**: The cross-subject gap is the main bottleneck. Even unsupervised adaptation could help the model adjust to each subject's unique electrode-brain geometry.
**Downside**: Risk of overfitting to noise if too many steps. Must be carefully tuned.
**Expected gain**: +0.02-0.05 r

### 7. Cross-Attention Channel Decoder (Confidence: 60%)
**What**: Instead of projecting features through a single linear layer, use cross-attention where 12 learnable output queries attend to the 46 input channel features. Each output channel learns which input channels are most relevant for it.
**Why**: Not all scalp channels are equally relevant for each in-ear channel. Left in-ear channels should attend more to left scalp channels. The current architecture doesn't model this explicitly.
**Downside**: More parameters in decoder, potential overfitting.
**Expected gain**: +0.01-0.03 r

### 8. Multi-Task: Predict Around-Ear + In-Ear (Confidence: 55%)
**What**: Use 27 scalp channels as input, predict BOTH 19 around-ear AND 12 in-ear channels. The around-ear prediction acts as auxiliary task that regularizes the shared encoder.
**Why**: Around-ear channels are easier to predict (physically closer to scalp). This gives the encoder more gradient signal and forces it to learn richer spatial representations.
**Downside**: Need to reprocess data with 27 input → 31 output. Changes evaluation setup.
**Expected gain**: +0.01-0.03 r

### 9. Adversarial Subject-Invariant Training (Confidence: 50%)
**What**: Add a subject classifier head with gradient reversal layer. The encoder is trained to FOOL the subject classifier while the main decoder is trained normally. This pushes representations toward subject-invariant features.
**Why**: If the encoder learns subject-specific features, it won't generalize. Gradient reversal is a proven domain adaptation technique.
**Downside**: Tricky to balance adversarial vs main loss. Can destabilize training.
**Expected gain**: +0.01-0.04 r

### 10. Per-Channel Output Heads + Confidence Weighting (Confidence: 55%)
**What**: Instead of one shared output projection, have 12 separate small decoders (one per in-ear channel). Weight the loss per channel by inverse difficulty (channels with higher baseline r get less weight). This focuses compute on hard channels like ELC, ERT.
**Why**: Some channels are 100% NaN for some subjects, others have artifacts. A shared decoder wastes capacity on these. Separate heads + confidence weighting can focus on what's learnable.
**Downside**: 12x more decoder params. But decoders are tiny.
**Expected gain**: +0.01-0.02 r

---

## REJECTED (with reasons)

- **Meta-learning (MAML)**: Too complex for marginal gain. Our inner loop would need many steps.
- **Contrastive pretraining**: Limited by 15 subjects — not enough diversity for contrastive.
- **GNN on electrode graph**: Electrode positions aren't precise enough for graph structure.
- **Frequency-domain prediction**: Adds complexity, unclear benefit for narrowband signals.
- **Progressive window growth**: Complex scheduler for uncertain benefit.
- **Noise2Noise**: Requires careful pairing of trials, architecture changes.
- **Subject-conditioned model**: We don't have subject embeddings at test time.
- **Curriculum learning**: Only 12 training subjects, not enough for meaningful curriculum.
- **Streaming causal**: Different use case, acausal is strictly better for offline.
- **SWA**: Already tried in iter019, marginal improvement.
- **FPN**: Over-engineered for our sequence length (256).
