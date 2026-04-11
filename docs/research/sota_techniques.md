# State-of-the-Art ML Techniques for EEG Signal Prediction

**Project context**: Predicting 12 in-ear EEG channels from 46 input channels (27 scalp + 19 around-ear), 128 Hz, 2s windows (256 samples), 15 subjects LOSO.
**Current best**: Closed-form linear r=0.577, deep transformer r~0.68 (val). Large val-to-test gap (overfitting).
**Goal**: Break r=0.8 on held-out test subjects.

---

## Ranked Recommendations

### 1. SUBJECT-ADAPTIVE TEST-TIME ALIGNMENT (Priority: CRITICAL)

**Why**: Cross-subject variability is our #1 bottleneck (Subject 14 always scores ~0.27 vs Subject 13 at ~0.46 in narrowband). No model improvement will matter if we can't handle inter-subject distribution shift.

**Concrete techniques**:

- **Euclidean Alignment (EA)**: Compute per-subject covariance matrix, whiten to identity before feeding to model. A 2024 systematic evaluation (arXiv:2401.10746) showed EA consistently improves deep learning EEG decoders by 2-5% across multiple architectures. Our earlier attempt (iter018) failed because we used per-batch covariance -- must use per-subject full-session covariance instead.

- **Reversible Instance Normalization (RevIN)**: Unlike standard InstanceNorm (which we proved hurts in iter020), RevIN normalizes input, passes through the model, then denormalizes output. This preserves amplitude dynamics while reducing subject shift. Published in Nature Scientific Reports 2025 for EEG artifact removal.

- **Test-Time Adaptation (TTA) via BN statistics**: At test time, update BatchNorm running stats using the test subject's data (unlabeled). A 2024 BCI Winter Conference paper showed +4.9% accuracy on SSVEP and +3.6% on MI datasets. This is a free lunch -- no retraining needed.

- **Dual-Stage Alignment**: Apply EA to raw input, then adapt BN statistics in feature space. Recent 2025 work (arXiv:2509.19403) combines both stages as a plug-and-play module.

**Implementation plan**:
```python
# Per-subject Euclidean alignment
def euclidean_align(X_subject):
    """X: (n_windows, C, T) for one subject"""
    # Compute spatial covariance across all windows
    R = np.mean([x @ x.T / x.shape[1] for x in X_subject], axis=0)
    # Whitening matrix
    W = scipy.linalg.inv(scipy.linalg.sqrtm(R))
    return np.array([W @ x for x in X_subject])

# RevIN wrapper
class RevIN(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._mean = x.mean(dim=-1, keepdim=True).detach()
            self._std = x.std(dim=-1, keepdim=True).detach() + self.eps
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            x = (x - self.affine_bias) / self.affine_weight
            x = x * self._std + self._mean
        return x
```

**Expected impact**: +0.03 to +0.08 r. This should be the very first thing we implement.

**Sources**:
- [Systematic Evaluation of EA with Deep Learning for EEG](https://arxiv.org/abs/2401.10746)
- [Revisiting EA for Transfer Learning in EEG-based BCIs](https://arxiv.org/abs/2502.09203)
- [Online Adaptation via Dual-Stage Alignment](https://arxiv.org/abs/2509.19403)
- [ReVIN for EEG artifact removal](https://www.nature.com/articles/s41598-025-28855-0)

---

### 2. OPTUNA HYPERPARAMETER OPTIMIZATION (Priority: HIGH)

**Why**: We have been hand-tuning hyperparameters. With broadband 128 Hz data and a deep model, the search space is large (LR, weight decay, hidden dim, kernel sizes, dropout, loss weights, number of layers). Systematic optimization could easily find +0.02-0.05 r.

**Concrete setup**:

```python
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

def objective(trial):
    # Architecture
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    n_layers = trial.suggest_int('n_layers', 2, 6)
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    kernel_sizes = trial.suggest_categorical('kernels', ['3,7,15', '3,7,15,31', '5,11,21'])

    # Training
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    corr_weight = trial.suggest_float('corr_weight', 0.0, 0.5)

    model = build_model(hidden_dim, n_layers, n_heads, dropout, kernel_sizes)
    # ... train loop ...

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(...)
        val_r = validate(...)
        trial.report(val_r, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_r

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(n_startup_trials=10),
    pruner=HyperbandPruner(min_resource=10, max_resource=200, reduction_factor=3),
)
study.optimize(objective, n_trials=100, timeout=3600 * 4)  # 4 hour budget
```

**Best practices**:
- Use **TPESampler** (default) with **HyperbandPruner** -- best combination per Optuna docs
- Set `n_startup_trials=10` for TPE to have enough random trials before Bayesian optimization kicks in
- Report validation correlation every epoch; Hyperband will aggressively prune bad configs
- Use `log=True` for learning rate and weight decay (log-uniform distribution)
- Run ~50-100 trials; diminishing returns beyond that for our model size
- Save study to SQLite for persistence: `optuna.create_study(storage='sqlite:///optuna.db')`

**Expected impact**: +0.02 to +0.05 r from better hyperparameters alone.

**Sources**:
- [Optuna Efficient Optimization Algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [HyperbandPruner docs](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html)

---

### 3. LEARNING RATE SCHEDULING: OneCycleLR (Priority: HIGH)

**Why**: We have been using fixed LR or simple cosine annealing. OneCycleLR (super-convergence) is consistently the best scheduler for training from scratch on small-to-medium datasets. It warms up, peaks, then decays aggressively.

**Concrete implementation**:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,           # Find via LR range test
    epochs=200,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,         # 30% warmup
    anneal_strategy='cos',  # Cosine decay (default, best)
    div_factor=25,          # initial_lr = max_lr / 25
    final_div_factor=1e4,   # min_lr = initial_lr / 10000
)

# CRITICAL: step per BATCH, not per epoch
for batch in train_loader:
    loss = ...
    loss.backward()
    optimizer.step()
    scheduler.step()  # After every batch
```

**Key details**:
- `pct_start=0.3`: 30% warmup is a good default. For small datasets, try 0.2-0.4.
- Use LR range test first: train for 1 epoch with LR increasing from 1e-6 to 1e-1, pick max_lr where loss is still decreasing rapidly (typically 1e-3 to 5e-3 for our model sizes).
- OneCycleLR also manages momentum inversely (high momentum during low LR, low momentum during high LR).
- This replaces separate warmup + cosine scheduling with a single, better-tuned policy.

**Expected impact**: +0.01 to +0.02 r from faster, more stable convergence.

**Sources**:
- [PyTorch OneCycleLR docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
- [Visual Guide to LR Schedulers in PyTorch](https://www.leoniemonigatti.com/blog/pytorch-learning-rate-schedulers.html)

---

### 4. TRANSFORMER WITH FLASH ATTENTION (Priority: HIGH)

**Why**: Our RTX 4060 (Ada architecture) natively supports FlashAttention-2 via PyTorch's `scaled_dot_product_attention`. At sequence length 256, transformers are actually optimal -- Mamba only wins at longer sequences.

**Key facts**:
- `torch.nn.functional.scaled_dot_product_attention` automatically selects the best backend (FlashAttention, Memory-Efficient, or Math) based on GPU and input shape.
- RTX 4060 = Ada Lovelace = **fully supported** for FlashAttention-2.
- For seq_len=256, attention is O(256^2) = 65K operations per head -- trivial on GPU. No need for linear attention alternatives.
- PyTorch 2.2+ required (we should already have this).

**Architecture recommendation** (based on EEGMamba and EEGPT research):

```python
class EEGTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        # Self-attention with Flash Attention
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # This auto-selects FlashAttention on RTX 4060
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(attn_out))
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x
```

**Mamba/SSM verdict**: Skip for now. At seq_len=256, transformers are faster to train and match or beat Mamba. Mamba's advantage only appears at seq_len > 1024. If we later move to longer windows (e.g., 10s = 1280 samples), revisit Mamba.

**Expected impact**: Architecture itself is similar to what we have. The win is from Flash Attention making training 2x faster, enabling more Optuna trials.

**Sources**:
- [PyTorch SDPA docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)

---

### 5. EEG FOUNDATION MODEL FINE-TUNING (Priority: MEDIUM-HIGH)

**Why**: Pre-trained EEG models have learned universal spatial-temporal representations from thousands of hours of EEG. Fine-tuning could provide much better initialization than our closed-form linear start.

**Top candidates (ranked)**:

| Model | Params | Pre-training Data | Regression? | Channel Flexibility | Availability |
|-------|--------|-------------------|-------------|---------------------|-------------|
| **LaBraM** | 5.8M-369M | 2,500h / 20 datasets | Yes (gait prediction) | Yes (channel patches) | [GitHub](https://github.com/935963004/LaBraM) |
| **EEGPT** | 10M | Large multi-task | Yes (linear probing) | Yes (hierarchical spatial-temporal) | [GitHub](https://github.com/BINE022/EEGPT) |
| **FEMBA** | ~10M | 21,000h | Reconstruction | Yes (patch-based) | [arXiv](https://arxiv.org/abs/2502.06438) |
| **BIOT** | ~5M | Multi-biosignal | Classification mainly | Per-channel tokenization | Limited |

**Recommendation: Start with LaBraM-base (5.8M params)**:
- ICLR 2024 spotlight, well-maintained code
- Handles variable channel counts via channel-patch tokenization
- Has been fine-tuned for regression (gait prediction)
- 5.8M params fits easily on RTX 4060
- Replace classification head with regression head (12 output channels x 256 time points)

**Caveat**: These models are pre-trained on classification tasks and standard 10-20 montage EEG. Our task (scalp-to-inear regression) is quite different. The spatial representations may not transfer well. Worth trying but don't expect miracles.

**Expected impact**: +0.02 to +0.06 r if spatial representations transfer; +0.00 if they don't.

**Sources**:
- [LaBraM - ICLR 2024](https://github.com/935963004/LaBraM)
- [EEGPT - NeurIPS 2024](https://github.com/BINE022/EEGPT)
- [FEMBA - 2025](https://arxiv.org/abs/2502.06438)

---

### 6. KAN (KOLMOGOROV-ARNOLD NETWORKS) FOR CHANNEL MIXING (Priority: MEDIUM)

**Why**: KANs replace fixed activations (ReLU/GELU) with learnable spline-based activation functions on edges. For our problem, the scalp-to-inear mapping has nonlinear components that a standard linear layer might miss. A KAN layer for channel mixing could capture nonlinear spatial relationships.

**Practical assessment**:
- KAN was published at ICLR 2025 (main conference), so it's real, not hype
- For time series: KAN-AD achieved 15% improvement over baselines with <1000 params
- KAN has faster neural scaling laws than MLPs -- important for our small dataset
- B-spline basis is natural for smooth signal transformations (which EEG volume conduction is)
- **BUT**: Training is slower than MLP (spline evaluation overhead), and PyTorch implementations are less mature

**Recommendation**: Use KAN as a drop-in replacement for the spatial mixing layer only (46 -> 12 channels), not for the full temporal model. This keeps the parameter count low and targets the part of the model where nonlinearity matters most.

```python
# pip install pykan  OR  use efficient-kan
from efficient_kan import KANLinear

class SpatialKAN(nn.Module):
    def __init__(self, C_in, C_out, grid_size=5):
        super().__init__()
        # KAN layer for spatial mixing: learns nonlinear channel combinations
        self.kan = KANLinear(C_in, C_out, grid_size=grid_size)

    def forward(self, x):
        # x: (B, C_in, T)
        B, C, T = x.shape
        x = x.permute(0, 2, 1)  # (B, T, C_in)
        x = self.kan(x)          # (B, T, C_out) -- learnable nonlinear mixing
        return x.permute(0, 2, 1) # (B, C_out, T)
```

**Expected impact**: +0.01 to +0.03 r if nonlinear spatial mixing helps; +0.00 if the mapping is truly linear.

**Sources**:
- [KAN: Kolmogorov-Arnold Networks (ICLR 2025)](https://arxiv.org/abs/2404.19756)
- [KANs for Time Series Analysis](https://arxiv.org/abs/2405.08790)
- [KAN-AD: Time Series Anomaly Detection](https://arxiv.org/abs/2411.00278)

---

### 7. MAMBA / STATE-SPACE MODELS (Priority: LOW for current setup)

**Why it's low priority**: At 256 samples (2s at 128 Hz), transformers are better or equal. Mamba's linear scaling advantage only kicks in at seq_len > 1024.

**When to revisit**: If we move to longer windows (5-10s = 640-1280 samples) or streaming inference.

**Key EEG-Mamba papers for future reference**:
- **EEGMamba** (2024): Bidirectional Mamba + MoE, SOTA on seizure/emotion/sleep/MI classification
- **FEMBA** (2025): Bidirectional Mamba foundation model, 21,000h pre-training, linear scaling
- **MI-Mamba** (2025): 6x fewer params than transformers for motor imagery

**If we do try Mamba later**:
```bash
pip install mamba-ssm  # Requires CUDA, works on RTX 4060 (Ada)
```

```python
from mamba_ssm import Mamba

class EEGMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return x + self.mamba(self.norm(x))
```

**Sources**:
- [EEGMamba](https://arxiv.org/abs/2407.20254)
- [FEMBA](https://arxiv.org/abs/2502.06438)
- [Mamba GitHub](https://github.com/state-spaces/mamba)

---

## Concrete Iteration Plan to Break r=0.8

Based on the research above, here is the recommended iteration sequence:

| Iter | Technique | Expected r | Rationale |
|------|-----------|------------|-----------|
| 040 | Per-subject Euclidean Alignment (full-session cov) | 0.60-0.62 | Fix the #1 bottleneck: subject variability |
| 041 | RevIN wrapper around deep model | 0.62-0.64 | Preserve amplitude dynamics while reducing shift |
| 042 | OneCycleLR + Optuna (20 trials) | 0.65-0.68 | Systematic HP search on broadband |
| 043 | Transformer with Flash Attention + tuned HPs | 0.68-0.72 | Proper attention architecture |
| 044 | Test-Time BN adaptation | 0.70-0.74 | Free lunch for test subjects |
| 045 | LaBraM fine-tuning | 0.72-0.76 | Pre-trained spatial representations |
| 046 | KAN spatial mixing layer | 0.73-0.77 | Nonlinear channel combinations |
| 047 | Full Optuna sweep (100 trials) | 0.75-0.80 | Polish everything |

**Key insight**: The path to r=0.8 is NOT about finding a magical architecture. It's about:
1. **Fixing cross-subject alignment** (biggest gap)
2. **Systematic hyperparameter optimization** (we've been guessing)
3. **Using the right training recipe** (OneCycleLR, proper warmup, correct batch size)
4. **Leveraging pre-trained representations** (foundation models)

The architecture (transformer vs conv vs Mamba) matters much less than these four factors.

---

## What NOT to Pursue

- **Standard InstanceNorm**: Proven harmful in iter020 -- removes useful amplitude dynamics
- **Pure correlation loss**: Produces degenerate scale (iter031)
- **Causal-only temporal filters**: Acausal is critical for our offline prediction task (iter034-035)
- **Mamba for 256-length sequences**: No advantage over transformers at this length
- **Very deep networks (>10 layers)**: Dataset too small (15 subjects); will overfit
- **Contrastive pre-training from scratch**: Not enough data; use existing foundation models instead
