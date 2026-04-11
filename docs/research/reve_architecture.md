# REVE Architecture Details for Reimplementation

**Paper:** REVE: A Foundation Model for EEG -- Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects  
**Authors:** El Ouahidi et al. (2025), NeurIPS 2025  
**arXiv:** 2510.21585  
**Code:** https://brain-bzh.github.io/reve/

## Overview

REVE is a masked autoencoder (MAE) for EEG with a transformer encoder-decoder. The key innovation is a 4D Fourier positional encoding that handles arbitrary electrode layouts and sequence lengths. All transformer blocks are non-causal (standard encoder-style attention).

---

## 1. Input Tokenization (Patch Embedding)

- **Input:** `X in R^{C x T}` where C = num electrodes, T = num time samples
- **Patching:** Each channel is segmented into temporal patches
  - Window size `w = 1 second` (at 200 Hz = 200 samples per patch)
  - Overlap `o = 0.1 seconds` (= 20 samples overlap)
  - Number of patches: `p = ceil((T - w) / (w - o)) + 1[remainder != 0]`
  - Incomplete patches are discarded (no padding)
- **Embedding:** Each patch `(w samples)` is linearly projected to dimension `D_E`
  - `Xp in R^{C x p x w}` -> `E in R^{C x p x D_E}` via a single Linear layer
- **Minimum input:** 1 second, must be multiples of 1 second

---

## 2. 4D Positional Encoding

### 2a. Spatial Coordinates
- Electrode positions `P in R^{C x 3}` are the 3D (x, y, z) coordinates on the head
- Gaussian noise with `sigma_noise = 0.25 cm` is added during training for robustness

### 2b. Extension to 4D
- Temporal component added: discrete values from 1 to p (number of patches)
- Scaled by factor `s_t` to match spatial scale
- Result: `P_ext in R^{C x p x 4}` with (x, y, z, t) per token

### 2c. 4D Fourier Encoding
- Based on Defossez et al. (2023) 2D approach, extended to 4D
- Each component (x, y, z, t) projected into multi-frequency space using `n_freq` frequencies per dimension
- Frequencies use **Cartesian product structure**: all combinations across 4 dims
- Flattened vector dimension: `n_freq^4`
- Hierarchical periodicity: period of x is `n_freq^1`, y is `n_freq^2`, z is `n_freq^3`, t is `n_freq^4`
- Sine + cosine applied, doubling size: final dim = `2 * n_freq^4`

**n_freq values by model size:**
| Size  | n_freq | 4DPE dim (2 * n_freq^4) | Model dim (D_E) |
|-------|--------|-------------------------|-----------------|
| Small | 4      | 2 * 256 = 512           | 512             |
| Base  | 4      | 2 * 256 = 512           | 512             |
| Large | 5      | 2 * 625 = 1250          | 1250            |

No truncation needed -- dims are matched exactly.

### 2d. Learnable Adjustment
- `P_ext` is also passed through: `Linear -> GELU -> LayerNorm` producing `F_lin in R^{C x p x D_E}`
- Final positional encoding: `P_enc = LayerNorm(F_pe + F_lin)`
- Added to non-masked patch embeddings before encoder (standard absolute PE)

---

## 3. Block Masking Strategy

- **Masking ratio:** M_r = 55%
- **Spatial masking radius:** R_s = 3 cm (masks nearby electrodes)
- **Temporal masking radius:** R_t = 3 seconds (masks contiguous time blocks)
- **Dropout ratio:** D_r = 10% (proportion of masked tokens where entire channel time-series is dropped)
- **Dropout spatial radius:** R_d = 4 cm
- Produces binary mask `B in R^{C x p}`
- Only unmasked tokens are passed to encoder (like original MAE)

---

## 4. Transformer Architecture

### Encoder Configurations

| Size  | Depth | Heads | Dim (D_E) | Params (M) | n_freq |
|-------|-------|-------|-----------|------------|--------|
| Small | 4     | 8     | 512       | 12         | 4      |
| Base  | 22    | 8     | 512       | 69         | 4      |
| Large | 22    | 19    | 1250      | 408        | 5      |

Head dimension: `D_E / n_heads`
- Small/Base: 512 / 8 = 64
- Large: 1250 / 19 ~= 65.8 (approximate)

### Transformer Block Details

**Normalization:** RMSNorm (not LayerNorm) -- better training stability

**Activation:** GEGLU (Gated GELU) in FFN layers -- more expressive gating

**FFN Design:**
- Two-layer structure (Linear -> GEGLU -> Linear)
- Expansion ratio: 8/3 of D_E
- With GLU gating, actual intermediate dim = 2 * (8/3) * D_E = (16/3) * D_E
  - Small/Base: intermediate dim ~= 2731
  - Large: intermediate dim ~= 6667

**Bias terms:** Removed from ALL linear layers except:
- Final decoder projection layer
- (RMSNorm has no bias by definition)

**Attention:** Flash Attention v2 for efficiency. Standard multi-head self-attention, non-causal.

**Pre-norm architecture** (standard transformer with normalization before attention/FFN).

### Decoder

- Lighter/smaller than encoder (following MAE convention)
- Same transformer block design (RMSNorm, GEGLU, Flash Attention v2)
- Exact decoder depth/width not specified in paper (likely 2-4 layers, typical for MAE)
- Decoder input: encoder output for visible tokens + learned mask tokens, both with positional encodings re-added
- **Same positional encoding** reused for both encoder and decoder (unlike original MAE which uses separate)
- Final linear projection maps latent patches back to signal space (w samples)

---

## 5. Pretraining Objective

### Primary Task: Masked EEG Reconstruction
- Reconstruct raw EEG of masked patches from visible patches
- **Loss:** L1 (not L2) -- more robust to EEG noise/outliers
- `L = (1/|P_m|) * sum_i ||P_hat_m^i - P_m^i||_1`

### Secondary Task: Global Token Reconstruction
- Attention pooling across outputs of ALL MHA layers in encoder:
  1. Extract output tokens (post-FFN) from each MHA block
  2. Concatenate them as keys and values
  3. A **single learned query token** attends to all concatenated outputs
  4. This pooled token is repeated N_m times (one per masked token)
  5. Positional encodings added for each masked position
  6. Passed through a **2-layer FFN** to reconstruct masked patches
- Also uses L1 loss

### Total Loss
```
Loss = Primary_Loss + lambda * Secondary_Loss
lambda = 0.1
```

The secondary loss forces the encoder to distribute useful information across ALL layers (not just the final one), improving linear probing quality.

---

## 6. Training Details

- **Optimizer:** StableAdamW with Adafactor-style gradient clipping
- **Scheduler:** Warmup Stable Decay (trapezoidal)
  - 10% warmup, 80% at peak LR, linear decay to 1% of peak
  - Cyclic trapezoidal across epochs
- **Peak LR:** 2.4e-4 (for Small model)
- **LR scaling:** eta proportional to D^(-0.90) for larger models
- **Betas:** (0.9, 0.95)
- **Epsilon:** 1e-9
- **Batch size:** 4096
- **Initialization:** Megatron-style, sigma = 0.02 for all transformer layers and mask token
- **Precision:** Half precision (FP16/BF16)
- **Data preprocessing:**
  - Resampled to 200 Hz
  - Band-pass filtered 0.5-99.5 Hz
  - Z-score normalization per recording session
  - Clip values exceeding 15 standard deviations

### Compute
- ~260 A100 GPU hours for Base model pretraining
- Peak throughput: 312 TFLOPs at half precision
- MFU: 50%

---

## 7. Fine-tuning Procedure

### Two-Step Strategy (single continuous training run)
1. **Linear probe phase:** Encoder frozen, only classification head trains
2. **Full fine-tune phase:** Encoder unfrozen, entire network trains

### LoRA Integration
- Applied to attention blocks: Q, K, V, and O projection layers
- Low-rank matrices for parameter-efficient adaptation

### Regularization
- Dropout
- Mixup data augmentation
- Warmup + Reduce-on-Plateau LR scheduler

### Model Souping
- Average weights of multiple fine-tuning runs
- ~1.5% gain when combining >= 5 models
- Works for Base and Large, limited benefit for Small

---

## 8. Key Design Decisions for Our Use Case

### What matters for scalp-to-in-ear EEG prediction:

1. **4D PE is critical** -- handles arbitrary electrode positions via 3D coordinates + temporal index. This means we can encode both scalp and in-ear electrode positions naturally.

2. **Patch size = 1 second at 200 Hz** -- for our 20 Hz data, patches would be 20 samples. We may want to adjust.

3. **No regression tasks evaluated** -- all downstream tasks are classification. The encoder produces embeddings that would need a regression head for our correlation prediction task.

4. **Variable channel count** is handled by the 4D PE -- no fixed channel ordering needed. Tokens are (channel, time_patch) pairs with positional encoding from 3D coordinates.

5. **Pretrained weights available** -- could potentially use REVE-Base (69M params) as a frozen feature extractor and train a lightweight regression head on top for our scalp-to-in-ear mapping.

6. **The secondary loss / global token** provides a compact representation suitable for downstream tasks with frozen backbone -- relevant if we use pretrained weights.

### Potential adaptation strategy:
- Load REVE encoder with pretrained weights
- Feed 27 scalp channels as input (with their 3D positions)
- Extract per-channel, per-patch embeddings from encoder
- Train a lightweight decoder/regression head to predict 12 in-ear channels
- The 4D PE naturally handles the different spatial positions of scalp vs in-ear electrodes

---

## 9. Comparison with BIOT (iter040)

REVE improves on BIOT (which we tried in iter040) in several ways:
- 4D Fourier PE vs BIOT's learned absolute PE (better generalization)
- RMSNorm + GEGLU vs standard LayerNorm + GELU
- Secondary loss for better frozen representations
- Block masking vs random masking
- Much larger pretraining corpus (60K hours vs ~2K)
- L1 loss vs L2 for robustness to EEG noise
