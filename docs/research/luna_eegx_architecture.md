# Channel-Agnostic EEG Foundation Models: LUNA and EEG-X

Architecture details extracted from arXiv papers for reimplementation.
TeX sources stored in `docs/external/luna/` and `docs/external/eeg_x/`.

## LUNA (arXiv:2510.22257) — NeurIPS 2025

**Core idea**: Compress variable-channel EEG into a fixed-size latent space via learned queries + cross-attention, then process temporally. Computation scales linearly with channel count.

### Architecture Overview

```
Input: x in R^{B x C x T}  (B=batch, C=channels, T=time)
  |
  v
[Patch Feature Extraction]  -- segment each channel into S = T/P patches
  |-- Temporal: 1D Conv (GroupNorm + GELU), 3 layers
  |-- Frequency: FFT magnitude+phase -> MLP
  |-- Sum both -> x_features in R^{B x (C*S) x E}
  |
  v
[Channel Positional Encoding]  -- NeRF-style sinusoidal on 3D coordinates + MLP
  |-- E_pos in R^{B x C x E}, added to x_features
  |
  v
[Channel-Unification Module]  -- THE KEY INNOVATION
  |-- Q learned queries Q_learn in R^{Q x E} (orthogonal init, no batch dim)
  |-- Reshape x_features to R^{(B*S) x C x E}
  |-- Cross-attention: A_out = MultiHeadAttn(Q_tilde, X', X')
  |     where Q_tilde in R^{(B*S) x Q x E} (queries repeated)
  |-- FFN with residual: A_out + FFN(A_out)
  |-- L Transformer encoder layers on query dim Q
  |-- Output: X_unified in R^{(B*S) x Q x E}
  |
  v
[Patch-wise Temporal Encoder]
  |-- Reshape to R^{B x S x (Q*E)}
  |-- Transformer encoder blocks with RoPE
  |-- Sequence length = S (NOT S*C) -> massive speedup
  |-- Output: E_out in R^{B x S x (Q*E)}
  |
  v
[Decoder] (task-dependent)
  |-- Reconstruction: C decoder queries cross-attend to E_out -> linear -> R^P per channel
  |-- Classification: 1 aggregation query cross-attends to E_out -> MLP
```

### Hyperparameters (3 scales)

| Parameter | Base | Large | Huge |
|-----------|------|-------|------|
| Temporal Conv channels | {1,8,8}->{16,16,16} | {1,16,16}->{24,24,24} | {1,32,32}->{32,32,32} |
| Conv kernel sizes | {20, 3, 3} | {20, 3, 3} | {20, 3, 3} |
| Conv strides | {10, 1, 1} | {10, 1, 1} | {10, 1, 1} |
| Patch size P | 40 | 40 | 40 |
| Num queries Q | 4 | 6 | 8 |
| Query size E | 64 | 96 | 128 |
| Hidden size | 256 | 576 | 1024 |
| MLP size | 1024 | 2304 | 4096 |
| Attention heads | 8 | 12 | 16 |
| Transformer layers | 8 | 10 | 24 |
| Params (approx) | ~7M | ~40M | ~200M |

### Training Details

- **Pretraining data**: TUEG + Siena (21,000+ hours, 20-29 channels, bipolar+unipolar)
- **Objective**: Masked patch reconstruction (50% mask ratio)
- **Loss**: Smooth-L1 on masked patches + alpha*Smooth-L1 on visible (alpha=0.05)
- **Auxiliary loss**: Query specialization (lambda=0.8) -- penalizes query-channel affinity overlap
- **Optimizer**: AdamW, beta=(0.9, 0.98), weight decay=0.05
- **LR**: 1.25e-4 peak, cosine to 2.5e-7, 10 warmup epochs, 60 total
- **Precision**: bf16-mixed
- **Hardware**: 8x A100 (Base/Large), 16x A100 (Huge), ~1 day pretraining

### Channel Positional Encoding (NeRF-style)

Uses normalized 3D electrode coordinates -> sinusoidal NeRF encoding -> MLP projection to R^E.
This allows ANY electrode position (including non-standard like around-ear) to get a meaningful embedding.

### Complexity Analysis

| Stage | Complexity |
|-------|-----------|
| Channel-Unification (cross-attn) | O(B * S * Q * C * E) -- linear in C |
| Query self-attention | O(B * S * Q^2 * E) -- independent of C |
| Temporal encoder (self-attn) | O(B * S^2 * Q * E) -- independent of C |

vs. Full attention (LaBraM): O(B * S^2 * C^2 * E) -- quadratic in both S and C

### Q vs E Tradeoff (Base, fixed Q*E=256)

| Q | E | TUAB AUROC | TUAR AUROC |
|---|---|------------|------------|
| 4 | 64 | 0.887 | 0.902 |
| 2 | 128 | 0.885 | 0.885 |
| 8 | 32 | 0.884 | 0.899 |
| 16 | 16 | 0.874 | 0.866 |

Best: Q=4, E=64. Too many queries with small E degrades performance.

### Code

GitHub: https://github.com/pulp-bio/BioFoundation

---

## EEG-X (arXiv:2511.08861) — ICLR 2026 submission

**Core idea**: Location-based sinusoidal channel embeddings from 2D scalp coordinates + noise-aware masked reconstruction targeting ICA-cleaned signals + DiCT loss.

### Architecture Overview

```
Input: X = {x_1, ..., x_C}, each x_c in R^L
  |
  v
[Signal Embedding / Tokenizer]
  |-- Patchify: window size w=128 (1s at 128Hz), overlap o=32
  |-- STFT magnitude of each patch
  |-- Linear projection: e_{c,i} = W * |STFT(t_{c,i})| + b, in R^{d_e}
  |
  v
[Location-based Channel Embedding]  -- THE KEY INNOVATION
  |-- Map electrode name -> (u, v) Cartesian coordinates via universal lookup
  |-- Sinusoidal encoding (see formula below)
  |-- Added to signal embeddings
  |
  v
[Student Encoder]  -- 4-layer Transformer, processes MASKED tokens
  |
  v
[Teacher Encoder]  -- 4-layer Transformer, processes UNMASKED tokens (EMA of student)
  |
  v
[Predictor]  -- 2-layer cross-attention Transformer
  |-- Predicts masked representations to match teacher
  |
  v
[Decoder]  -- 2-layer Transformer + 1D transposed conv
  |-- Reconstructs ICA-cleaned signal (NOT raw noisy signal)
```

### Location-based Channel Embedding Formula

Given electrode position (u, v) in 2D Cartesian coordinates on scalp:

```
p_{u,v}(4k)   = sin(u * omega_k)
p_{u,v}(4k+1) = cos(u * omega_k)
p_{u,v}(4k+2) = sin(v * omega_k)
p_{u,v}(4k+3) = cos(v * omega_k)

where omega_k = 1000^{-4k/d_e}, k = 0, 1, ..., d_e/4
```

**Key property**: Nearby electrodes get similar embeddings (high dot product).
Formally: Dist((u_i,v_i), (u_j,v_j)) < Dist((u_i,v_i), (u_k,v_k))
  => p_{u_i,v_i} . p_{u_j,v_j} > p_{u_i,v_i} . p_{u_k,v_k}

This is entirely position-based with no learnable parameters, so it generalizes
to ANY electrode configuration without retraining.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim d_e | 16 |
| Attention heads | 8 |
| FFN expansion | 4x |
| Student encoder layers | 4 |
| Teacher encoder layers | 4 |
| Predictor layers | 2 (cross-attention) |
| Decoder layers | 2 + transposed conv |
| Patch size | 128 (1s at 128Hz) |
| Overlap | 32 time steps |
| Batch size | 256 |
| Optimizer | Adam, lr=1e-3 |
| LR schedule | Cosine |
| Pretrain epochs | 300 |
| Finetune epochs | 10 |

### Training Losses

```
L_total = L_Rec + L_Align + L_Reg
```

- **L_Align**: L2 distance between predictor output and teacher targets for masked patches
- **L_Rec**: MSE in DiCT-transformed space between reconstruction and ICA-cleaned signal
- **L_Reg**: VICReg regularization (variance + covariance) to prevent representation collapse

### DiCT (Dictionary Convolution Transformation)

Applied to BOTH artifact-removed target and reconstruction BEFORE computing MSE loss:
- G=32 groups of random convolutional kernels
- K=8 competing kernels per group
- At each time step, max/min responding kernel is selected (dictionary competition)
- Varying dilations capture multiple temporal scales
- Benefits: noise robustness, frequency balance, shape-awareness
- Inspired by Hydra/ROCKET random convolution literature

### Noise-aware Reconstruction

Instead of reconstructing raw (noisy) EEG:
1. Apply ICA decomposition to raw EEG
2. Use ICLabel to identify artifact components
3. Remove high-probability artifact components
4. Reconstruct cleaned signal as target
5. Model learns to predict clean signal from noisy input

### Code

GitHub: https://github.com/Emotiv/EEG-X

---

## Relevance to Our Task (Ear-SAAD: 46ch -> 12ch in-ear prediction)

### What We Can Adopt

#### 1. Location-based Channel Embedding (from EEG-X) -- EASIEST TO IMPLEMENT

Our 46 input channels include non-standard around-ear electrodes (cEEGrid) that don't
have standard 10-20 positions. The EEG-X sinusoidal encoding works with ANY (u,v) coordinate:

```python
import numpy as np

def location_embedding(u, v, d_e=64):
    """EEG-X style sinusoidal channel embedding from 2D scalp coordinates."""
    emb = np.zeros(d_e)
    for k in range(d_e // 4):
        omega_k = 1000 ** (-4 * k / d_e)
        emb[4*k]     = np.sin(u * omega_k)
        emb[4*k + 1] = np.cos(u * omega_k)
        emb[4*k + 2] = np.sin(v * omega_k)
        emb[4*k + 3] = np.cos(v * omega_k)
    return emb
```

For around-ear electrodes, we can assign approximate (u, v) coordinates based on their
physical placement near the ears (roughly at T7/T8 level, slightly posterior).
For in-ear electrodes, they're essentially at the ear canal location.

#### 2. Cross-Attention Channel Unification (from LUNA) -- MOST POWERFUL

The LUNA approach is ideal for our variable-channel scenario:
- Input: 46 channels (27 scalp + 19 around-ear) with spatial position embeddings
- Channel-unification: Q=4-8 learned queries compress 46 channels -> Q latent tokens
- Temporal processing operates on Q tokens, not 46 channels
- Output decoder: 12 learned queries (one per in-ear channel) cross-attend to latent

This naturally handles:
- Different channel counts between training subjects
- Missing/NaN channels (just exclude from cross-attention keys)
- Non-standard electrode positions (via positional encoding)

#### 3. Combined Approach (Recommended for iter038+)

```
46-ch input (27 scalp + 19 around-ear)
  |-- Each channel: temporal conv -> patch features
  |-- Add location embedding (EEG-X style sinusoidal from 2D coords)
  |-- Channel-Unification (LUNA style): 46 channels -> Q=6 latent queries via cross-attn
  |-- Temporal transformer on Q*E latent sequence
  |-- Decoder: 12 in-ear queries cross-attend to latent -> predict 12 output channels
```

Key advantages over our current FIR approach:
- Handles all 46 input channels simultaneously (not just 27 scalp)
- Spatial awareness through position embeddings
- Can learn which input channels are most relevant per output channel
- Attention weights are interpretable (which scalp regions predict which ear channels)

#### 4. NeRF-style 3D Positional Encoding (from LUNA)

LUNA uses 3D coordinates + NeRF encoding, which is better for around-ear electrodes
since they have a distinct z-coordinate (below scalp plane). This could differentiate
scalp vs. around-ear vs. in-ear channels more precisely than 2D.

### What We Should NOT Adopt (overkill for our task)

- Masked pretraining (we have too little data for self-supervised pretraining)
- DiCT loss (we're doing regression, not reconstruction)
- Teacher-student EMA (unnecessary complexity for supervised prediction)
- ICA cleaning (our data is already filtered to 1-9 Hz)

### Implementation Priority

1. **iter038**: EEG-X-style location embedding + simple linear/FIR on all 46 channels
2. **iter039**: LUNA-style cross-attention channel unification (Q=4, E=64)
3. **iter040**: Full LUNA encoder with temporal transformer on broadband data (1-45 Hz)
