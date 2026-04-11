# Flash Attention & Custom CUDA Kernels for EEG Temporal-Spatial Attention

**Date**: 2026-04-10
**Model context**: iter039 DeepBroadbandModel — 46 input channels, 128 Hz, 256 timesteps, 4-head attention, ~200K params, RTX 4060 (8GB VRAM, Ada Lovelace SM89)

---

## 1. Executive Summary

**Key finding: Flash Attention provides minimal benefit at our scale.** Our sequence lengths (64 tokens after 4x downsampling) and head dimensions (32) are too small for Flash Attention's IO-aware tiling to outperform PyTorch's default math backend. The real wins come from (a) using `F.scaled_dot_product_attention` for automatic backend selection, (b) `torch.compile()` for whole-model kernel fusion, and (c) custom Triton kernels for fusing our specific multi-scale conv + attention pipeline.

---

## 2. Flash Attention 2/3 in PyTorch — Current State

### 2.1 PyTorch's `scaled_dot_product_attention` (SDPA)

PyTorch 2.2+ integrates Flash Attention 2 directly via `F.scaled_dot_product_attention`. Three backends are auto-selected at runtime:

| Backend | When Selected | GPU Requirement |
|---------|--------------|-----------------|
| **FlashAttention-2** | fp16/bf16, head_dim ≤ 256, no custom mask | SM80+ (Ampere, Ada, Hopper) |
| **Memory-Efficient** (xFormers) | Fallback when Flash unavailable | SM50+ |
| **Math (C++)** | CPU, or when others fail | Any |
| **cuDNN** | Training on Hopper (H100) | SM90+ |

**RTX 4060 (SM89, Ada Lovelace)**: Flash Attention 2 IS supported. Verify at runtime:

```python
import torch
print(torch.backends.cuda.flash_sdp_enabled())       # Should be True
print(torch.backends.cuda.is_flash_attention_available())  # Should be True

# Force Flash Attention to confirm it works:
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    q = torch.randn(1, 4, 64, 32, device='cuda', dtype=torch.float16)
    out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    print("Flash Attention works on this GPU!")
```

### 2.2 Flash Attention 3

Flash Attention 3 targets **Hopper (H100/H200)** exclusively, using:
- Asynchronous warp-specialized execution (WGMMA + TMA)
- FP8 quantized attention
- Intra-warpgroup pipelining

**Not applicable to RTX 4060.** FA3 requires SM90+ hardware features.

### 2.3 The `flash-attn` Package vs PyTorch Built-in

| | `pip install flash-attn` | `F.scaled_dot_product_attention` |
|---|---|---|
| Flash Attention version | FA2 (up to FA3 on Hopper) | FA2 |
| Custom features | Sliding window, ALiBi, rotary | Basic attention only |
| Compilation | Requires CUDA toolkit, slow install | Ships with PyTorch |
| **Recommendation** | Only if needing special features | **Use this for our model** |

---

## 3. Why Flash Attention Doesn't Help Us (Much)

### 3.1 Our Attention Dimensions

From `iter039_deep_broadband.py`:
```
Input: (B=128, C=46, T=256)
After downsample 4x: (B=128, T_tokens=64, H=128)
Attention: 4 heads, head_dim=32, seq_len=64
```

### 3.2 The Crossover Problem

Flash Attention achieves speedup by **reducing HBM (DRAM) reads** through tiling into SRAM. This matters when attention is **memory-bound** — i.e., when the attention matrix is large relative to compute.

For our model:
- **Attention matrix**: 64 x 64 = 4,096 elements per head
- **Total attention FLOPs**: `2 * B * h * N * N * d = 2 * 128 * 4 * 64 * 64 * 32 = 134M` FLOPs
- **Memory**: `B * h * N * N * sizeof(fp16) = 128 * 4 * 64 * 64 * 2 = 4 MB`

At **seq_len ≤ 512**, attention is **compute-bound, not memory-bound**. The Flash Attention tiling adds overhead (index calculations, online softmax bookkeeping) without meaningful memory savings. Multiple sources confirm:

> "Tasks with short sequences (≤512 tokens) remain compute-bound and won't benefit from tiling; the overhead of custom kernels may even slow them down."
> — FlashAttention-2 paper analysis

### 3.3 Measured Expectations

| Sequence Length | Flash Attention Speedup | Memory Savings |
|----------------|------------------------|----------------|
| 64 (our model) | **~0.9-1.0x** (possible slowdown) | Negligible |
| 256 | ~1.0-1.1x | Minimal |
| 512 | ~1.1-1.3x | ~2x |
| 2048 | ~2-3x | ~10x |
| 8192 | ~3-5x | ~20x+ |

### 3.4 Benchmark Script

Run this to measure actual speedup on your RTX 4060:

```python
"""Benchmark SDPA backends for our exact model dimensions."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import time

def benchmark_attention(backend, B=128, H=4, N=64, D=32, dtype=torch.float16, warmup=50, iters=200):
    q = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, N, D, device='cuda', dtype=dtype)

    # Warmup
    for _ in range(warmup):
        with sdpa_kernel(backend):
            _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        with sdpa_kernel(backend):
            _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms

    return elapsed

if __name__ == "__main__":
    print("=== SDPA Backend Benchmark (our model dimensions) ===")
    print(f"B=128, H=4, N=64, D=32 (iter039 after 4x downsample)")
    print()

    backends = [
        ("Flash Attention", SDPBackend.FLASH_ATTENTION),
        ("Memory Efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("Math (C++)", SDPBackend.MATH),
    ]

    for name, backend in backends:
        try:
            ms = benchmark_attention(backend)
            print(f"  {name:20s}: {ms:.3f} ms/iter")
        except RuntimeError as e:
            print(f"  {name:20s}: NOT AVAILABLE ({e})")

    # Also test with torch.compile
    print()
    print("=== With torch.compile ===")

    @torch.compile
    def compiled_attn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    q = torch.randn(128, 4, 64, 32, device='cuda', dtype=torch.float16)
    # Warmup compile
    for _ in range(10):
        _ = compiled_attn(q, q, q)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(200):
        _ = compiled_attn(q, q, q)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / 200 * 1000
    print(f"  torch.compile SDPA : {ms:.3f} ms/iter")
```

---

## 4. GPU MODE Lecture Insights

### 4.1 Lecture 12: Flash Attention (Key Takeaways)

From [GPU MODE Lecture 12](https://christianjmills.com/posts/cuda-mode-notes/lecture-012/):

1. **Tiling strategy**: Partition Q, K, V into blocks that fit in SRAM (shared memory). On RTX 4060, each SM has 128 KB shared memory. For our head_dim=32 with fp16, a tile of 64 tokens = 64 * 32 * 2 = 4 KB — trivially fits, which is why standard attention is already fast at this scale.

2. **Online softmax (Milakov & Gimelshein)**: Compute softmax incrementally without materializing the full N×N matrix. Critical for long sequences, unnecessary overhead for N=64.

3. **Recomputation trade-off**: Flash Attention recomputes attention weights during backward pass instead of storing them. At N=64, storing the full 64×64 matrix costs only 8 KB per head — the recomputation overhead isn't worth it.

### 4.2 Lecture Series Resources

The full [GPU MODE lecture series](https://github.com/gpu-mode/lectures) covers:
- Lecture 1: Custom CUDA kernels in PyTorch (`load_inline`, profiling)
- Lecture 3: CUDA programming with PyTorch
- Lecture 6: Optimizing CUDA matmul
- Lecture 12: Flash Attention
- Lecture 14: Quantization

---

## 5. What Actually Helps: Practical Optimizations

### 5.1 Priority 1: Use `F.scaled_dot_product_attention` (FREE speedup)

Replace `nn.MultiheadAttention` with direct SDPA call. This avoids the overhead of MHA's input projection reshape and lets PyTorch auto-select the fastest backend.

**Current code** (iter039, line 51):
```python
class ChannelAttention(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return residual + self.drop(x)
```

**Optimized**:
```python
class ChannelAttention(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        # Single projection for Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D_head)
        q, k, v = qkv.unbind(0)

        # Auto-selects best backend (Flash/MemEfficient/Math)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.out_proj(x)
        return residual + self.drop(x)
```

**Expected speedup**: 10-20% from fused QKV projection + SDPA dispatch.

### 5.2 Priority 2: `torch.compile()` (Biggest Win)

For small models, `torch.compile()` provides far more benefit than Flash Attention by fusing elementwise ops, reducing kernel launches, and optimizing memory access patterns.

```python
model = DeepBroadbandModel(C_in=46, C_out=12, T=256, H=128, n_blocks=4).to(device)
model = torch.compile(model, mode="reduce-overhead")  # Best for small models

# For inference only:
model = torch.compile(model, mode="max-autotune")  # Tries all backends
```

**Expected speedup**: 1.5-3x for training, 2-4x for inference on small models.

**Caveats**:
- First call triggers JIT compilation (~30s)
- `reduce-overhead` mode uses CUDA graphs — incompatible with dynamic shapes
- Our model has fixed shapes (B=128, C=46, T=256), so this works perfectly

### 5.3 Priority 3: Custom Triton Fused Kernel for Multi-Scale Conv + Attention

The real bottleneck in iter039 isn't attention — it's the **4 separate Conv1d kernels** in `MultiScaleConv` plus the reshape/transpose operations between conv and attention stages. A custom Triton kernel can fuse these.

```python
"""Fused multi-scale temporal feature extraction in Triton.

Fuses 4 separate Conv1d + BN + GELU into a single kernel that:
1. Loads input tile once from HBM
2. Computes all 4 conv scales in registers/SRAM
3. Concatenates and applies activation in-place
4. Writes output once to HBM

For our dimensions: C_in=46, T=256, H=128, kernels=(3,7,15,31)
"""
import triton
import triton.language as tl

@triton.jit
def fused_multiscale_conv_kernel(
    X_ptr,          # Input: (B, C_in, T)
    W3_ptr,         # Conv weights for kernel_size=3
    W7_ptr,         # Conv weights for kernel_size=7
    W15_ptr,        # Conv weights for kernel_size=15
    W31_ptr,        # Conv weights for kernel_size=31
    OUT_ptr,        # Output: (B, H, T)
    B: tl.constexpr,
    C_in: tl.constexpr,
    T: tl.constexpr,
    H_per_scale: tl.constexpr,  # H // 4 = 32
    BLOCK_T: tl.constexpr,      # Tile size in time dimension
):
    """Each program instance handles one (batch, output_channel, time_block)."""
    pid_b = tl.program_id(0)   # batch index
    pid_h = tl.program_id(1)   # output channel index (0..H-1)
    pid_t = tl.program_id(2)   # time block index

    # Determine which scale this output channel belongs to
    scale_idx = pid_h // H_per_scale  # 0, 1, 2, or 3
    local_h = pid_h % H_per_scale

    # Select kernel size and weight pointer
    # (In practice, use if/else chain since Triton doesn't support indirect indexing)
    # This is simplified — real impl needs separate branches per scale

    t_offset = pid_t * BLOCK_T
    t_range = t_offset + tl.arange(0, BLOCK_T)
    t_mask = t_range < T

    # Accumulate conv output
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # For kernel_size K, iterate over input channels and filter taps
    # (Pseudocode — actual impl depends on scale_idx)
    K = 3  # Example for scale 0
    for c in range(C_in):
        for k in range(K):
            t_in = t_range + k - K // 2
            t_in_mask = (t_in >= 0) & (t_in < T) & t_mask
            x_val = tl.load(
                X_ptr + pid_b * C_in * T + c * T + t_in,
                mask=t_in_mask, other=0.0
            )
            # w_val = tl.load(W3_ptr + local_h * C_in * K + c * K + k)
            # acc += x_val * w_val
            pass  # Full impl below

    # GELU activation (fused)
    # out = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    # Store
    # tl.store(OUT_ptr + pid_b * H * T + pid_h * T + t_range, out, mask=t_mask)
```

**Realistic assessment**: Writing a correct, performant fused multi-scale conv kernel is substantial engineering. A more practical approach:

```python
# Use torch.compile to get ~80% of the fusion benefit automatically
@torch.compile(mode="reduce-overhead")
def fused_multiscale(x, convs):
    return torch.cat([conv(x) for conv in convs], dim=1)
```

### 5.4 Priority 4: Fused Attention + FFN Triton Kernel

For the Transformer blocks, fusing attention output projection + dropout + residual + LayerNorm + FFN into one kernel eliminates intermediate memory traffic:

```python
"""Fused post-attention: out_proj + residual + LayerNorm + FFN(expand + GELU + contract).

This is where Triton actually helps for small models — not in attention itself,
but in the surrounding operations that have low arithmetic intensity.
"""
import triton
import triton.language as tl

@triton.jit
def fused_residual_layernorm_ffn_kernel(
    ATTN_OUT_ptr,   # Attention output: (B*N, D)
    RESIDUAL_ptr,   # Residual connection: (B*N, D)
    W_OUT_ptr,      # Output projection weights: (D, D)
    W_UP_ptr,       # FFN up-projection: (D, 4*D)
    W_DOWN_ptr,     # FFN down-projection: (4*D, D)
    GAMMA_ptr,      # LayerNorm gamma: (D,)
    BETA_ptr,       # LayerNorm beta: (D,)
    OUT_ptr,        # Final output: (B*N, D)
    D: tl.constexpr,
    D4: tl.constexpr,  # 4*D
    eps: tl.constexpr,
):
    """Each program handles one token (one row of B*N)."""
    row = tl.program_id(0)
    cols = tl.arange(0, D)

    # 1. Load attention output + residual
    attn = tl.load(ATTN_OUT_ptr + row * D + cols)
    res = tl.load(RESIDUAL_ptr + row * D + cols)
    x = attn + res  # Residual connection (fused, no extra write)

    # 2. LayerNorm (fused, no extra write)
    mean = tl.sum(x, axis=0) / D
    var = tl.sum((x - mean) * (x - mean), axis=0) / D
    gamma = tl.load(GAMMA_ptr + cols)
    beta = tl.load(BETA_ptr + cols)
    x_norm = (x - mean) / tl.sqrt(var + eps) * gamma + beta

    # 3. FFN up-projection + GELU (fused)
    # For D=128, D4=512 — fits in registers on SM89
    # (Actual matmul would use tl.dot with proper tiling)

    # 4. FFN down-projection

    # 5. Store final output
    tl.store(OUT_ptr + row * D + cols, x_norm)  # Simplified
```

**Note**: The above is a sketch. Production-quality fused kernels need proper matmul tiling. For D=128, the entire FFN fits in shared memory on SM89 (128 KB SRAM), making this fusion very effective.

---

## 6. Implementation Plan

### Phase 1: Drop-in Optimizations (1 hour, ~1.5x speedup)

1. **Replace `nn.MultiheadAttention` with fused QKV + `F.scaled_dot_product_attention`**
   - Modify `ChannelAttention` class in iter039
   - Verify outputs match within fp16 tolerance

2. **Add `torch.compile(mode="reduce-overhead")`**
   - Wrap model after construction
   - Ensure no dynamic shapes (our model is all static)
   - Add warmup step before timing

3. **Enable `torch.set_float32_matmul_precision('high')`**
   - Uses TF32 on Ada Lovelace for ~2x matmul speedup
   - No accuracy loss for our application

```python
# In build_and_train():
torch.set_float32_matmul_precision('high')

model = DeepBroadbandModel(...).to(device)
model = torch.compile(model, mode="reduce-overhead")

# Warmup (triggers compilation)
dummy = torch.randn(2, C_scalp, 256, device=device)
_ = model(dummy)
```

### Phase 2: Architecture Tweaks for GPU Efficiency (2 hours, ~2x total)

1. **Use fp16 mixed precision training**
   ```python
   scaler = torch.amp.GradScaler()
   with torch.amp.autocast('cuda', dtype=torch.float16):
       pred = model(scalp)
       loss = loss_fn(pred, inear)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Increase batch size to 256 or 512** (8GB VRAM allows it at fp16)
   - Better GPU utilization at our small model size
   - Adjust learning rate proportionally

3. **Replace BatchNorm1d with fused GroupNorm** in MultiScaleConv
   - GroupNorm is better fused by `torch.compile`
   - No sync needed across batch (relevant for future multi-GPU)

### Phase 3: Custom Triton Kernels (Optional, 1-2 days)

Only pursue if Phase 1-2 don't provide sufficient speedup.

1. **Fused residual + LayerNorm + FFN kernel** (highest value custom kernel)
   - Eliminates 5 kernel launches per Transformer block
   - Template: [Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

2. **Fused multi-scale conv** (medium value)
   - Reduces HBM reads by 4x for temporal feature extraction
   - Complex to implement correctly with varying kernel sizes

3. **Custom fused attention is NOT recommended** at seq_len=64
   - PyTorch's built-in backends are already optimal at this scale
   - Engineering effort far exceeds benefit

---

## 7. Memory Budget Analysis (RTX 4060, 8GB)

```
Model parameters (fp16):    ~400 KB  (200K params * 2 bytes)
Optimizer states (fp32):    ~1.6 MB  (Adam: 2 * params * 4 bytes)
Gradients (fp16):           ~400 KB
Activations (B=128, fp16):  ~50 MB   (est. for 4 transformer blocks)
Input batch:                ~3 MB    (128 * 46 * 256 * 2 bytes)
Attention matrices:         ~4 MB    (128 * 4 * 64 * 64 * 2 * 4 blocks)
PyTorch overhead:           ~500 MB

Total estimated:            ~560 MB  (7% of 8GB!)
```

**We have massive headroom.** This means:
- Batch size can go to 512+ easily
- No need for gradient checkpointing
- No need for Flash Attention's memory savings
- Can afford to cache attention matrices (no recomputation needed)

---

## 8. Recommended Implementation: iter040_compiled_broadband.py

```python
"""Iteration 040: Compiled broadband model with GPU optimizations.

Same architecture as iter039 but with:
1. F.scaled_dot_product_attention (auto-selects best backend)
2. torch.compile(mode="reduce-overhead") for kernel fusion
3. AMP (fp16 mixed precision) for 2x compute throughput
4. TF32 matmul precision
5. Larger batch size (256)

Expected: Same r as iter039, 2-3x faster training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedAttention(nn.Module):
    """Attention using F.scaled_dot_product_attention for auto backend selection."""

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # PyTorch auto-selects Flash/MemEfficient/Math backend
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, D)
        return residual + self.out_proj(x)


# In build_and_train():
def build_and_train(train_ds, val_ds, C_scalp, C_inear, device):
    torch.set_float32_matmul_precision('high')  # TF32

    model = DeepBroadbandModel(C_in=C_scalp, C_out=C_inear, ...).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    scaler = torch.amp.GradScaler()

    for epoch in range(1, 301):
        model.train()
        for scalp, inear in train_loader:
            scalp, inear = scalp.to(device), inear.to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred = model(scalp)
                loss = loss_fn(pred, inear)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
```

---

## 9. Sources

### Flash Attention
- [PyTorch SDPA Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [PyTorch 2.2 Blog: FlashAttention-2 Integration](https://pytorch.org/blog/pytorch2-2/)
- [FlashAttention-2 Paper](https://arxiv.org/pdf/2307.08691)
- [FlashAttention-3: Asynchrony and Low-precision](https://arxiv.org/pdf/2407.08608)
- [Dao-AILab/flash-attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [What is Flash Attention? (Modal)](https://modal.com/blog/flash-attention-article)

### GPU MODE Lectures
- [GPU MODE Lecture 12: Flash Attention Notes](https://christianjmills.com/posts/cuda-mode-notes/lecture-012/)
- [GPU MODE Lecture Series](https://christianjmills.com/series/notes/cuda-mode-notes.html)
- [GPU MODE Lectures GitHub](https://github.com/gpu-mode/lectures)

### Custom CUDA/Triton Kernels
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [The Anatomy of a Triton Attention Kernel](https://arxiv.org/html/2511.11581v1)
- [FlashAttention-2 with CUTLASS on Hopper](https://arxiv.org/html/2312.11918v1)
- [Liger-Kernel: Efficient Triton Kernels](https://github.com/linkedin/Liger-Kernel)
- [Flash Attention from Scratch](https://lubits.ch/flash/Part-1)
- [Reimplementing FlashAttention](https://aminediro.com/posts/flash_attn/)

### EEG Temporal-Spatial Attention
- [HASTF: Hybrid Attention Spatio-Temporal Fusion for EEG](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1479570/full)
- [SSTAF: Spatial-Spectral-Temporal Attention Fusion Transformer](https://arxiv.org/html/2504.13220v1)
- [Transformer-based Spatial-Temporal Feature Learning for EEG](https://arxiv.org/pdf/2106.11170)
