# KAN (Kolmogorov-Arnold Networks) for EEG Spatial Channel Mixing

## Research Summary

**Question**: Could a KAN layer replace our `Linear(46, 12)` spatial mixing with learnable B-spline activations that capture nonlinear volume conduction?

**Recommendation**: **Not recommended for our use case.** The mapping is approximately linear (CF baseline is strong at r=0.366), the parameter overhead is 5-10x, and published KAN4TSF results show KAN does not reliably beat MLP on time-series regression. A simpler nonlinearity (e.g., a 2-layer MLP with GELU) would be a better test of the nonlinearity hypothesis.

---

## 1. What Are KANs?

Kolmogorov-Arnold Networks (KAN) were published at ICLR 2025 ([Liu et al., arXiv 2404.19756](https://arxiv.org/abs/2404.19756)). Instead of fixed activations on nodes (like ReLU in MLPs), KANs place **learnable univariate functions on edges**, parameterized as B-splines.

A KAN layer maps R^n -> R^m by learning m*n independent univariate spline functions:

```
output_j = sum_i phi_{j,i}(input_i)
```

where each `phi_{j,i}` is a B-spline with `(grid_size + spline_order)` learnable coefficients, plus a residual linear base weight.

### Parameter Count Formula

For a single KANLinear(in_features, out_features):
- **Spline weights**: `out_features * in_features * (grid_size + spline_order)`
- **Base weights**: `out_features * in_features` (linear residual)
- **Total**: `out_features * in_features * (grid_size + spline_order + 1)`

### Our Use Case: 46 -> 12

| Configuration | Parameters |
|---|---|
| `nn.Linear(46, 12)` | 46 * 12 + 12 = **564** |
| `KANLinear(46, 12, grid_size=3, spline_order=3)` | 12 * 46 * (3+3+1) = **3,864** (~7x more) |
| `KANLinear(46, 12, grid_size=5, spline_order=3)` | 12 * 46 * (5+3+1) = **4,968** (~9x more) |
| `KANLinear(46, 12, grid_size=10, spline_order=3)` | 12 * 46 * (10+3+1) = **7,728** (~14x more) |

---

## 2. PyTorch Libraries

Several pip-installable KAN implementations exist:

| Library | Install | Notes |
|---|---|---|
| **efficient-kan** ([GitHub](https://github.com/Blealtan/efficient-kan)) | `pip install efficient-kan` | Reformulates splines as matrix ops. Best for embedding in existing models. |
| **pykan** ([GitHub](https://github.com/KindXiaoming/pykan)) | `pip install pykan` | Official reference implementation. Heavier, includes visualization. |
| **torchkan** ([GitHub](https://github.com/1ssb/torchkan)) | Clone + pip install | Legendre polynomial variant, ~500us forward pass. |

The **efficient-kan** library is the most practical: pure PyTorch, no exotic dependencies, `KANLinear` is a drop-in `nn.Module`.

### API (efficient-kan)

```python
from efficient_kan import KANLinear

layer = KANLinear(
    in_features=46,
    out_features=12,
    grid_size=5,       # number of B-spline intervals (default 5)
    spline_order=3,    # B-spline degree (default 3, cubic)
    scale_noise=0.1,
    scale_base=1.0,
    scale_spline=1.0,
    base_activation=torch.nn.SiLU,
    grid_range=[-1, 1],
)
```

---

## 3. KAN for EEG: Published Results

### KAN-EEG (Royal Society Open Science, March 2025)
- **Task**: Seizure detection (classification, not regression)
- **Approach**: Replace MLP classifier head with KAN after a CNN/transformer backbone
- **Result**: Improved accuracy on 3 datasets, fewer parameters than MLP head
- **Relevance to us**: Low. They use KAN for classification, not spatial mixing or regression.

### KAN4TSF (arXiv 2408.11306)
- **Task**: Multivariate time series forecasting (regression)
- **Key finding**: KAN and KAN-based models (RMoK) are effective but **do not consistently outperform MLPs** on time series regression
- **Training time**: KAN variants take ~1.6-2x longer to train
- **Relevance**: High. Our task is time-series channel-to-channel regression.

### CNN-KAN-F2CA (PLOS ONE, 2025)
- **Task**: EEG emotion recognition with sparse channels
- **Approach**: KAN for nonlinear feature transformation after CNN
- **Relevance**: Moderate. Shows KAN can capture cross-channel nonlinearities in EEG.

### KAN vs MLP Fair Comparison (arXiv 2407.16674)
- At matched parameter count, **KAN advantage is small and task-dependent**
- KAN excels on smooth low-dimensional functions (symbolic regression)
- KAN shows **less advantage on noisy real-world data**
- KAN trains ~10x slower than MLP with same parameter count

---

## 4. Analysis for Our Specific Use Case

### Arguments FOR KAN spatial mixing:
1. **Nonlinear volume conduction**: EEG volume conduction is approximately linear, but electrode impedance, tissue inhomogeneity, and reference effects introduce nonlinearities that B-splines could model.
2. **Low dimensional**: 46 -> 12 is exactly the regime where KAN shines (low input dim, curse of dimensionality not an issue).
3. **Interpretable**: Each spline phi_{j,i} shows how input channel i contributes to output channel j. Could reveal nonlinear transfer functions.
4. **Parameter efficient vs deep MLP**: A single KAN layer with grid_size=3 (3,864 params) may capture nonlinearities that would require a 2-layer MLP with hidden dim 32 (46*32 + 32*12 = 1,856 params) to approximate.

### Arguments AGAINST KAN spatial mixing:
1. **The mapping IS approximately linear**: Our CF baseline (pure linear) achieves r=0.366, and the best learned model (FIR) achieves r=0.378. The gap is only 0.012, mostly from temporal filtering, not spatial nonlinearity.
2. **Parameter overhead**: 7-14x more parameters for a single spatial layer risks overfitting on our small dataset (12 training subjects).
3. **Training speed**: 2-10x slower training. Our benchmark needs to run in <5 minutes.
4. **KAN4TSF evidence is weak**: Published results show KAN does not reliably improve time-series regression over MLP.
5. **Noisy data**: KAN's advantage is on smooth functions. EEG is inherently noisy, which negates KAN's smoothness advantage.
6. **The bottleneck is cross-subject variability**, not model expressiveness. Subject 14 is always ~0.27 regardless of model complexity.
7. **InstanceNorm already failed**: Adding nonlinear normalization hurt performance. More nonlinearity in the spatial layer may do the same.

### Parameter count comparison with alternatives:

| Model | Params | Captures nonlinearity? |
|---|---|---|
| Linear(46, 12) | 564 | No |
| KANLinear(46, 12, grid=3) | 3,864 | Yes (B-spline) |
| KANLinear(46, 12, grid=5) | 4,968 | Yes (B-spline) |
| MLP: Linear(46,32) + GELU + Linear(32,12) | 1,868 | Yes (piecewise linear) |
| MLP: Linear(46,16) + GELU + Linear(16,12) | 940 | Yes (piecewise linear) |

---

## 5. Recommendation

**Do not implement KAN spatial mixing.** The risk/reward is unfavorable:

- The mapping is fundamentally linear in the 1-9 Hz band (and likely in broadband too, since volume conduction physics is linear)
- Parameter overhead (7-14x) on a small dataset will likely cause overfitting
- Training speed penalty conflicts with fast iteration requirement
- Published KAN time-series regression results are not compelling
- The performance bottleneck is cross-subject transfer, not spatial nonlinearity

### If you still want to test nonlinear spatial mixing:

A **2-layer MLP with small hidden dim** is a better test of the nonlinearity hypothesis:
- Fewer parameters than KAN
- Faster training
- Well-understood optimization landscape
- If MLP nonlinearity helps, THEN try KAN for potentially better approximation

### Code Snippet: Minimal KAN Spatial Layer (if desired)

If you want to test KAN anyway, here is a self-contained implementation that avoids external dependencies:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANSpatialMixer(nn.Module):
    """KAN-based spatial mixing layer: C_in -> C_out per time step.
    
    Replaces nn.Linear with learnable B-spline activation per edge.
    Each (output_ch, input_ch) pair has its own univariate spline.
    
    Args:
        C_in: number of input channels (e.g., 46)
        C_out: number of output channels (e.g., 12)
        grid_size: number of B-spline grid intervals (default 5)
        spline_order: B-spline polynomial degree (default 3 = cubic)
    """
    
    def __init__(self, C_in, C_out, grid_size=5, spline_order=3):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Linear residual (base weight, like standard Linear)
        self.base_weight = nn.Parameter(torch.empty(C_out, C_in))
        nn.init.kaiming_normal_(self.base_weight)
        
        # B-spline coefficients: one spline per (out, in) pair
        n_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.empty(C_out, C_in, n_bases))
        nn.init.normal_(self.spline_weight, std=0.1)
        
        # Fixed B-spline grid (uniform, covers [-1, 1] with extensions)
        h = 2.0 / grid_size
        grid = torch.linspace(-1 - spline_order * h, 1 + spline_order * h,
                              grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)
        
        self.bias = nn.Parameter(torch.zeros(C_out))
    
    def b_spline_bases(self, x):
        """Compute B-spline basis values. x: (..., C_in) -> (..., C_in, n_bases)"""
        x = x.unsqueeze(-1)  # (..., C_in, 1)
        grid = self.grid       # (grid_size + 2*k + 1,)
        k = self.spline_order
        
        # Order-0 bases
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # (..., C_in, n_intervals)
        
        # Recursion for higher orders
        for p in range(1, k + 1):
            left = (x - grid[:-(p+1)]) / (grid[p:-1] - grid[:-(p+1)] + 1e-8)
            right = (grid[p+1:] - x) / (grid[p+1:] - grid[1:-p] + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        
        return bases  # (..., C_in, n_bases)
    
    def forward(self, x):
        """x: (batch, C_in, T) -> (batch, C_out, T)"""
        B, C, T = x.shape
        
        # Transpose to (B, T, C_in) for per-sample spatial mixing
        x_t = x.permute(0, 2, 1)  # (B, T, C_in)
        
        # Normalize input to [-1, 1] for stable spline evaluation
        # (use running stats or simple standardization)
        x_norm = torch.tanh(x_t)  # simple bounded normalization
        
        # Base linear component: (B, T, C_in) @ (C_in, C_out) -> (B, T, C_out)
        base_out = F.linear(x_t, self.base_weight)  # (B, T, C_out)
        
        # Spline component
        bases = self.b_spline_bases(x_norm)  # (B, T, C_in, n_bases)
        # spline_weight: (C_out, C_in, n_bases)
        # Einsum: sum over C_in and n_bases
        spline_out = torch.einsum("btcn,ocn->bto", bases, self.spline_weight)
        
        out = base_out + spline_out + self.bias  # (B, T, C_out)
        return out.permute(0, 2, 1)  # (B, C_out, T)
    
    def init_from_cf(self, W_cf):
        """Initialize base weights from closed-form solution.
        W_cf: (C_out, C_in) numpy array from ClosedFormLinear.W
        """
        with torch.no_grad():
            self.base_weight.copy_(torch.from_numpy(W_cf).float())
            self.spline_weight.zero_()  # start with pure linear
```

### Usage in a model iteration:

```python
# In build_and_train():
model = KANSpatialMixer(C_scalp, C_inear, grid_size=3, spline_order=3)
model.init_from_cf(cf.W)  # Initialize from closed-form linear solution
# Then train with combined loss as usual
```

---

## 6. Sources

- [KAN: Kolmogorov-Arnold Networks (ICLR 2025)](https://arxiv.org/abs/2404.19756) - Original paper
- [KAN-EEG: Seizure Detection](https://royalsocietypublishing.org/rsos/article/12/3/240999/66115/KAN-EEG-towards-replacing-backbone-MLP-for-an) - KAN applied to EEG classification
- [KAN4TSF: KAN for Time Series Forecasting](https://arxiv.org/html/2408.11306v2) - Benchmark showing KAN does not consistently beat MLP on time series
- [KAN or MLP: A Fairer Comparison](https://arxiv.org/html/2407.16674v1) - Matched-parameter comparison
- [efficient-kan (GitHub)](https://github.com/Blealtan/efficient-kan) - Best PyTorch implementation
- [pykan (Official)](https://github.com/KindXiaoming/pykan) - Reference implementation
- [awesome-kan](https://github.com/mintisan/awesome-kan) - Comprehensive resource list
- [CNN-KAN-F2CA for EEG Emotion](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322583) - KAN + CNN for EEG
