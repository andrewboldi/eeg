# Frequency Domain Prediction for EEG Reconstruction

## Research Summary

Investigation into whether frequency-domain prediction or spectral losses could
improve scalp-to-in-ear EEG reconstruction beyond the current r=0.378 plateau.

---

## 1. Key Papers and Findings

### FreDF: Learning to Forecast in the Frequency Domain (ICLR 2025)
- **arxiv**: 2402.02399
- **Core idea**: Standard direct forecasting ignores label autocorrelation.
  Computing loss in FFT domain reduces estimation bias.
- **Loss formula**: `L_freq = mean(|FFT(pred) - FFT(target)|)` (complex MAE after rfft)
- **Key insight**: Both magnitude AND phase alignment matter. Phase alignment is
  especially critical -- aligning amplitude without phase gives subpar results.
- **Weighting**: Hyperparameter alpha blends time-domain and frequency-domain loss.
  Increasing alpha from 0 to 1 generally reduced forecast error.
- **Relevance to us**: HIGH. Our 2s windows at 128 Hz give 128 FFT bins (1-45 Hz).
  The frequency loss would enforce spectral consistency that MSE alone misses.

### WGAN with Temporal-Spatial-Frequency Loss (Frontiers 2020)
- **Core idea**: EEG reconstruction benefits from multi-domain loss combining
  temporal MSE, spatial (CSP) MSE, and frequency (PSD) MSE.
- **Frequency component**: MSE between power spectral densities of predicted and
  target EEG signals.
- **Result**: Adding frequency loss improved reconstruction quality and downstream
  classification accuracy beyond time-domain-only loss.
- **Relevance to us**: MEDIUM. Confirms spectral loss helps EEG specifically.

### Multi-Resolution STFT Loss (from audio/speech synthesis)
- **Core idea**: Apply STFT at multiple resolutions (different FFT sizes, hop
  lengths, window sizes) and penalize magnitude differences at each.
- **Used in**: Parallel WaveGAN, HiFi-GAN, and most modern neural vocoders.
- **Key insight**: Multi-resolution captures both fine-grained and coarse spectral
  details. Combining with time-domain loss (SI-SDR) gives best results.
- **Relevance to us**: MEDIUM-HIGH. Our 256-sample windows are short, but we
  could use 2-3 resolutions (e.g., FFT sizes 64, 128, 256).

### FNet: Fourier Transform as Token Mixing (NAACL 2022)
- **Core idea**: Replace self-attention with FFT along sequence and hidden dims.
  Achieves 92-97% of BERT accuracy at 80% faster training.
- **Relevance to us**: LOW for prediction, but suggests FFT-based feature mixing
  layers could replace attention in future architectures.

---

## 2. Three Approaches Evaluated

### Approach A: Predict in Frequency Domain (FFT coefficients)
**Idea**: Model outputs FFT(in-ear) coefficients; inverse FFT to get time domain.

**Pros**:
- Each frequency bin is independent -- reduces autocorrelation in targets
- Different frequency bands can be weighted differently
- Magnitude spectrum is smoother than time-domain signal

**Cons**:
- Phase prediction is notoriously unstable (observed in neural vocoders)
- Phase wrapping creates discontinuities that hurt gradient flow
- Need to predict complex values (real + imaginary) or magnitude + phase
- For 256 samples: 129 complex bins = 258 real values vs 256 time samples
  (no dimensionality savings)
- Inverse FFT can amplify small phase errors into large time-domain errors

**Verdict**: NOT RECOMMENDED as primary approach. Phase instability is a known
problem. The FreDF paper specifically avoids predicting in frequency domain;
it only computes loss there.

### Approach B: Auxiliary Spectral Loss (time-domain prediction + frequency penalty)
**Idea**: Keep time-domain prediction but add frequency-domain loss terms.

**Pros**:
- No architectural changes needed -- just modify the loss function
- FreDF shows this reduces estimation bias (ICLR 2025)
- We already have SpectralLoss and BandPowerLoss in `src/losses/combined.py`
- Enforces spectral consistency without phase prediction instability
- Multi-resolution STFT loss proven in audio domain

**Cons**:
- Additional hyperparameters (loss weights)
- FFT computation adds ~10% training time overhead
- Our existing SpectralLoss uses log-magnitude MSE, not the complex MAE
  that FreDF recommends

**Verdict**: STRONGLY RECOMMENDED. Lowest risk, proven in both EEG and
time-series literature. Two specific variants to try (see Section 3).

### Approach C: Wavelet Domain Prediction
**Idea**: Predict wavelet coefficients instead of raw time series.

**Pros**:
- Multi-resolution time-frequency representation
- Natural for EEG (different bands have different spatial patterns)
- No phase wrapping issues (real-valued coefficients)

**Cons**:
- Adds significant complexity (choice of wavelet, decomposition levels)
- Reconstruction from wavelets can introduce boundary artifacts
- 256-sample windows are short for multi-level wavelet decomposition
- No strong evidence of improvement over FFT-based spectral loss for
  EEG reconstruction specifically

**Verdict**: LOW PRIORITY. More complexity for uncertain benefit. Try
spectral loss first.

---

## 3. Concrete Implementation Plan

### Iteration 045: FreDF-Style Spectral Loss

Replace our existing CorrMSELoss with a three-term loss:

```python
class FreDFLoss(nn.Module):
    """FreDF-inspired loss: time MSE + correlation + frequency MAE."""

    def __init__(self, alpha_mse=0.5, alpha_corr=0.5, alpha_freq=0.3):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_corr = alpha_corr
        self.alpha_freq = alpha_freq

    def forward(self, pred, target):
        # Time-domain MSE
        mse = torch.mean((pred - target) ** 2)

        # Correlation loss
        pred_m = pred - pred.mean(dim=-1, keepdim=True)
        target_m = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_m * target_m).sum(dim=-1)
        pred_std = (pred_m ** 2).sum(dim=-1).sqrt()
        target_std = (target_m ** 2).sum(dim=-1).sqrt()
        corr_loss = 1.0 - (cov / (pred_std * target_std + 1e-8)).mean()

        # FreDF frequency loss (complex MAE on FFT coefficients)
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        freq_loss = (pred_fft - target_fft).abs().mean()

        total = (self.alpha_mse * mse
                 + self.alpha_corr * corr_loss
                 + self.alpha_freq * freq_loss)
        return total
```

**Key differences from our existing SpectralLoss**:
1. Uses complex MAE (|FFT(pred) - FFT(target)|) not log-magnitude MSE
2. Preserves phase information (complex difference, not just magnitude)
3. FreDF paper shows this formulation is more effective

**Hyperparameter sweep** (alpha_freq): 0.1, 0.3, 0.5, 1.0

### Iteration 046: Multi-Resolution STFT Loss

Add multi-resolution spectral matching at 2-3 FFT sizes:

```python
class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss adapted from audio synthesis."""

    def __init__(self, fft_sizes=[64, 128, 256], eps=1e-8):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps

    def forward(self, pred, target):
        loss = 0.0
        for n_fft in self.fft_sizes:
            # Magnitude loss (spectral convergence)
            pred_fft = torch.fft.rfft(pred[..., :n_fft], dim=-1)
            target_fft = torch.fft.rfft(target[..., :n_fft], dim=-1)
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)

            # Spectral convergence
            sc = torch.norm(target_mag - pred_mag) / (torch.norm(target_mag) + self.eps)

            # Log magnitude loss
            log_loss = torch.mean(torch.abs(
                torch.log(target_mag + self.eps) - torch.log(pred_mag + self.eps)
            ))

            loss += sc + log_loss
        return loss / len(self.fft_sizes)
```

### Iteration 047: Band-Weighted Frequency Loss

Weight frequency bins by EEG band importance:

```python
def band_weighted_freq_loss(pred, target, fs=128.0):
    """Weight frequency loss by EEG band importance."""
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    freqs = torch.fft.rfftfreq(pred.shape[-1], 1.0 / fs).to(pred.device)

    # Weight: emphasize theta (4-8), alpha (8-13), beta (13-30)
    weights = torch.ones_like(freqs)
    weights[(freqs >= 4) & (freqs < 8)] = 2.0    # theta
    weights[(freqs >= 8) & (freqs < 13)] = 3.0   # alpha (most important for EEG)
    weights[(freqs >= 13) & (freqs < 30)] = 1.5   # beta

    diff = (pred_fft - target_fft).abs()
    return (diff * weights.unsqueeze(0).unsqueeze(0)).mean()
```

---

## 4. Why This Could Break the 0.378 Plateau

The current best uses CorrMSELoss which optimizes:
- MSE: penalizes amplitude errors uniformly across time
- Correlation: penalizes shape mismatch

Neither explicitly encourages spectral fidelity. The FIR filter with 7 taps
at 128 Hz has a frequency response that may not match the target spectrum
well at all frequencies. A spectral loss would directly penalize frequency
response mismatch, potentially guiding the FIR coefficients to better
spectral alignment.

**Expected improvement**: +0.002 to +0.010 based on:
- FreDF shows consistent +1-3% improvement on time series benchmarks
- WGAN-TSF shows improved reconstruction with spectral loss for EEG
- Our data is 1-45 Hz with 128 Hz sampling -- plenty of frequency content
  for spectral loss to be meaningful (unlike the old 1-9 Hz / 20 Hz data)

---

## 5. Recommendations (Priority Order)

1. **[HIGH] Iteration 045 -- FreDF spectral loss**: Add complex MAE frequency
   loss to existing CorrMSE. Minimal code change, strong theoretical backing.
   Start with alpha_freq=0.3.

2. **[MEDIUM] Iteration 046 -- Multi-resolution STFT**: If iter045 shows
   improvement, try multi-resolution variant for additional gains.

3. **[MEDIUM] Iteration 047 -- Band-weighted frequency loss**: Weight
   alpha/theta bands more heavily since they dominate EEG.

4. **[LOW] Frequency-domain prediction**: Only attempt if spectral losses
   show large gains, suggesting frequency alignment is a major bottleneck.

5. **[LOW] Wavelet domain**: Defer unless all FFT approaches plateau.

---

## 6. Existing Codebase Assets

We already have in `src/losses/combined.py`:
- `SpectralLoss`: log-magnitude MSE (not optimal -- should switch to complex MAE)
- `BandPowerLoss`: per-band power MSE (useful but coarse)
- `CombinedLoss`: weighted combination (lambda_spec=0.1, lambda_band=0.1)

The existing `CombinedLoss` was NOT used in the best model (iter030 uses
`CorrMSELoss` only). This means spectral loss has never been combined with
the correlation objective that produces our best results.

## Sources

- [FreDF: Learning to Forecast in the Frequency Domain](https://arxiv.org/abs/2402.02399) - ICLR 2025
- [EEG Reconstruction with WGAN and TSF Loss](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2020.00015/full)
- [Multi-Resolution STFT Loss (Parallel WaveGAN)](https://arxiv.org/abs/1910.11480)
- [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) - NAACL 2022
- [Rethinking Magnitude and Phase Estimation](https://arxiv.org/html/2509.18806v1)
- [Adaptive Frequency-Time Attention for Seizure Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC12025975/)
- [hvEEGNet for High-Fidelity EEG Reconstruction](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2024.1459970/full)
