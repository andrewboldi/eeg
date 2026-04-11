# State-of-the-Art Architectures for EEG Signal Prediction

**Date**: 2026-04-10
**Task**: Predict 12 in-ear EEG channels from 46 scalp channels (128 Hz, broadband, 2s windows)
**Current best**: r=0.577 (closed-form linear), deep model val_r=0.68 but overfits on test
**Key bottleneck**: Cross-subject generalization (12 train, 3 test subjects)

---

## Ranked Architecture Recommendations

### Tier 1: Highest Expected Impact

#### 1. Test-Time BN Adaptation (TENT / NeuroTTT)
**Expected impact: HIGH | Complexity: LOW | Risk: LOW**

The simplest path to closing the train-test gap. Add BatchNorm layers to the model, train normally on subjects 1-12, then at test time update *only* BN affine parameters (gamma/beta) by minimizing prediction entropy on each test subject's data.

**Evidence**:
- NeuroTTT (2025) shows BN-only adaptation outperforms full-parameter TTT for cross-subject EEG
- StableSleep (2025) adds entropy gates to prevent catastrophic adaptation on noisy segments
- SPD-BN / TSMNet (NeurIPS 2022): per-subject BN statistics → SOTA on 6 BCI datasets
- Dual-Stage Alignment (2025): EA + BN stat update gives +3-5% across 5 datasets, 7 backbones

**Implementation** (~50 lines):
1. Add BatchNorm after spatial mixing and before temporal conv
2. Train on subjects 1-12 normally
3. At test time per subject: freeze all weights, update BN gamma/beta via entropy minimization
4. Safety: only adapt when entropy is in [0.3, 0.9] range (StableSleep's entropy gate)

**Why this fits our problem**: Our val_r=0.68 vs test_r=0.577 gap is classic domain shift. Subject 14 consistently underperforms (~0.27 r). BN adaptation directly addresses per-subject distribution differences with zero extra training cost.

**References**: TENT (ICLR 2021, github.com/DequanWang/tent), NeuroTTT (arXiv 2509.26301), StableSleep (arXiv 2509.02982), TSMNet (NeurIPS 2022, github.com/rkobler/TSMNet)

---

#### 2. Bidirectional Mamba (State-Space Model)
**Expected impact: HIGH | Complexity: MEDIUM | Risk: MEDIUM**

Consensus architecture across 2024-2026 EEG papers. Bidirectional Mamba is inherently acausal (critical -- our experiments show acausal filtering is essential) with learnable selective state dynamics that go beyond fixed FIR taps.

**Key papers**:
- **FEMBA** (2026): 2 BiMamba blocks, 7.8M params, 27x fewer FLOPs than LaBraM. Pretrained on 21,600 hours TUEG. Conv2d tokenizer patches (channels, time) jointly. GitHub: pulp-bio/biofoundation
- **SAMBA** (2025): 1.0M params, U-shaped encoder-decoder with Differential Mamba (forward - λ·backward). 3D spatial-adaptive embedding handles variable channel positions. Scales 2s-100s windows.
- **BioMamba** (2025): 6 blocks, d=128, spectro-temporal embedding (STFT + time domain), sparse FFN. SOTA on 5/6 biosignal datasets.
- **SSM2Mel** (2025): S4-UNet for EEG→mel spectrogram regression (closest to our regression task). Subject-specific Embedding Strength Modulator (ESM) for cross-subject adaptation.

**Recommended architecture for our task**:
```
Input: [B, 46, 256]  (46 channels, 2s @ 128Hz)
├── Conv1d spatial: 46 → 64 channels
├── BiMamba block 1: d_model=64, expand=2 (forward + backward)
├── BiMamba block 2: d_model=64, expand=2
├── Linear head: 64 → 12 channels
Output: [B, 12, 256]
```
~50K params. Use `pip install mamba-ssm` (Tri Dao's implementation).

**Why BiMamba over FIR**: The selective state space mechanism learns input-dependent temporal filters -- it can adapt its effective filter length per-sample, unlike fixed 7-tap FIR. Bidirectional = acausal. Linear complexity in sequence length.

---

#### 3. Channel-Wise Mixture-of-Experts + Subject-Specific Routing
**Expected impact: HIGH | Complexity: MEDIUM | Risk: MEDIUM**

Route different scalp channel groups to specialized expert networks. The key insight from MoGE (BIBM 2024) and BrainMoE (2025): route per-channel, not per-subject. Different brain regions need different spatial filters.

**Key papers**:
- **MoGE** (BIBM 2024): Per-channel routing to GNN experts. Decomposes brain into functional areas. No domain-alignment loss needed. GitHub: XuanhaoLiu/MoGE
- **BrainMoE** (2025): Channel-wise MoE in MAE framework. Each channel dynamically assigned to expert.
- **MoRE-Brain** (NeurIPS 2025): Shared experts + subject-specific routers. At test time, freeze experts, fine-tune only router (~5% of params). GitHub: yuxiangwei0808/MoRE-Brain

**Recommended architecture**:
```
Input: [B, 46, 256]
├── Router: Linear(46) → softmax over K=4 experts (per-sample routing)
├── Expert k: Conv1d spatial 46→12 + FIR temporal (each expert has own filter bank)
├── Output = Σ(router_weight_k × expert_k_output)
Output: [B, 12, 256]
```
~10K params (4 expert filter banks + router). MoRE-Brain variant: at test time, freeze experts, fine-tune router on unlabeled test data.

**Why MoE**: Our Subject 14 problem (r~0.27 vs 0.46 for Subject 13) suggests different subjects need fundamentally different spatial mappings. MoE can learn multiple mapping strategies and route appropriately.

---

### Tier 2: Moderate Expected Impact

#### 4. CBraMod Foundation Model Fine-Tuning
**Expected impact: MEDIUM-HIGH | Complexity: MEDIUM | Risk: MEDIUM**

Best-fit EEG foundation model for our task. Only 4.0M params with criss-cross spatial+temporal attention. Pretrained on >9,000 hours (TUEG). Handles variable channels via Asymmetric Conditional Positional Encoding (ACPE).

**Architecture**: 12-layer criss-cross transformer (4 spatial + 4 temporal attention heads per layer), 200 hidden dim, 800 FFN dim, 3-layer 1D conv patch encoder.

**How to adapt**: Use pretrained encoder as feature extractor, add linear decoder head mapping to 12 output channels. The spatial attention directly learns spatial filters; temporal attention handles temporal dynamics.

**Caveat**: All EEG foundation models are trained for classification. Our regression task needs a custom decoder head. The pretrained temporal features may not align with our prediction objective.

**Weights**: Available in braindecode (`pip install braindecode`), load via `CBraMod.from_pretrained()`. Also at github.com/wjq-learning/CBraMod (ICLR 2025).

**Other foundation models considered**:
- **LaBraM** (ICLR 2024): 5.8M-369M params, BEiT-style, variable channels via channel patches. braindecode/labram-pretrained on HuggingFace.
- **EEGPT** (NeurIPS 2024): 10M params, handles missing channels, separate spatial/temporal processing.
- **REVE** (NeurIPS 2025): Pretrained on 60,000+ hours, 25,000 subjects. Largest EEG pretraining. GitHub: elouayas/reve_eeg
- **ZUNA** (Zyphra 2026): 380M params, masked diffusion autoencoder with 4D rotary positional encoding. Trained with heavy channel-dropout -- directly applicable to predict missing channels. Apache-2.0, HuggingFace: Zyphra/ZUNA

---

#### 5. xLSTM-Mixer / minGRU for Temporal Modeling
**Expected impact: MEDIUM | Complexity: LOW-MEDIUM | Risk: LOW**

Modern RNN variants that are fully parallelizable and competitive with Transformers.

**minGRU** (2024, arXiv 2410.01201) -- simplest option:
```python
# Entire forward pass:
z = sigmoid(linear_z(x))        # gate from input only
h_tilde = linear_h(x)           # candidate from input only  
h = (1 - z) * h_prev + z * h_tilde  # interpolate
```
175-1324x faster training than traditional LSTM. Only 2 linear layers per cell. Parallelizable via scan.

**xLSTM-Mixer** (NeurIPS 2024, arXiv 2410.16928):
- Purpose-built for multivariate time series
- Uses sLSTM blocks with exponential gating for cross-variate mixing
- Two views: temporal + variate, combined by FC layer
- Beats PatchTST, iTransformer, S-Mamba on standard benchmarks

**Griffin's RG-LRU** (Google DeepMind 2024):
- Diagonal gated linear recurrence, parallelizable via scan
- Conv1d(kernel=4) + RG-LRU pattern could replace FIR filter
- Scales to 14B parameters, matches Transformer speed

**For our task**: Start with minGRU as spatial-temporal layer replacement. If it helps, upgrade to xLSTM-Mixer's dual-view architecture.

---

#### 6. KAN (Kolmogorov-Arnold Networks) for Nonlinear Spatial Filters
**Expected impact: MEDIUM | Complexity: LOW | Risk: MEDIUM**

Replace linear spatial mixing with learnable nonlinear activation functions on edges. Three variants:

**B-spline KAN**: Standard. Grid_size=5, spline_order=3. ~10x params per edge vs linear, but achieves comparable results with 3x fewer total nodes (109K vs 329K). Source: arXiv 2405.08790.

**Wav-KAN** (arXiv 2405.12832): Wavelet basis (Mexican hat, Morlet) instead of B-splines. Multi-resolution analysis captures both slow oscillations and transients. Natural for EEG. More robust than B-spline KAN (less overfitting).

**FourierKAN** (github.com/GistNoesis/FourierKAN): Simplest (25 lines). Uses sin/cos basis. Smooth initialization attenuates high frequencies. Best for periodic signals.

**TKAN** (arXiv 2405.07344): KAN inside LSTM gates. Dominates at longer horizons (6+ steps ahead). Lower variance across runs.

**Recommended for our task**:
```
# KAN as spatial filter (replacing linear W)
For each of T taps:
    KAN([46] → [12], grid_size=3, spline_order=3)  # wavelet basis
Sum across taps → output
```
~46×12×6×T params. Risk: the mapping may be sufficiently linear at broadband that KAN's nonlinearity doesn't help.

---

### Tier 3: Worth Investigating

#### 7. ZUNA-Style Channel Dropout Training
**Expected impact: MEDIUM | Complexity: LOW | Risk: LOW**

From Zyphra's ZUNA (2026): train on all 46+12=58 channels, randomly mask the 12 in-ear channels during training, force the model to predict them from remaining scalp channels. This is a self-supervised pretext task that directly matches our inference objective.

Key innovation: 4D rotary positional encoding over (x,y,z,t) enables inference on arbitrary channel subsets and positions. Trained on 208 datasets (~2M channel-hours).

**Implementation**: Even without using ZUNA's pretrained weights, the channel-dropout training strategy is applicable to any architecture. During training, randomly drop 30-50% of output channels and compute loss only on remaining channels. This acts as strong regularization.

#### 8. Dynamic Spatial Filtering (DSF)
**Expected impact: MEDIUM | Complexity: LOW | Risk: LOW**

Multi-head attention module plugged BEFORE the first layer. Learns per-channel attention weights that adapt to channel quality. From Banville et al. (NeuroImage 2022). Matches baseline on clean data, significantly outperforms under channel corruption. Tested on ~4000 clinical + ~100 mobile EEG recordings.

Plug-and-play: add before any existing model as a learned channel reweighting layer.

#### 9. VAE-cGAN for Probabilistic Prediction
**Expected impact: LOW-MEDIUM | Complexity: HIGH | Risk: HIGH**

VAE-cGAN (Sensors 2025): encode scalp EEG to 256-dim latent, condition generator on both raw input and latent code. Achieved +11% over least-squares for scalp-to-intracranial translation. NeuroFlowNet (2026): conditional normalizing flow models full p(output|input). Both capture prediction uncertainty but are complex to train.

#### 10. MAML-Style Meta-Learning
**Expected impact: LOW-MEDIUM | Complexity: HIGH | Risk: HIGH**

META-EEG (2024, github.com/MAILAB-korea/META-EEG): MAML training where inner loop simulates adaptation to held-out subject, outer loop optimizes post-adaptation performance. Powerful but tricky to tune (inner LR, number of inner steps, which params to adapt).

---

## Implementation Priority

| Priority | Architecture | Expected Δr | Effort | Key Risk |
|----------|-------------|-------------|--------|----------|
| **P0** | TENT BN adaptation | +0.02-0.05 | 1 day | Minimal |
| **P1** | BiMamba (2 blocks) | +0.03-0.08 | 2-3 days | Overfitting |
| **P1** | Channel-wise MoE | +0.02-0.05 | 2 days | Router collapse |
| **P2** | CBraMod fine-tune | +0.02-0.06 | 2-3 days | Task mismatch |
| **P2** | minGRU temporal | +0.01-0.03 | 1 day | May not help |
| **P2** | Channel dropout training | +0.01-0.03 | 0.5 days | None |
| **P3** | Wav-KAN spatial | +0.01-0.02 | 1-2 days | Overfitting |
| **P3** | xLSTM-Mixer | +0.01-0.03 | 2 days | Complexity |
| **P3** | DSF attention | +0.01-0.02 | 0.5 days | Minimal |

## Recommended Experiment Sequence

1. **iter040**: Add BatchNorm + TENT adaptation to current best model → test if BN adaptation closes the train-test gap
2. **iter041**: BiMamba (d=64, 2 blocks) with combined loss → test if SSM temporal modeling beats FIR
3. **iter042**: MoE with K=4 experts + per-channel routing → test if subject-specific routing helps
4. **iter043**: Combine winners from 040-042
5. **iter044**: CBraMod encoder fine-tuning with regression decoder → test if pretrained features help
6. **iter045**: Channel dropout regularization on best architecture

## Key Architectural Insights

1. **Bidirectional/acausal is essential**: Every successful EEG architecture uses bidirectional processing. Our experiments confirm acausal FIR >> causal-only.

2. **Small models suffice**: SAMBA uses 1.0M params, FEMBA-Tiny 7.8M, our best FIR is ~2.3K. The sweet spot for 15 subjects is likely 10K-100K params.

3. **Cross-subject adaptation > bigger models**: The train-test gap (0.68 vs 0.577) suggests adaptation techniques (TENT, subject-specific BN, MoE routing) will outperform raw model capacity.

4. **Channel-wise operations are the trend**: MoGE, BrainMoE, CBraMod's criss-cross attention, DSF -- all operate per-channel rather than flattening channels.

5. **Foundation model features are classification-oriented**: All pretrained EEG models target classification, not regression. Fine-tuning for our task requires a custom decoder and may not transfer well.

## References

### EEG Prediction & Reconstruction
- VAE-cGAN scalp-to-intracranial: doi.org/10.3390/s25020494
- NeuroFlowNet: arXiv 2603.03354
- ZUNA: arXiv 2602.18478, HuggingFace: Zyphra/ZUNA
- STAD Diffusion: arXiv 2407.03089
- SRGDiff: arXiv 2510.19166
- Dynamic Spatial Filtering: doi.org/10.1016/j.neuroimage.2022.119006

### State-Space Models / Mamba
- FEMBA: arXiv 2603.26716, github.com/pulp-bio/biofoundation
- SAMBA: arXiv 2511.18571
- BioMamba: arXiv 2503.11741
- SSM2Mel: arXiv 2501.10402, github.com/fchest/SSM2Mel
- LuMamba: arXiv 2603.19100

### Modern RNNs
- xLSTM: arXiv 2405.04517
- xLSTM-Mixer: arXiv 2410.16928 (NeurIPS 2024)
- Griffin/Hawk: arXiv 2402.19427
- minGRU/minLSTM: arXiv 2410.01201

### KAN
- KAN for time series: arXiv 2405.08790
- TKAN: arXiv 2405.07344
- Wav-KAN: arXiv 2405.12832
- FourierKAN: github.com/GistNoesis/FourierKAN

### MoE for EEG
- MoGE: BIBM 2024, github.com/XuanhaoLiu/MoGE
- BrainMoE: OpenReview 2025
- MoRE-Brain: NeurIPS 2025, github.com/yuxiangwei0808/MoRE-Brain
- SpecMoE: arXiv 2603.16739

### Domain Adaptation / TTA
- TENT: ICLR 2021, github.com/DequanWang/tent
- NeuroTTT: arXiv 2509.26301
- StableSleep: arXiv 2509.02982
- TSMNet / SPD-BN: NeurIPS 2022, github.com/rkobler/TSMNet
- META-EEG: github.com/MAILAB-korea/META-EEG

### Foundation Models
- CBraMod: ICLR 2025, github.com/wjq-learning/CBraMod
- LaBraM: ICLR 2024, github.com/935963004/LaBraM
- EEGPT: NeurIPS 2024, github.com/BINE022/EEGPT
- REVE: NeurIPS 2025, github.com/elouayas/reve_eeg
- BENDR: github.com/SPOClab-ca/BENDR
