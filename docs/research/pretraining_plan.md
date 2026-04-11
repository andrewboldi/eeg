# Pretraining Plan: EEG Foundation Models for Scalp-to-In-Ear Prediction

**Task**: Predict 12 in-ear EEG channels from 46 input channels (27 scalp + 19 around-ear cEEGrid), 128 Hz, 2s windows.
**Current best**: r = 0.378 (7-tap FIR + SGD, 1-9 Hz @ 20 Hz — narrowband plateau)
**Goal**: Break the 0.378 ceiling by moving to broadband (1-45 Hz @ 128 Hz) with pretrained representations.

---

## 1. Large EEG Datasets (Ranked by Size and Relevance)

### Tier 1: Best for Pretraining

| Dataset | Size | Channels | Fs | Access | Notes |
|---------|------|----------|----|--------|-------|
| **TUH EEG Corpus (TUEG)** | ~30,000 recordings, 26,846 sessions, ~21,000+ hours, 572 GB uncompressed | 24-36 channels (10-20 system), some up to 128 | 250+ Hz | Free, email request to help@nedcdata.org | Gold standard for EEG pretraining. Used by FEMBA, LUNA, BIOT. Download via rsync. |
| **MGH EEG (PREST)** | ~5M samples used by BIOT | ~18 channels | 200 Hz | Restricted access | Used by BIOT for pretraining. May require institutional agreement. |
| **SHHS (Sleep Heart Health Study)** | ~5M samples (sleep EEG) | 2-6 channels | 125 Hz | NSRR (free registration) | Sleep EEG, used by BIOT. Limited channels but huge volume. |

### Tier 2: Good Supplementary Data

| Dataset | Size | Channels | Fs | Access | Notes |
|---------|------|----------|----|--------|-------|
| **MOABB collection** | ~40+ datasets, varied sizes | 3-128 channels | Varied | Free via `moabb` Python package | BCI datasets (motor imagery, P300, SSVEP). Easy to download programmatically. |
| **THINGS-EEG2** | 1.6M+ trials, 20 subjects | 32 channels | 1000 Hz | HuggingFace | Visual stimulus EEG. Large but task-specific. |
| **Siena Scalp EEG** | ~500+ hours | 19-21 channels | 256-512 Hz | Free | Used by LUNA alongside TUEG. |

### Tier 3: Niche / Small

| Dataset | Size | Channels | Fs | Access | Notes |
|---------|------|----------|----|--------|-------|
| **Ear-SAAD** (our dataset) | 15 subjects, ~10 hours total | 27 scalp + 19 cEEGrid + 12 in-ear | 256 Hz (raw BIDS) | Already downloaded | Our target domain. Too small for pretraining alone. |
| **CHB-MIT** | 23 subjects, ~980 hours | 23 channels | 256 Hz | PhysioNet (free) | Seizure data. Used by some pretrained models. |

**Recommendation**: Use **TUEG** as the primary pretraining corpus. It is the largest, most diverse, freely available, and already validated by FEMBA, LUNA, and BIOT.

---

## 2. Pretrained EEG Foundation Models

### Model Comparison

| Model | Params | Pretraining Data | Architecture | Pretrained Weights | Channel Handling | Regression? |
|-------|--------|-----------------|--------------|-------------------|-----------------|-------------|
| **BIOT** | **3.3M** | 10M samples (PREST+SHHS) | Transformer, channel tokenization | [GitHub](https://github.com/ycq091044/BIOT), Braindecode | Each channel tokenized independently; handles variable channels | Yes (adaptable) |
| **LaBraM-Base** | **5.8M** | 2,500 hours, ~20 datasets | ViT-style, channel patches | [GitHub](https://github.com/935963004/LaBraM), [HuggingFace](https://huggingface.co/braindecode/Labram-Braindecode) | Channel-patch with spatial embedding; fixed patch size 200 (1s @ 200Hz) | Yes (R2 metric, MSE loss) |
| **LaBraM-Large** | **46M** | Same as Base | Same, deeper | Same repo | Same | Yes |
| **FEMBA-Tiny** | **7.8M** | 21,000+ hours (TUEG) | Bidirectional Mamba (SSM) | [HuggingFace](https://huggingface.co/PulpBio/FEMBA) | 2D-conv tokenizer over channels x time | Adaptable |
| **FEMBA-Base** | **47.7M** | Same | Same, deeper | Same | Same | Adaptable |
| **LUNA** | ~10-50M (est.) | 21,000+ hours (TUEG+Siena) | Cross-attention to fixed latent space | BioFoundation GitHub | **Topology-agnostic** via learned queries; any channel count | Adaptable |
| **EEGPT** | **10M** | 37.5M samples, up to 138 electrodes | Electrode-wise modeling | [GitHub](https://github.com/BINE022/EEGPT) | Electrode-wise: each electrode is a unit; handles diverse montages | Adaptable |
| **LaBraM-Huge** | **369M** | Same as Base | Same, deepest | Same repo | Same | Yes |
| **SingLEM** | Varied | 71 datasets, 357K+ hours | Single-channel model | [GitHub](https://github.com/ttlabtuat/SingLEM) | **Fully channel-agnostic** (single-channel); handles ANY electrode | Adaptable |

### Key Observations

1. **BIOT** is the smallest (3.3M) and most practical for 8GB VRAM fine-tuning.
2. **LaBraM-Base** (5.8M) has explicit regression support and is available through Braindecode with one-line loading.
3. **FEMBA-Tiny** (7.8M) is the most efficiently pretrained (21K hours on TUEG), uses Mamba (linear scaling), and fits on edge devices.
4. **LUNA** is the most architecturally suited to our channel mismatch problem (topology-agnostic latent space).
5. **SingLEM** sidesteps the channel problem entirely by operating per-channel, but loses spatial information.

---

## 3. The Channel Mismatch Problem

**Core issue**: Our 19 around-ear (cEEGrid) and 12 in-ear channels do not exist in ANY standard pretraining dataset. Standard 10-20 montages have ~21 scalp channels; our 27 scalp channels overlap well, but the 19+12 ear channels are unique.

### Strategies (Ranked)

#### Strategy A: Pretrain encoder on scalp channels only, learn ear mapping from scratch
- Use a pretrained model (LaBraM/FEMBA/BIOT) as a **scalp encoder** for the 27 scalp input channels
- Add a **lightweight decoder head** that maps the pretrained scalp representations to 12 in-ear targets
- The 19 around-ear channels get their own small encoder (trained from scratch or with channel-agnostic model)
- **Pros**: Leverages pretraining where it exists (scalp); simple
- **Cons**: No pretraining benefit for ear channels; scalp-only encoder misses around-ear info

#### Strategy B: Channel-agnostic architecture (LUNA/SingLEM approach)
- Use LUNA's cross-attention mechanism or SingLEM's single-channel approach
- All 46 input channels are projected into a shared latent space regardless of electrode position
- The model learns channel identity from learned spatial embeddings (can be trained for novel positions)
- **Pros**: Cleanest solution; all channels treated uniformly; pretrained weights transfer
- **Cons**: LUNA may not have public weights yet; SingLEM loses inter-channel spatial structure

#### Strategy C: Electrode-wise pretraining (EEGPT approach)
- Treat each electrode as an independent unit
- Pretrain a temporal encoder on all channels from TUEG (learns universal temporal EEG patterns)
- At fine-tuning, apply the same temporal encoder to all 46 channels (scalp + around-ear)
- Add a cross-channel attention layer on top to learn the spatial mapping
- **Pros**: Temporal patterns transfer to any electrode; around-ear channels benefit from pretraining
- **Cons**: Must add spatial reasoning at fine-tuning time

#### Strategy D: Two-stage approach
1. **Stage 1**: Pretrain a masked autoencoder on TUEG scalp EEG (learn temporal+spatial EEG priors)
2. **Stage 2**: Fine-tune on Ear-SAAD with all 46 input channels, initializing scalp channel weights from pretrained model and randomly initializing ear channel weights
- **Pros**: Maximum control; can design architecture for our exact task
- **Cons**: Requires pretraining from scratch (expensive)

**Recommendation**: **Strategy C (electrode-wise)** is the best balance of practicality and effectiveness. It lets us leverage pretrained temporal representations for ALL channels (including novel ear electrodes) while learning the spatial mapping during fine-tuning on our data.

---

## 4. Recommended Pretraining Strategy

### Primary Plan: BIOT + Electrode-wise Fine-tuning

**Why BIOT?**
- Smallest model (3.3M params) -- fits easily in 8GB VRAM
- Explicitly designed for cross-data learning with mismatched channels
- Channel tokenization: each channel is independently tokenized into fixed-length segments
- Pretrained weights available (multiple variants on GitHub)
- Available through Braindecode with simple loading API
- Contrastive pretraining objective learns general temporal features

**Architecture for our task:**
```
Input: (batch, 46 channels, 256 samples)  # 2s @ 128Hz

1. Channel Tokenizer (from BIOT pretrained):
   - Each of 46 channels -> fixed-length token embedding
   - Pretrained weights transfer temporal EEG knowledge

2. Spatial Transformer (fine-tuned):
   - Cross-attention over 46 channel tokens
   - Learns which scalp/around-ear channels map to each in-ear target
   - Add learnable position embeddings for all 46 channel positions

3. Decoder Head (trained from scratch):
   - MLP or linear layer: latent -> 12 in-ear channels x 256 samples
   - Or per-target-channel decoder for flexibility
```

### Fallback Plan: LaBraM-Base

If BIOT underperforms, LaBraM-Base (5.8M params) offers:
- Explicit regression support with MSE loss
- Stronger pretraining (2,500 hours vs BIOT's mixed setup)
- Channel-patch architecture adaptable to our 46 channels
- Braindecode integration: `model = Labram.from_pretrained("braindecode/labram-pretrained")`

### Ambitious Plan: FEMBA-Tiny

If we want the strongest pretrained representations:
- FEMBA-Tiny (7.8M params) pretrained on 21,000+ hours of TUEG
- Bidirectional Mamba architecture (linear complexity, efficient)
- Would need custom adaptation for regression + channel mismatch
- Weights on HuggingFace: `PulpBio/FEMBA`

---

## 5. Concrete Implementation Plan

### Phase 1: Setup and Baseline (Day 1)

```bash
# Install braindecode (has BIOT, LaBraM, LUNA pretrained models)
uv add braindecode

# Verify model loading
python -c "
from braindecode.models import BIOT
model = BIOT.from_pretrained('braindecode/biot-pretrained-prest-16chs')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

1. Load BIOT pretrained checkpoint via Braindecode
2. Inspect architecture: input format, channel tokenizer, transformer layers
3. Run existing broadband benchmark as baseline (iter038/039 if available)

### Phase 2: Adapt BIOT for Regression (Day 1-2)

```python
# Pseudocode for adapted BIOT regression model
class BIOTRegression(nn.Module):
    def __init__(self, pretrained_biot, n_input_channels=46, n_output_channels=12, n_times=256):
        # Load pretrained channel tokenizer and transformer
        self.tokenizer = pretrained_biot.tokenizer  # per-channel temporal encoder
        self.transformer = pretrained_biot.transformer  # cross-channel attention

        # New: learnable position embeddings for 46 channels (27 scalp + 19 cEEGrid)
        self.channel_embed = nn.Embedding(n_input_channels, embed_dim)

        # New: regression decoder head
        self.decoder = nn.Linear(embed_dim, n_output_channels * n_times)

    def forward(self, x):  # x: (B, 46, 256)
        # Tokenize each channel independently (pretrained weights)
        tokens = self.tokenizer(x)  # (B, 46, embed_dim)
        tokens = tokens + self.channel_embed.weight  # add spatial info

        # Cross-channel transformer (pretrained weights)
        features = self.transformer(tokens)  # (B, 46, embed_dim)

        # Pool and decode to targets
        pooled = features.mean(dim=1)  # (B, embed_dim)
        out = self.decoder(pooled)  # (B, 12*256)
        return out.view(-1, 12, 256)
```

### Phase 3: Fine-tune on Ear-SAAD Broadband (Day 2-3)

1. Preprocess Ear-SAAD at 128 Hz, 1-45 Hz (broadband), 2s windows
2. Use all 46 input channels (27 scalp + 19 around-ear)
3. Training recipe:
   - Freeze BIOT tokenizer + transformer for first 20 epochs (train decoder only)
   - Unfreeze all layers, fine-tune with small LR (1e-5) for 100 epochs
   - Combined MSE + correlation loss (proven best in our experiments)
   - Correlation-based early stopping on validation set
   - LOSO: train on subjects 1-12, test on 13-15

### Phase 4: Iterate on Architecture (Day 3-5)

- Try LaBraM-Base if BIOT underperforms
- Try FEMBA-Tiny for stronger temporal representations
- Experiment with decoder architectures:
  - Per-target-channel decoders (12 separate heads)
  - Temporal upsampling decoder (if using lower-resolution features)
  - Cross-attention decoder (target channels attend to input representations)

---

## 6. Compute Requirements

### Fine-tuning (RTX 4060, 8GB VRAM)

| Model | Params | Est. VRAM (batch=32) | Feasible? |
|-------|--------|---------------------|-----------|
| BIOT | 3.3M | ~1-2 GB | Yes, easily |
| LaBraM-Base | 5.8M | ~2-3 GB | Yes |
| FEMBA-Tiny | 7.8M | ~2-3 GB | Yes |
| LaBraM-Large | 46M | ~4-6 GB | Tight, reduce batch size |
| FEMBA-Base | 47.7M | ~4-6 GB | Tight, reduce batch size |
| LaBraM-Huge | 369M | ~20+ GB | No -- needs multi-GPU |

**All small variants (BIOT, LaBraM-Base, FEMBA-Tiny) fit comfortably on RTX 4060.**

### Training Time Estimates (RTX 4060)

- **Fine-tuning BIOT** (3.3M, 100 epochs, ~10K windows): ~5-15 minutes
- **Fine-tuning LaBraM-Base** (5.8M, 100 epochs): ~10-20 minutes
- **Fine-tuning FEMBA-Tiny** (7.8M, 100 epochs): ~10-20 minutes
- **Pretraining from scratch on TUEG** (not recommended): ~days to weeks on single GPU

### Pretraining from Scratch (NOT recommended for RTX 4060)

Pretraining on TUEG from scratch would require:
- Downloading ~330 GB of compressed EEG data
- Processing into training format (~1-2 TB)
- Training for days/weeks on a single GPU
- **Verdict**: Use existing pretrained weights instead. The whole point of foundation models is to skip this step.

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Pretrained features don't transfer to ear EEG | Medium | Temporal features should transfer; spatial learned at fine-tune time |
| Channel mismatch degrades performance | Medium | Electrode-wise tokenization; new spatial embeddings for ear channels |
| Broadband data doesn't help (noise > signal) | Low | Our narrowband plateau strongly suggests broadband will help |
| 8GB VRAM insufficient | Low | BIOT is 3.3M params; even with overhead, <4GB needed |
| Pretrained model expects different sampling rate | Medium | Resample or adjust tokenizer patch size; most models handle 128-256 Hz |

---

## 8. Summary: Recommended Action Plan

1. **Install braindecode** and load BIOT pretrained weights
2. **Build BIOTRegression**: pretrained channel tokenizer + new spatial embeddings + regression decoder
3. **Fine-tune on Ear-SAAD broadband** (128 Hz, 46 input channels, 12 output channels)
4. **Benchmark against current best** (r = 0.378)
5. **If BIOT fails, try LaBraM-Base** (explicit regression support, slightly larger)
6. **If both fail, try FEMBA-Tiny** (strongest pretraining, 21K hours)
7. **If all pretrained models fail**, fall back to training our own masked autoencoder on the 27 scalp channels from TUEG (expensive but feasible)

**Expected outcome**: Breaking the r = 0.378 plateau by (a) moving to broadband and (b) leveraging pretrained temporal representations that generalize better across subjects.

---

## References

- [TUH EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/) - Primary pretraining dataset
- [LaBraM GitHub](https://github.com/935963004/LaBraM) - ICLR 2024 spotlight
- [BIOT GitHub](https://github.com/ycq091044/BIOT) - NeurIPS 2023
- [FEMBA on HuggingFace](https://huggingface.co/PulpBio/FEMBA) - Bidirectional Mamba
- [LUNA paper](https://arxiv.org/abs/2510.22257) - Topology-agnostic, NeurIPS 2025
- [EEGPT GitHub](https://github.com/BINE022/EEGPT) - Electrode-wise modeling
- [SingLEM GitHub](https://github.com/ttlabtuat/SingLEM) - Single-channel foundation model
- [Braindecode pretrained models](https://braindecode.org/stable/auto_examples/model_building/plot_load_pretrained_models.html) - Unified loading API
- [BioFoundation project](https://thorirmar.com/project/biofoundation/) - LUNA, FEMBA, CEReBrO
- [EEGMirror (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_EEGMirror_Leveraging_EEG_Data_in_the_Wild_via_Montage-Agnostic_Self-Supervision_ICCV_2025_paper.pdf) - Montage-agnostic
- [EEG-X](https://arxiv.org/html/2511.08861v1) - Device-agnostic foundation model
