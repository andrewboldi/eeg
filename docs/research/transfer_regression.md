# Transfer Learning for EEG Signal Regression/Reconstruction

## Research Summary (April 2025)

This document surveys transfer learning approaches specifically for EEG **regression** (continuous value prediction), not classification. The goal is to identify strategies applicable to our scalp-to-in-ear EEG reconstruction task.

---

## 1. EEG Foundation Models That Support Regression

### REVE (2025) — Best Candidate
- **Paper**: [arXiv 2510.21585](https://arxiv.org/abs/2510.21585)
- **Code**: [github.com/elouayas/reve_eeg](https://github.com/elouayas/reve_eeg)
- **Also in**: [braindecode.models.REVE](https://braindecode.org/stable/generated/braindecode.models.REVE.html)
- **Architecture**: Patch embedding + 4D positional encoding (uses true 3D electrode coordinates + temporal index) + Transformer backbone
- **Pretraining**: Masked autoencoder (MAE) on 60,000+ hours from 92 datasets, 25,000 subjects
- **Key innovation**: 4D positional encoding enables adaptation to *any* electrode montage (arbitrary channel count and placement)
- **Regression support**: Yes — can replace classification head with regression head
- **Fine-tuning strategy**: Two-step: (1) linear probe with frozen encoder, (2) unfreeze encoder + LoRA for full fine-tuning
- **Why relevant**: Handles arbitrary electrode setups, pretrained on massive EEG data, state-of-the-art linear probing results

### BIOT (NeurIPS 2023)
- **Paper**: [arXiv 2305.10351](https://arxiv.org/abs/2305.10351)
- **Code**: [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT)
- **Also in**: [braindecode.models.BIOT](https://braindecode.org/dev/generated/braindecode.models.BIOT.html)
- **Architecture**: Per-channel tokenization into fixed-length segments + channel embeddings + relative position embeddings + Transformer
- **Pretraining**: Cross-dataset learning across EEG, ECG, and activity signals
- **Key innovation**: Handles mismatched channels, variable lengths, and missing values by tokenizing each channel separately
- **Regression support**: Yes — unified sentence structure allows regression head
- **Performance**: ~3-4% improvement over baselines on seizure detection after pretraining

### LaBraM (ICLR 2024 Spotlight)
- **Paper**: [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)
- **Code**: [github.com/935963004/LaBraM](https://github.com/935963004/LaBraM)
- **Architecture**: EEG channel patches + neural tokenizer (vector-quantized spectrum prediction) + masked prediction
- **Pretraining**: 2,500 hours across ~20 datasets
- **Regression support**: Yes — validated on gait prediction (a regression task)
- **Fine-tuning**: Replace task head; fine-tuning scripts provided for custom datasets

### CBraMod (ICLR 2025)
- **Paper**: [arXiv 2412.07236](https://arxiv.org/abs/2412.07236)
- **Code**: [github.com/wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod)
- **Architecture**: Criss-cross transformer (parallel spatial and temporal attention) + asymmetric conditional positional encoding
- **Pretraining**: Patch-based masked EEG reconstruction on large corpus
- **Regression support**: Yes — uses Pearson r, R2, and RMSE as regression metrics
- **Fine-tuning**: Flatten learned representations -> task-specific regression head

### CSBrain (2025)
- **Paper**: [arXiv 2506.23075](https://arxiv.org/html/2506.23075v1)
- **Architecture**: Cross-scale tokens aggregating multi-resolution info within temporal windows and anatomical brain regions
- **Regression support**: Yes — explicitly uses MSE loss for regression tasks
- **Performance**: Outperforms CBraMod, LaBraM, and BIOT by 3-8% on aggregate benchmarks

### EEG2Vec (2023)
- **Paper**: [arXiv 2305.13957](https://arxiv.org/abs/2305.13957)
- **Architecture**: CNN + Transformer with contrastive + reconstruction self-supervised pretraining
- **Regression support**: Yes — validated on ICASSP 2023 Auditory EEG Challenge regression task
- **Key finding**: Contrastive pretraining produced more informative representations than reconstruction pretraining for downstream regression

### BENDR (2021)
- **Paper**: Adapted from Wav2vec 2.0 for EEG
- **Architecture**: Convolutional encoder + Transformer with contrastive learning
- **Regression support**: Limited — primarily designed for classification, but encoder features usable for regression

### EEGPT (NeurIPS 2024)
- **Paper**: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf)
- **Architecture**: Dual self-supervised learning (spatio-temporal alignment + mask-based reconstruction)
- **Regression support**: Yes — universal pretrained transformer for multiple downstream tasks

---

## 2. Fine-Tuning Strategies Compared

### Strategy A: Linear Probing (Frozen Encoder)
- Freeze all encoder weights, train only a new regression head
- **Pros**: Fast, no overfitting risk, works with very small datasets
- **Cons**: Often insufficient — linear probing performs significantly worse than fine-tuning for most EEG tasks
- **When to use**: Initial baseline; when dataset is extremely small (<100 samples); to validate representation quality

### Strategy B: Two-Step Fine-Tuning (REVE's approach — RECOMMENDED)
1. **Step 1**: Freeze encoder, train linear regression head (5-20 epochs)
2. **Step 2**: Unfreeze encoder, fine-tune entire network with lower learning rate + LoRA
- **Pros**: Best of both worlds — stable initialization + task-specific adaptation
- **Cons**: More complex training pipeline
- **Evidence**: REVE achieves SOTA with this approach across 10 downstream tasks

### Strategy C: Full Fine-Tuning
- Unfreeze everything from the start, train end-to-end
- **Pros**: Maximum flexibility for task adaptation
- **Cons**: Risk of catastrophic forgetting; overfitting on small datasets
- **When to use**: When dataset is large enough (>1000 samples per subject)

### Strategy D: Partial Fine-Tuning (Selective Layer Unfreezing)
- Freeze early layers (general features), unfreeze later layers (task-specific)
- **Pros**: Good balance; early layers retain general EEG knowledge
- **Cons**: Requires tuning which layers to unfreeze
- **Common approach**: Freeze first N transformer layers, unfreeze last 1-2

### Strategy E: Parameter-Efficient Fine-Tuning (LoRA/Adapters)
- Insert small trainable adapter modules; keep most weights frozen
- **Paper**: [Graph Adapter for EEG Foundation Models](https://arxiv.org/html/2411.16155v2)
- **Pros**: Very few trainable parameters; minimal overfitting; fast
- **Cons**: May not capture large domain shifts
- **When to use**: When pretrained model is large and target data is small

### Key Finding from EEG-FM-Bench (2025)
> "Compact architectures with domain-specific inductive biases consistently outperform significantly larger models."

This suggests that for our narrow task (27ch -> 12ch spatial mapping), a smaller pretrained model fine-tuned with the two-step strategy may beat a huge foundation model.

---

## 3. Critical Findings for Our Task

### What Works
1. **Two-step fine-tuning** (linear probe then full fine-tune) is the consensus best practice
2. **Contrastive pretraining** produces more informative features than reconstruction-only pretraining (EEG2Vec finding)
3. **Channel-agnostic architectures** (REVE, BIOT) handle our 27-input / 12-output setup naturally
4. **LoRA** keeps trainable parameter count low, critical for our small dataset (12 training subjects)

### What Doesn't Work
1. **Linear probing alone** is consistently weak — frozen encoders don't capture task-specific nuances
2. **Large models without domain-specific inductive bias** underperform compact specialized models
3. **Foundation models don't always beat simple baselines** on small/specialized datasets (EEG-FM-Bench finding)

### Scalp-to-Ear Specific
- A 2024 study on [scalp-to-ear-EEG transfer for sleep scoring](https://pmc.ncbi.nlm.nih.gov/articles/PMC11100860/) showed fine-tuning pretrained scalp models significantly improved ear-EEG prediction
- Substantial agreement between scalp and ear-EEG was confirmed, supporting the feasibility of cross-modality transfer
- However, pretrained models need adaptation for unique ear-EEG signal characteristics

---

## 4. Recommended Approach for Our Project

### Option A: Fine-Tune REVE (Most Promising)
```
1. Load REVE pretrained weights (available via braindecode or HuggingFace)
2. Replace classification head with 12-channel regression head
3. Configure 4D positional encoding for our 27 scalp electrode positions
4. Step 1: Linear probe (freeze encoder, train regression head, 20 epochs)
5. Step 2: Full fine-tune with LoRA (unfreeze, lr=1e-5, 100 epochs)
6. Evaluate on LOSO benchmark
```
- **Why**: 4D positional encoding handles our exact electrode layout; pretrained on 25K subjects
- **Risk**: May not help if the mapping is truly linear (our 0.378 ceiling)

### Option B: Fine-Tune BIOT (Simpler)
```
1. Load BIOT pretrained weights
2. Per-channel tokenization fits our variable-quality channels
3. Add regression head for 12 in-ear channels
4. Fine-tune with combined MSE+correlation loss
```
- **Why**: Handles missing channels gracefully; simpler architecture
- **Risk**: Less pretrained data than REVE

### Option C: Self-Supervised Pretraining on Scalp-Only Data
```
1. Use MAE-style pretraining on scalp EEG from multiple public datasets
2. Learn general scalp EEG representations
3. Fine-tune decoder to predict in-ear channels
```
- **Why**: Most task-specific; could learn EEG dynamics relevant to our mapping
- **Risk**: Need to curate pretraining data; engineering effort

### Key Consideration
Our current best (r=0.378) uses a simple 7-tap FIR filter. The fundamental question is whether pretrained features can capture **nonlinear** cross-subject patterns that linear filters miss. If the mapping is truly linear in this frequency band (1-9 Hz), transfer learning may not help. The broadband (1-45 Hz) setting is where transfer learning is most likely to break through the plateau.

---

## 5. Implementation Priority

| Priority | Approach | Expected Impact | Effort |
|----------|----------|----------------|--------|
| 1 | REVE fine-tune on broadband data | High | Medium (braindecode integration) |
| 2 | BIOT fine-tune on narrowband data | Medium | Low (simpler API) |
| 3 | CBraMod with regression head | Medium | Medium |
| 4 | Custom MAE pretraining on public EEG | High | High |
| 5 | EEG2Vec contrastive pretraining | Medium | High |

---

## Sources

- [REVE paper](https://arxiv.org/abs/2510.21585) — Foundation model with 4D positional encoding, 25K subjects
- [BIOT paper](https://arxiv.org/abs/2305.10351) — Cross-data biosignal transformer
- [LaBraM paper (ICLR 2024)](https://github.com/935963004/LaBraM) — Large brain model with neural tokenizer
- [CBraMod paper (ICLR 2025)](https://arxiv.org/abs/2412.07236) — Criss-cross brain foundation model
- [CSBrain paper](https://arxiv.org/html/2506.23075v1) — Cross-scale spatiotemporal model
- [EEG2Vec paper](https://arxiv.org/abs/2305.13957) — Self-supervised EEG representation learning
- [EEGPT (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf) — Pretrained transformer for EEG
- [EEG-FM-Bench](https://arxiv.org/abs/2508.17742) — Comprehensive benchmark of EEG foundation models
- [EEG Foundation Models Critical Review](https://arxiv.org/html/2507.11783v3) — Survey of current progress
- [Graph Adapter for EEG](https://arxiv.org/html/2411.16155v2) — Parameter-efficient fine-tuning
- [Scalp-to-Ear-EEG Transfer](https://pmc.ncbi.nlm.nih.gov/articles/PMC11100860/) — Transfer learning for ear-EEG
- [Are EEG Foundation Models Worth It?](https://openreview.net/forum?id=5Xwm8e6vbh) — Comparison with traditional decoders
