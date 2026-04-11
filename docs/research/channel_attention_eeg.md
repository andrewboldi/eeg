# Channel Attention Mechanisms for EEG Deep Learning

Research survey compiled 2026-04-11 for Ear-SAAD scalp-to-in-ear prediction project.

## Problem Context

Our current best model (iter030, r=0.378) uses a learned FIR spatial-temporal filter that treats all 46 input channels (27 scalp + 19 cEEGrid) equally. Subject analysis shows large variability in per-channel predictiveness: Subject 13 achieves r~0.46 while Subject 14 gets r~0.27. A channel attention mechanism could learn to weight informative channels higher and suppress noisy ones, potentially breaking the 0.378 plateau.

---

## 1. Squeeze-and-Excitation (SE) Blocks for EEG

**Paper**: Li et al., "A Multi-Branch CNN with Squeeze-and-Excitation Attention Blocks for EEG-Based Motor Imagery Classification" (Biosensors 2022, PMC9032940)

**Mechanism**:
1. Global Average Pooling across the temporal dimension to get a per-channel descriptor (C x 1)
2. Two FC layers with reduction ratio r: FC(C, C/r) -> ReLU -> FC(C/r, C) -> Sigmoid
3. Output sigmoid weights (0-1) are multiplied element-wise with the input feature map

**Results**: MBEEGSE achieves 82.87% on BCI-IV2a (4-class MI), 96.15% on High Gamma dataset. The SE block adaptively recalibrates channel-wise feature responses by modeling channel interdependencies.

**Relevance to our project**: Directly applicable. After the spatial convolution in our FIR model, an SE block could learn which output channels (in-ear targets) to emphasize. More importantly, an SE block on the INPUT channels could learn which scalp channels are most predictive.

**Key concern**: The dimensionality reduction in SE (C -> C/r -> C) destroys the direct correspondence between a channel and its weight, which can hurt performance for small channel counts like ours (27 or 46).

---

## 2. Efficient Channel Attention (ECA) -- Preferred for Small Channel Counts

**Paper**: Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" (CVPR 2020, arXiv:1910.03151)
**EEG application**: Liu et al., "A learnable EEG channel selection method for MI-BCI using efficient channel attention" (Frontiers in Neuroscience 2023, PMC10622956)

**Mechanism**:
1. Global Average Pooling across time to get per-channel descriptor
2. 1D convolution with adaptive kernel size k across channels (NO dimensionality reduction)
3. Sigmoid activation to produce channel weights

The kernel size k is determined adaptively:
```
k = |log2(C) / gamma + b / gamma|_odd
```
where C = number of channels, gamma=2, b=1. For C=27, k ~ 3; for C=46, k ~ 5.

**Critical difference from SE**: ECA avoids dimensionality reduction entirely. Each channel weight is computed from a local neighborhood of channels via 1D conv, preserving the direct channel-to-weight correspondence.

**EEG-specific modification**: When the number of EEG channels is small and global cross-channel interactions are needed, the 1D conv can be replaced with a full FC layer (still no bottleneck).

**Results**: ECA-based channel selection achieved 75.76% accuracy (22 channels) and 69.52% (8 selected channels) on 4-class MI-BCI, outperforming other channel selection methods.

**Relevance to our project**: STRONG FIT. Our 27-46 input channels are small enough that a single FC layer (no bottleneck) can compute per-channel attention weights. This is essentially a learnable diagonal scaling matrix applied before the spatial filter -- very few extra parameters.

---

## 3. EEG Channel-Wise Attention Module (ECWAM)

**Paper**: "Enhancing the performance of a deep CNN model for motor imagery classification using EEG channel-wise attention module" (PubMed 41642225, 2025)

**Mechanism**:
1. Compute per-channel score based on mu-band power for each EEG channel
2. Amplify prominent EEG channels based on their scores
3. Integrate with deep CNN for MI decoding

**Results**: Improved average classification accuracy from 63.96% to 68.98% on subject-specific MI tasks.

**Relevance**: The band-specific scoring is interesting but our data is already band-limited (1-9 Hz). The general principle of amplifying informative channels before the main model is directly applicable.

---

## 4. PSAEEGNet -- Pyramid Squeeze Attention

**Paper**: "PSAEEGNet: pyramid squeeze attention mechanism-based CNN for single-trial EEG classification in RSVP task" (Frontiers in Human Neuroscience 2024)

**Mechanism**: Multi-scale channel attention that processes features at different spatial resolutions simultaneously, then fuses the attention maps. This captures both local and global channel relationships.

**Relevance**: Potentially over-engineered for our 27-46 channels. The multi-scale aspect is more useful for high-density montages (64+ channels).

---

## 5. CBAM: Dual Channel + Spatial Attention

**Paper**: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018, arXiv:1807.06521)

**Mechanism**: Sequential application of:
1. **Channel attention**: MaxPool + AvgPool across spatial dims -> shared MLP -> sigmoid weights
2. **Spatial attention**: MaxPool + AvgPool across channel dim -> conv -> sigmoid mask

**EEG applications**: Used in CBAM-DeepConvNet for visual evoked potential recognition. Also adapted for emotion recognition with EEG.

**Relevance**: The dual attention (channel + spatial/temporal) could be useful. Channel attention selects which EEG channels matter; temporal attention selects which time lags matter. However, most EEG work uses CBAM for classification, not regression.

---

## 6. Subject-Adaptive Channel Attention

### 6a. Few-Shot Fine-Tuning with Channel Attention

**Paper**: He et al., "Dual Attention Relation Network With Fine-Tuning for Few-Shot EEG Motor Imagery Classification" (IEEE TNSRE 2023, PubMed 37379192)

**Approach**: Pre-train a model with channel attention on multiple subjects, then fine-tune only the attention weights on a few samples from the target subject. The temporal-attention and aggregation-attention modules adapt to each subject's distribution.

### 6b. Composable Channel-Adaptive Architecture

**Paper**: "A Composable Channel-Adaptive Architecture for Seizure Classification" (arXiv:2512.19123, Dec 2024)

**Key innovation**: 
1. Process each channel independently through a shared encoder
2. Fuse channel features using a **trainable scalar per channel** (holographic channel fusion)
3. Accumulate in long-term memory for classification

The trainable-scalar-per-channel is the simplest possible channel attention -- just a learnable vector of weights, one per channel. Despite its simplicity, it works well because:
- It can be pre-trained across subjects with different channel configurations
- Fine-tuning requires 5x fewer epochs than training from scratch
- It naturally handles missing/noisy channels by driving their weights toward zero

### 6c. Adaptive Channel Mixing Layer (ACML)

Dynamically adjusts input signal weights through a learnable transformation matrix based on inter-channel correlations. This goes beyond scalar per-channel weights to learn channel interactions.

### 6d. Domain Adaptation with Channel Attention

**Paper**: SDA-FSL approach integrating CBAM with domain adaptation and Prototypical Networks for cross-subject EEG emotion recognition.

**Relevance to our project**: VERY HIGH. Our LOSO setup (train on subjects 1-12, test on 13-15) is exactly the cross-subject transfer problem. Per-subject channel attention that adapts to each subject's unique electrode impedances and cortical geometry could directly address the Subject 14 problem (consistently low r).

---

## 7. Graph-Based Channel Attention

**Paper**: STGAT-CS (Spatio-Temporal Graph Attention Network for Channel Selection, Cognitive Neurodynamics 2024)

**Mechanism**: Models EEG channels as nodes in a graph, with edges representing spatial/functional connectivity. Graph attention learns which channel-to-channel relationships matter for the task. Channel selection emerges from the learned attention weights.

**Relevance**: Interesting but likely overkill for our linear FIR model. Would be more relevant if we move to a deeper architecture.

---

## Recommended Implementation Plan for Our Project

### Option A: Minimal -- Learnable Channel Weights (1 line of code)
```python
# Add a learnable weight vector before spatial filter
self.channel_attn = nn.Parameter(torch.ones(n_input_channels))

# In forward:
x = x * torch.sigmoid(self.channel_attn).unsqueeze(0).unsqueeze(-1)
# Then apply existing FIR spatial filter
```
- Adds only 27-46 parameters
- Sigmoid ensures weights are in (0, 1)
- Channels can be driven toward zero if uninformative
- Can inspect learned weights to understand channel importance

### Option B: ECA-style -- 1D Conv Channel Attention
```python
class ECABlock(nn.Module):
    def __init__(self, n_channels, k=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, time)
        y = self.avg_pool(x)  # (batch, channels, 1)
        y = y.squeeze(-1).unsqueeze(1)  # (batch, 1, channels)
        y = self.conv(y)  # local cross-channel interaction
        y = y.squeeze(1).unsqueeze(-1)  # (batch, channels, 1)
        return x * self.sigmoid(y)
```
- Adds k parameters (k=3 to 5 for our channel count)
- Captures local channel neighborhood effects (nearby electrodes)
- Still very lightweight

### Option C: Subject-Conditioned Attention
```python
class SubjectChannelAttn(nn.Module):
    def __init__(self, n_channels, n_subjects=12):
        super().__init__()
        # Shared base weights + per-subject offset
        self.base = nn.Parameter(torch.zeros(n_channels))
        self.subject_offset = nn.Embedding(n_subjects, n_channels)
    
    def forward(self, x, subject_id):
        weights = self.base + self.subject_offset(subject_id)
        return x * torch.sigmoid(weights).unsqueeze(0).unsqueeze(-1)
```
- Learns which channels each subject should trust
- During test on unseen subjects, uses only the base weights (or adapts with a few samples)
- Directly addresses the cross-subject variability problem

### Recommended iteration sequence:
1. **iter045**: Option A (learnable scalar weights) -- simplest test of the hypothesis
2. **iter046**: Option B (ECA block) -- if A shows promise, add cross-channel interaction
3. **iter047**: Option C (subject-conditioned) -- if channel attention helps, make it subject-adaptive

---

## Key References

### Channel Attention Architectures
- [SE-Net: Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (Hu et al., CVPR 2018)
- [ECA-Net: Efficient Channel Attention](https://arxiv.org/abs/1910.03151) (Wang et al., CVPR 2020)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) (Woo et al., ECCV 2018)

### EEG-Specific Applications
- [Multi-Branch CNN with SE for EEG MI](https://pmc.ncbi.nlm.nih.gov/articles/PMC9032940/) (Li et al., Biosensors 2022)
- [Learnable EEG Channel Selection via ECA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10622956/) (Liu et al., Frontiers 2023)
- [ECWAM: EEG Channel-Wise Attention Module](https://pubmed.ncbi.nlm.nih.gov/41642225/) (2025)
- [PSAEEGNet: Pyramid Squeeze Attention for EEG](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1385360/full) (2024)
- [Composable Channel-Adaptive Architecture](https://arxiv.org/abs/2512.19123) (Dec 2024)

### Cross-Subject Adaptation
- [Dual Attention Relation Network for Few-Shot EEG](https://pubmed.ncbi.nlm.nih.gov/37379192/) (IEEE TNSRE 2023)
- [STGAT-CS: Graph Attention Channel Selection](https://link.springer.com/article/10.1007/s11571-024-10154-5) (Cognitive Neurodynamics 2024)
- [Subject-Adaptive Transfer Learning via Resting State EEG](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_63) (2024)
- [Channel Attention for Subject-Independent Emotion Recognition](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2025.1494369/full) (Frontiers 2025)
