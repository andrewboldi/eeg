# EEG-DisGCMAE: Graph Topology Distillation for EEG

## Paper Details

- **Title**: Pre-Training Graph Contrastive Masked Autoencoders are Strong Distillers for EEG
- **Authors**: Xinxu Wei, Kanhao Zhao, Yuan Jiao, Haozhe Xie, Yu Zhang, Lina He
- **Venue**: ICML 2025 (Poster)
- **ArXiv**: [2411.19230](https://arxiv.org/abs/2411.19230)
- **OpenReview**: [YKfJFTiRz8](https://openreview.net/forum?id=YKfJFTiRz8)
- **Code**: [github.com/weixinxu666/EEG_DisGCMAE](https://github.com/weixinxu666/EEG_DisGCMAE) -- **Code is available and complete**

## Problem Statement

Bridge the gap between **high-density (HD) unlabeled** EEG and **low-density (LD) labeled** EEG.
The paper formulates this as graph transfer learning + knowledge distillation: a teacher model
trained on HD montage (e.g., 54-channel) distills knowledge to a student model operating on
LD montage (e.g., 30- or 16-channel).

## Architecture Overview

### Two-Stage Framework

1. **Stage 1 -- GCMAE Pre-Training** (self-supervised, on unlabeled HD data)
   - Combines Graph Contrastive Learning (GCL) with Graph Masked Autoencoder (GMAE)
   - GCL branch: MoCo-style contrastive learning on graph-augmented views
   - GMAE branch: Mask random nodes (electrodes), reconstruct them
   - The two branches are unified -- reconstructed samples are contrasted, contrastive samples are reconstructed

2. **Stage 2 -- Graph Topology Distillation (GTD)** (supervised, teacher -> student)
   - Teacher: Large model pre-trained on HD (54ch) montage
   - Student: Lightweight model trained on LD (30ch, 16ch, or 12ch) montage
   - GTD loss transfers structural topology knowledge, not just logits

### GCN Backbone

From the code (`models/GCN.py`):
- 2-layer GCNConv (PyTorch Geometric) with hidden dim
- Adjacency matrix converted to edge_index + edge_attr for message passing
- scatter_max pooling over nodes -> graph-level representation
- Linear classifier head (64 -> 32 -> num_classes)
- Also supports: DGCNN, EEG-Conformer, Graph Transformer (GFormer), HGNN (hypergraph)

### Channel Configuration

From `EEG_channels_list.py`:
- **Full (HD)**: 54 channels (standard 10-20 extended)
- **LD-32**: 31 channels (subset)
- **LD-16**: 12 channels (C3, C4, F3, F4, Fp1, Fp2, O1, O2, P3, P4, T7, T8)
- `mask_id` lists define which HD electrodes are "removed" to create the LD view

## Graph Topology Distillation (GTD) Loss -- Key Innovation

From `GTD_loss.py`, the loss is called **LSP_Loss** (Local Structure Preserving):

### How It Works

1. **Build kernel matrices** from node features (student and teacher):
   - Compute pairwise distance/similarity between all electrode embeddings
   - Supports euclidean, linear, polynomial, RBF kernels (default: euclidean)

2. **Binarize adjacency matrices** with threshold (default 0.3):
   - `student_adj_binary = (normalized_adj > threshold).float()`
   - Same for teacher

3. **Identify topology-preserving edges**:
   - **Same-density case** (teacher_nodes == student_nodes): edges connected in teacher graph
   - **Cross-density case** (student has fewer nodes): finds both direct connections AND
     indirect connections through removed nodes. If node A and node B in student were both
     connected to a now-removed node C in teacher, they get an "indirect connection" flag.

4. **Compute contrastive topology loss**:
   - Numerator: KL divergence of kernel distributions on edges that ARE connected in teacher
     (weighted by cosine similarity with temperature tau=0.1)
   - Denominator: KL divergence on edges connected in student but NOT in teacher
   - Loss = numerator / denominator (encourages student to preserve teacher's topology)

### Key Equations (from code)

```python
# Kernel matrix from node features
kernel_matrix[i,j] = ||f_i - f_j||_2  # euclidean distance between electrode embeddings

# Topology-aware KL divergence
numerator = KL(softmax(student_kernel) || softmax(teacher_kernel)) * exp(cos_sim / tau)
            masked to edges connected in teacher

denominator = same KL but masked to edges in student NOT in teacher

loss = numerator / denominator
```

### Handling Missing Electrodes

When student has fewer nodes than teacher:
```python
# Find indirect connections through removed nodes
indirect = any(teacher_adj[removed, :student] > 0 AND teacher_adj[:student, removed] > 0)
# Merge with direct teacher connections
connected_or_indirect = direct_teacher_connections | indirect
```

This is the key insight: even when electrodes are physically absent in the LD montage,
the GTD loss preserves the structural relationships that those electrodes mediated.

## Graph Augmentation (for Contrastive Pre-training)

From `graph_augmentation.py`:
- **Node dropping**: Random 20% of nodes zeroed out
- **Edge removal**: Random 20% of edges removed from adjacency
- **Feature noise**: Gaussian noise (std=0.1) added to node features
- **MoCo-style**: Query and key get different random augmentations (different seeds)
- **Masked node replacement**: Dropped nodes replaced with learnable embeddings

## Pre-training Details

From `train_cl_mae_moco.py`:
- **Batch size**: 64
- **Epochs**: 7 (pre-training is short)
- **Learning rate**: 0.002, Adam (beta1=0.5, beta2=0.999)
- **Teacher nodes**: 54 (HD), Student nodes: 31 (LD)
- **Feature dim**: 128
- **Losses**: CrossEntropy (cls) + MSE (reconstruction) + L1 + NTXent (contrastive)
- **Evaluation**: 10-fold cross-validation
- **Datasets**: EMBARC (depression EEG), HBN (pediatric EEG)

## Dependencies

- PyTorch
- PyTorch Geometric (torch_geometric)
- torch_scatter
- scikit-learn
- numpy, matplotlib, seaborn

## Datasets Used

- **EMBARC**: Clinical depression EEG, 54-channel, resting-state
- **HBN**: Healthy Brain Network, pediatric EEG, up to 128 channels
- **Tasks**: All classification (depression detection, sex classification, ASD, anxiety, HAMD score)

## Relevance to Our Scalp-to-In-Ear Problem

### Direct Mapping

| Their Problem | Our Problem |
|--------------|-------------|
| 54ch HD scalp -> 30/16ch LD scalp | 46ch scalp -> 12ch in-ear |
| Teacher on HD, student on LD | Teacher on scalp, student on in-ear |
| Missing electrodes (subset) | Completely different electrode locations |
| Classification tasks | Regression (signal prediction) |

### What We Can Adapt

1. **GTD Loss for Regression**: Their LSP_Loss is task-agnostic -- it operates on node embeddings,
   not on task outputs. We could use it as an auxiliary loss during training:
   - Train a 46ch scalp encoder (teacher)
   - Train a 12ch in-ear decoder (student)
   - GTD loss encourages the in-ear decoder to preserve the spatial topology learned by the scalp encoder
   - Main loss remains MSE/correlation for signal reconstruction

2. **Graph Construction**: Build adjacency matrices from 3D electrode coordinates:
   - Scalp: 46 electrodes with known 10-20 positions
   - In-ear: 12 electrodes with approximate positions (L/R ear canal)
   - Cross-montage edges: connect in-ear electrodes to nearest scalp electrodes

3. **Masked Pre-training**: Pre-train on scalp data by masking electrodes near ear positions,
   forcing the model to reconstruct "ear-like" signals from surrounding scalp channels

4. **Contrastive Augmentation**: Use their graph augmentation (node drop, edge remove, noise)
   during training to improve robustness

### Key Challenges for Adaptation

1. **Different electrode locations**: Their LD channels are a SUBSET of HD channels.
   Our in-ear channels are at completely different locations. The `removed_nodes` / indirect
   connection logic doesn't directly apply -- we'd need a cross-montage adjacency matrix.

2. **Regression vs Classification**: Their entire pipeline ends with CrossEntropy.
   We need MSE/correlation. The GTD loss itself is fine (feature-level), but the
   task head and pre-training objectives need modification.

3. **No pre-training data**: They pre-train on large unlabeled datasets (EMBARC, HBN).
   We have only 15 subjects. Could potentially pre-train on other EEG datasets
   (MOABB, TUH) and fine-tune on Ear-SAAD.

4. **Scale**: Their models are designed for 128-dim features per electrode with GCN message
   passing. Our current best is a 7-tap FIR filter. The GCN approach requires substantially
   more compute and may overfit on 12 training subjects.

### Proposed Adaptation Strategy

**Minimal viable approach** (can test without GPU):
1. Build scalp adjacency matrix from electrode distances (46x46)
2. Build cross-montage adjacency (46 scalp x 12 in-ear) from 3D coordinates
3. Use GTD loss as auxiliary loss alongside our existing FIR + MSE/corr training
4. Teacher = frozen closed-form spatial filter; Student = learnable GCN spatial filter
5. GTD loss encourages GCN to preserve inter-electrode relationships from closed-form solution

**Full approach** (needs GPU + more data):
1. Pre-train GCMAE on large scalp-only dataset (e.g., TUH EEG Corpus)
2. Fine-tune with GTD distillation: teacher=scalp encoder, student=in-ear decoder
3. Use graph augmentation for robustness
4. Multi-task: reconstruct in-ear signals + preserve topology

## Sources

- [ArXiv Paper](https://arxiv.org/abs/2411.19230)
- [ICML 2025 Poster](https://icml.cc/virtual/2025/poster/44482)
- [OpenReview](https://openreview.net/forum?id=YKfJFTiRz8)
- [GitHub Repository](https://github.com/weixinxu666/EEG_DisGCMAE)
