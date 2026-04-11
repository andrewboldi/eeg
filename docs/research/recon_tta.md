# Input Reconstruction for Test-Time Adaptation in EEG Regression

## Problem Statement

Our biggest unsolved problem is **test-time adaptation (TTA) for regression**. Standard TTA methods (Tent, SHOT, etc.) minimize prediction entropy, but entropy is undefined for scalar regression outputs. Cross-subject variability is the dominant bottleneck (Subject 14 consistently ~0.27 r vs Subject 13 ~0.46 r), and no amount of model capacity or training tricks has broken the r=0.378 plateau.

**Core idea**: Use input (scalp EEG) reconstruction as a self-supervised proxy objective at test time. The encoder can be adapted to a new subject's scalp distribution without needing in-ear labels.

---

## Literature Review

### 1. TTA for Regression: The Fundamental Challenge

**SSA: Test-time Adaptation for Regression by Subspace Alignment** (Adachi et al., ICLR 2025)
- [arXiv](https://arxiv.org/abs/2410.03263) | [OpenReview](https://openreview.net/forum?id=SXtl7NRyE5)
- Key insight: Naive feature alignment (as used in classification TTA) is **ineffective or harmful** for regression because features concentrate in a small subspace, and many raw feature dimensions have little significance to the output.
- Method: (1) Detect the feature subspace significant to regression output, (2) weight dimensions by output significance, (3) align only the significant subspace between source and target.
- Uses penultimate-layer features (2048-dim) projected to K=100 significant subspace.
- **Relevance**: Confirms that regression TTA needs special handling. Their subspace approach is complementary to reconstruction-based TTA.

**Q-PART** (CVPR 2025): Quasi-periodic adaptive regression with test-time training for medical imaging regression. Confirms growing interest in regression-specific TTA.

### 2. Autoencoder-Based Self-Supervised TTA

**He et al. (Medical Image Analysis, 2021)**
- [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8316425/) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1361841521001821)
- Architecture: Task model + autoencoder + adaptors. The autoencoder reconstruction loss serves as a **source domain similarity measurement**.
- At test time: Optimize adaptors to minimize reconstruction loss, transforming domain-shifted features back toward the source domain.
- Key finding: Adding autoencoders and adaptors gives the task model resistance to **unknown** test domain shifts.
- **Directly applicable**: This is essentially our proposed approach. Their medical imaging setting has similar properties (regression-like dense prediction, subject variability).

### 3. TTT-MAE: Test-Time Training with Masked Autoencoders

**Gandelsman et al. (NeurIPS 2022)**
- [Paper](https://papers.neurips.cc/paper_files/paper/2022/file/bcdec1c2d60f94a93b6e36f937aa0530-Paper-Conference.pdf) | [Project](https://yossigandelsman.github.io/ttt_mae/)
- Method: Train MAE alongside main task with shared encoder. At test time, adapt encoder by minimizing MAE reconstruction loss on each test sample.
- Masks 75% of input patches; uses SGD; performance **keeps improving even after 20 optimization steps**.
- Y-shaped architecture: shared encoder -> (main task head, reconstruction head).
- **Directly applicable**: The masked reconstruction approach could work for EEG channels or time segments.

**T4P: Test-Time Training for Trajectory Prediction** (Park et al., CVPR 2024)
- [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_T4P_Test-Time_Training_of_Trajectory_Prediction_via_Masked_Autoencoder_and_CVPR_2024_paper.pdf)
- Extends TTT-MAE to **regression** (trajectory prediction).
- Model is optimized from both regression and reconstruction loss simultaneously.
- Demonstrates that reconstruction-based TTT works for regression tasks, not just classification.

### 4. EEG-Specific Test-Time Adaptation

**NeuroTTT** (2025)
- [arXiv](https://arxiv.org/abs/2509.26301)
- First to unify domain-tuned self-supervision with test-time training for EEG foundation models.
- Two-stage: (1) Domain-specific self-supervised fine-tuning with auxiliary pretext tasks (stopped-band prediction + temporal jigsaw), (2) Test-time training on individual unlabeled test samples.
- Achieves SOTA across diverse BCI tasks (imagined speech, stress detection, motor imagery).
- **Highly relevant**: Proves TTT works for EEG specifically. Their pretext tasks are classification-oriented; reconstruction would be more natural for our regression setting.

**DiMAE: Domain-Invariant Masked AutoEncoder**
- [arXiv](https://arxiv.org/abs/2205.04771)
- Cross-domain reconstruction task: augment inputs with style noise from different domains, reconstruct original from augmented embedding.
- Learns domain-invariant features by forcing the encoder to "undo" domain-specific variations.
- **Applicable**: Could augment scalp EEG with subject-specific noise patterns and train reconstruction to remove them.

### 5. Auxiliary Reconstruction for Domain Adaptation

**Point-TTA** (Hatem et al., ICCV 2023)
- Uses 3D point cloud reconstruction as auxiliary task for TTA.
- Key insight: Adapting model parameters at test time using reconstruction auxiliary loss lets the model exploit **internal features of the test instance** before the primary task.

**General pattern across papers**: Self-supervised reconstruction objectives provide a reliable, label-free adaptation signal that is **complementary** to the primary task loss.

---

## Proposed Method: Reconstruction-TTA for EEG

### Architecture

```
Scalp EEG (B, 27, T)
       |
   [Encoder]  -- shared, trainable at test time
       |
   Latent z (B, H, T)
      / \
     /   \
[Decoder_pred]    [Decoder_recon]
     |                  |
In-ear pred        Scalp reconstruction
(B, 12, T)         (B, 27, T)
```

### Training Phase (Subjects 1-12)

Joint loss:
```
L_train = L_pred(y_hat, y_inear) + lambda * L_recon(x_hat, x_scalp)
```

Where:
- `L_pred`: Combined MSE + correlation loss (our current best)
- `L_recon`: MSE reconstruction of scalp input
- `lambda`: Weighting hyperparameter (start with 0.1-0.5)

Both decoders share the encoder. The reconstruction decoder forces the encoder to preserve rich scalp EEG representations rather than collapsing to a minimal prediction-sufficient subspace.

### Test-Time Adaptation (Subjects 13-15)

For each test subject:
1. Freeze `Decoder_pred` (prediction head)
2. Collect all test windows for that subject (no labels needed)
3. Run N gradient steps minimizing only `L_recon` on the test subject's scalp data
4. This adapts the encoder to the test subject's scalp distribution
5. Use adapted encoder + frozen prediction head for final predictions

### Key Design Decisions

**What to adapt:**
- Option A: Adapt only encoder (safest, prevents prediction head drift)
- Option B: Adapt encoder + prediction head BN stats only (SSA-inspired)
- Option C: Adapt encoder with very low LR, small number of steps

**Reconstruction target variants:**
1. **Full reconstruction**: Reconstruct all 27 scalp channels (simplest)
2. **Masked reconstruction**: Mask random channels/time segments, reconstruct missing (TTT-MAE style, stronger self-supervision signal)
3. **Denoised reconstruction**: Add noise to input, reconstruct clean version (denoising autoencoder, may learn better representations)

**Encoder architecture options:**
- Minimal: FIR filter bank (7-tap per channel) -> spatial mixing -> latent
- Medium: 1D conv encoder with 2-3 layers
- Our current best model is essentially a linear FIR filter. The encoder needs enough capacity to benefit from adaptation but not so much that it overfits during TTA.

### Why This Should Work for EEG

1. **Subject variability is primarily in the scalp distribution**: Different skull thickness, electrode impedance, cortical folding cause subject-specific scalp patterns. Reconstruction loss directly measures how well the encoder models these patterns.

2. **The mapping from cortex to in-ear is more stable**: The physics of volume conduction from cortex to in-ear is similar across subjects. The main variability is in the scalp-side pickup, not the cortex-to-ear transfer function.

3. **We have abundant unlabeled scalp data at test time**: Each test subject provides hundreds of windows of scalp EEG. This is far more data than needed for a few adaptation steps.

4. **Linear models may not benefit**: Our current FIR model has very few parameters. The reconstruction-TTA approach may require a slightly larger encoder to have parameters worth adapting. This could break our "linear is sufficient" finding -- but that finding was *without* TTA.

### Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Reconstruction loss doesn't correlate with prediction quality | Monitor both losses during training; ensure reconstruction loss varies across subjects in a way that correlates with prediction difficulty |
| TTA overfits to test subject (catastrophic adaptation) | Use very few steps (5-20), low LR, and early stopping on reconstruction loss convergence |
| Added model complexity hurts generalization | Keep reconstruction decoder lightweight (single linear layer or 1-layer conv) |
| Lambda tuning is fragile | Use gradient normalization or uncertainty weighting (Kendall et al. 2018) to auto-balance losses |
| Encoder capacity too low for meaningful adaptation | This is the main risk with our current 7-tap FIR. May need to increase to a small conv encoder. |

---

## Implementation Plan

### Phase 1: Proof of Concept (iter045)
- Architecture: Small conv encoder (2 layers, H=64) + linear prediction head + linear reconstruction head
- Training: Joint loss with lambda=0.1
- TTA: 10 SGD steps on reconstruction loss, LR=1e-4
- Adapt only encoder weights
- Compare with and without TTA

### Phase 2: Ablations (iter046-048)
- Vary lambda (0.01, 0.1, 0.5, 1.0)
- Vary TTA steps (1, 5, 10, 20, 50)
- Vary TTA LR (1e-5, 1e-4, 1e-3)
- Compare full vs masked vs denoising reconstruction
- Compare adapting encoder-only vs encoder+BN

### Phase 3: Integration with Best Model (iter049)
- Apply reconstruction-TTA to current best FIR model
- May require architectural changes to FIR to add reconstruction path

---

## Related Work in This Project

- `docs/research/test_time_adaptation.md` -- previous TTA research notes
- `models/iter041_subject_adapt.py` -- RevIN + adaptive BN approach (normalization-based TTA)
- `models/iter042_euclidean_align.py` -- Euclidean alignment (statistical TTA, hurt performance)
- Key finding from iter042: Euclidean alignment **hurt** performance, likely because batch covariance estimates were noisy. Reconstruction-TTA may be more robust because it uses gradient-based adaptation rather than statistical estimation.

---

## Key References

1. **SSA** (Adachi et al., ICLR 2025): Test-time adaptation for regression by subspace alignment. [arXiv:2410.03263](https://arxiv.org/abs/2410.03263)
2. **TTT-MAE** (Gandelsman et al., NeurIPS 2022): Test-time training with masked autoencoders. [arXiv:2209.07522](https://arxiv.org/abs/2209.07522)
3. **T4P** (Park et al., CVPR 2024): Test-time training of trajectory prediction via masked autoencoder. [CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_T4P_Test-Time_Training_of_Trajectory_Prediction_via_Masked_Autoencoder_and_CVPR_2024_paper.pdf)
4. **He et al. (MedIA 2021)**: Autoencoder-based self-supervised test-time adaptation. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8316425/)
5. **NeuroTTT** (2025): Test-time training for EEG foundation models. [arXiv:2509.26301](https://arxiv.org/abs/2509.26301)
6. **DiMAE** (2022): Domain-invariant masked autoencoders. [arXiv:2205.04771](https://arxiv.org/abs/2205.04771)
7. **Point-TTA** (Hatem et al., ICCV 2023): Multitask meta-auxiliary TTA for point clouds.

---

## Verdict

**Strong yes -- implement this.** The literature strongly supports reconstruction-based TTA for regression:

- T4P (CVPR 2024) proves reconstruction-TTT works for regression tasks
- NeuroTTT (2025) proves TTT works specifically for EEG
- He et al. (2021) provides the exact autoencoder+adaptor architecture we need
- SSA (ICLR 2025) confirms regression TTA needs special treatment (not just entropy minimization)

The main open question is whether our model needs more capacity to benefit from adaptation. The current 7-tap FIR has ~2K parameters -- possibly too few for meaningful gradient-based adaptation. The reconstruction-TTA approach may naturally push us toward a slightly larger encoder that can capture subject-specific patterns worth adapting.

**Expected impact**: +0.01 to +0.03 r if the encoder has sufficient capacity and the reconstruction loss provides a useful adaptation signal. The cross-subject gap (0.27 to 0.46) suggests substantial room for improvement through better subject adaptation.
