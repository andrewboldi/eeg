# EEG Data Augmentation for Cross-Subject Generalization

Literature review of techniques from 2023-2025, focused on applicability to our scalp-to-in-ear prediction task (Ear-SAAD, 12 subjects train, 3 test, 27ch scalp -> 12ch in-ear).

## 1. Channel-Level Recombination (BAR)

**Source:** Freer et al. (2021), "Data Augmentation: Using Channel-Level Recombination to Improve Classification Performance for Motor Imagery EEG" ([PMC7990774](https://pmc.ncbi.nlm.nih.gov/articles/PMC7990774/))

**Method:** Brain-Area-Recombination (BAR) splits each EEG sample into left/right hemisphere channel groups and recombines half-samples from different trials to create synthetic training examples. For N trials, this generates up to N^2 recombined samples.

**Results:** EEGNet achieved up to 8.3% accuracy improvement over CSP-SVM baseline. BAR significantly outperformed noise-addition and flipping augmentations.

**Applicability to our task:** MEDIUM. We could split 27 scalp channels into anatomical groups (frontal/central/parietal/temporal) and recombine across subjects. This would create synthetic "chimera subjects" with mixed spatial topographies. Risk: recombined channel groups may have inconsistent spatial relationships, harming spatial filter learning.

**Proposed adaptation:**
```python
# Split channels into regions, swap regions between subjects
regions = {
    'frontal': ['Fp1','Fp2','F3','F4','Fz','F7','F8'],
    'central': ['C3','C4','Cz','T7','T8'],
    'parietal': ['P3','P4','Pz','P7','P8'],
    'occipital': ['O1','O2','Oz']
}
# For each training batch, randomly swap one region's channels between two subjects
```

## 2. EEG-Specific Mixup (EEG-Mixup)

**Source:** Zhou et al. (2025), referenced in Frontiers survey on EEG augmentation ([fnins.2026.1789468](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1789468/full))

**Method:** Unlike standard mixup (which blends arbitrary samples), EEG-mixup strictly blends segments from the same subject and trial to preserve non-stationary EEG characteristics. This maintains valid within-subject data distributions.

**Results:** Outperformed standard mixup by 3.57% in classification accuracy.

**Applicability to our task:** HIGH. We already have mixup in iter030 (current best, r=0.378). The key insight is that *cross-subject* mixup may be harmful because it blends incompatible spatial topographies. We should test:
- Within-subject-only mixup (blend windows from same subject)
- Cross-subject mixup with Euclidean alignment first (align distributions, then mix)

## 3. Euclidean Alignment + Data Augmentation

**Source:** Rodrigues et al. (2024), "Combining Euclidean Alignment and Data Augmentation for BCI decoding" ([arXiv:2405.14994](https://arxiv.org/abs/2405.14994)); Autthasan et al. (2024), "A Systematic Evaluation of Euclidean Alignment with Deep Learning for EEG Decoding" ([arXiv:2401.10746](https://arxiv.org/abs/2401.10746))

**Method:** Euclidean Alignment (EA) transforms each subject's data so that the average covariance matrix becomes the identity matrix:
```
X_aligned = R^{-1/2} @ X
where R = mean(X @ X^T) over trials for that subject
```
This makes cross-subject distributions more similar. When combined with data augmentation, EA + DA generates synergies.

**Results:** EA + fine-tuning improves accuracy by 8.41%. EA alone improves target-subject decoding by 4.33% and reduces convergence time by >70%.

**Applicability to our task:** HIGH. We tried EA in iter018 (r=0.369, worse) but used per-batch covariance which was too noisy. The correct approach is:
1. Compute per-subject covariance over ALL windows (not per-batch)
2. Apply EA before training
3. Then apply augmentation (mixup, noise) on the aligned data
This addresses our core finding that cross-subject variability is the bottleneck.

**Critical implementation note:** iter018 failed because of per-batch covariance estimation. Need full-subject covariance matrices computed offline during preprocessing.

## 4. Generative Approaches (GANs and VAEs)

### 4a. EEG-GAN Toolkit (2025)

**Source:** AutoResearch Group, "EEG-GAN: A Generative EEG Augmentation Toolkit" ([bioRxiv 2025.06.23.661164](https://www.biorxiv.org/content/10.1101/2025.06.23.661164v1.full))

**Method:** Open-source GAN framework that generates trial-level synthetic EEG. Trained on real EEG, then synthetic samples augment classifier training data.

**Results:** Up to 16% accuracy improvement. Outperformed 6 benchmark augmentation techniques in 69% of comparisons. Strongest effects on datasets with <=30 participants (our case: 12 train subjects).

**Applicability to our task:** MEDIUM-LOW. Our task is regression (predict in-ear from scalp), not classification. GANs would need to generate paired (scalp, in-ear) samples, which is harder. Also, our 1-9 Hz narrowband data has limited temporal complexity -- a GAN may not add much beyond what simpler augmentations provide.

### 4b. Trans-cVAE-GAN (2024)

**Source:** "Trans-cVAE-GAN: Transformer-Based cVAE-GAN for High-Fidelity EEG Signal Generation" ([MDPI](https://www.mdpi.com/2306-5354/12/10/1028))

**Method:** Combines Transformer temporal modeling, conditional VAE for latent space learning, and adversarial training. Multi-dimensional loss preserves temporal correlation, frequency-domain consistency, and statistical distribution.

### 4c. Dual-Encoder VAE-GAN

**Source:** "Dual-Encoder VAE-GAN With Spatiotemporal Features for Emotional EEG Data Augmentation" ([PubMed:37053054](https://pubmed.ncbi.nlm.nih.gov/37053054/))

**Results:** 5% improvement with augmented data (97.21% accuracy).

## 5. Subject-Interpolation Augmentation (Novel Idea)

**No direct prior work found** for interpolating between subjects' spatial filters to create synthetic subjects. This is a novel idea worth exploring.

**Proposed method:** Given trained per-subject spatial filters W_i (from closed-form solution W* = R_YX @ inv(R_XX)):
```python
# For subjects i, j, generate synthetic subject k
alpha = np.random.beta(0.5, 0.5)  # Concentrate near 0 and 1
W_synthetic = alpha * W_i + (1 - alpha) * W_j

# Generate synthetic in-ear predictions
Y_synthetic = W_synthetic @ X_real  # Apply to real scalp data from either subject

# Or interpolate in covariance space:
R_XX_synth = alpha * R_XX_i + (1 - alpha) * R_XX_j
R_YX_synth = alpha * R_YX_i + (1 - alpha) * R_YX_j
```

**Rationale:** The cross-covariance matrices R_YX encode the spatial relationship between scalp and in-ear channels. Interpolating between subjects creates plausible "virtual subjects" whose spatial characteristics lie between real subjects. This is analogous to mixup but operates in the model parameter space rather than the data space.

**Risks:**
- Linear interpolation of covariance matrices may leave the positive-definite cone
- Use geodesic interpolation on SPD manifold instead: `R_synth = R_i^{1/2} @ (R_i^{-1/2} @ R_j @ R_i^{-1/2})^alpha @ R_i^{1/2}`
- The interpolated filter may not correspond to any real neuroanatomy

**Related work:** Riemannian transfer learning (Mellot et al., 2024, [fnins.2024.1381572](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1381572/full)) uses log-Euclidean metrics on SPD manifold for EEG transfer, providing theoretical foundation for interpolation on covariance manifolds.

## 6. Other Relevant Techniques

### 6a. Background EEG Mixing (BGMix)
Swaps background noise between EEG trials of different classes while preserving task-critical features. Neurophysiologically motivated.

### 6b. Siamese Network Augmentation
Data augmentation for cross-subject EEG features using Siamese neural networks ([ResearchGate](https://www.researchgate.net/publication/359044258)). Learns subject-invariant representations through contrastive pairs.

### 6c. META-EEG (2024)
Gradient-based meta-learning with intermittent freezing for cross-subject motor imagery BCI. Handles inter-subject variability through optimization-based adaptation.

### 6d. Robust Learning with Dynamic Spatial Filtering
"Robust learning from corrupted EEG with dynamic spatial filtering" ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1053811922001239)). Learns to dynamically adapt spatial filters to handle corrupted/variable channels.

## Priority Ranking for Our Task

| Priority | Technique | Expected Impact | Effort | Risk |
|----------|-----------|----------------|--------|------|
| 1 | EA (full-subject cov) + Mixup | HIGH | LOW | Low -- fixes iter018's bug |
| 2 | Subject-interpolation in cov space | HIGH | MEDIUM | Medium -- novel, untested |
| 3 | Within-subject-only mixup | MEDIUM | LOW | Low -- simple constraint |
| 4 | Channel region recombination (BAR) | MEDIUM | MEDIUM | Medium -- may break spatial structure |
| 5 | EEG-GAN for paired generation | LOW | HIGH | High -- complex, regression task |

## Recommended Next Iterations

### iter038: Euclidean Alignment (Correct Implementation)
Fix iter018 by computing per-subject covariance over all windows offline, then apply EA before training. Combine with our best setup (FIR + combined loss + corr validation + mixup).

### iter039: Subject Covariance Interpolation
Generate synthetic training subjects by interpolating cross-covariance matrices between subject pairs. Use Beta(0.5, 0.5) sampling to create diverse synthetic filters. Train FIR model on real + synthetic subject data.

### iter040: Within-Subject Mixup
Constrain mixup to only blend windows from the same subject (preserve spatial consistency). Compare against cross-subject mixup (current iter030).

## Sources

- [BAR Channel Recombination (Freer et al. 2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7990774/)
- [EEG Augmentation Survey (Frontiers 2026)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1789468/full)
- [GANs in EEG Analysis Overview (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10088201/)
- [EEG Data Augmentation Strategies (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9433610/)
- [EA + Data Augmentation for BCI (2024)](https://arxiv.org/abs/2405.14994)
- [Systematic Evaluation of EA (2024)](https://arxiv.org/abs/2401.10746)
- [EEG-GAN Toolkit (2025)](https://www.biorxiv.org/content/10.1101/2025.06.23.661164v1.full)
- [Trans-cVAE-GAN (2024)](https://www.mdpi.com/2306-5354/12/10/1028)
- [Riemannian Transfer Learning (2024)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1381572/full)
- [Cross-Subject Contrastive Learning (2025)](https://www.nature.com/articles/s41598-025-13289-5)
- [Hybrid Transfer Learning (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10687359/)
- [EEG-Mixup (Zhou et al. 2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1789468/full)
