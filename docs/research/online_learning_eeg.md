# Online & Continual Learning for EEG Subject Adaptation

## Context

This document covers **online/continual learning** approaches -- methods that adapt incrementally as new subject data streams in, rather than one-shot test-time adaptation (covered in `test_time_adaptation.md`). The key question for our project: can we train on subjects 1-11, do online adaptation on subject 12's data, then evaluate on subjects 13-15?

---

## 1. T-TIME: Test-Time Information Maximization Ensemble (Xu et al., IEEE TNNLS 2023; arXiv 2412.07228)

**The most directly relevant method for streaming EEG adaptation.**

**Problem:** EEG-based BCIs require subject-specific calibration. Existing transfer learning assumes all unlabeled test data is available upfront. T-TIME handles the hardest scenario: unlabeled EEG arrives one trial at a time, and classification must be immediate.

**Method:**
1. Initialize an ensemble of classifiers from Euclidean-Aligned source subject data
2. When each unlabeled test trial arrives:
   - Predict its label using ensemble voting
   - Update each classifier via:
     - **Conditional entropy minimization** (sharpen predictions)
     - **Adaptive marginal distribution regularization** (prevent class collapse)
3. The model continuously improves as more test data streams in

**Timing:** Pre-inference ~5ms, post-inference model update ~35-60ms (runs in background before next trial).

**Results:** Outperformed ~20 classical and SOTA transfer learning methods on 3 MI-BCI datasets. First work on truly online test-time adaptation for calibration-free BCIs.

**Relevance to our task:**
- Classification-specific (entropy minimization requires class probabilities)
- For regression, we'd need a surrogate objective (see Section 7)
- The ensemble + streaming update architecture is transferable

**Source:** [T-TIME (arXiv)](https://arxiv.org/abs/2412.07228) | [IEEE Xplore](https://ieeexplore.ieee.org/document/10210666/)

---

## 2. Continual Online EEG Decoding: Fine-Tuning Strategies (Sosulski & Tangermann, 2025; arXiv 2502.06828)

**Large-scale longitudinal study of online fine-tuning strategies for EEG.**

**Problem:** How should a deep learning decoder be updated as new sessions accumulate from the same subject over days/weeks?

**Strategies compared:**
1. **Naive fine-tuning**: Continue training on new session data only (catastrophic forgetting risk)
2. **Cumulative fine-tuning**: Retrain on all data from all sessions so far
3. **Sequential fine-tuning**: Fine-tune from the last checkpoint using new session data
4. **Joint sequential fine-tuning**: Fine-tune from last checkpoint, but include ALL prior subject-specific data (winner)
5. **Online test-time adaptation (OTTA)**: Update model during deployment without labels

**Key findings:**
- **Joint sequential fine-tuning** was most effective: builds on prior subject-specific information while incorporating all historical data
- OTTA enables calibration-free operation by adapting to evolving data distributions across sessions
- Naive fine-tuning caused catastrophic forgetting as expected
- Subject-specific history matters more than generic pre-training after a few sessions

**Relevance to our task:**
- Directly applicable protocol: train generic model on subjects 1-11, then do sequential fine-tuning on subject 12 (if we had labels), then deploy on 13-15
- Without labels for subject 12, OTTA is the applicable variant
- Suggests that even a small amount of labeled calibration data from the target domain is hugely valuable

**Source:** [arXiv](https://arxiv.org/abs/2502.06828) | [IEEE Xplore](https://ieeexplore.ieee.org/document/11253543/)

---

## 3. Online Continual Decoding with Balanced Memory Buffer (Li et al., Neural Networks 2024)

**Title:** "Online continual decoding of streaming EEG signal with a balanced and informative memory buffer"

**Problem:** In a streaming EEG setting, how do you decide which past data to keep in a memory buffer for experience replay?

**Method:**
- Maintain a fixed-size memory buffer of past EEG examples
- Balance buffer across classes/conditions to prevent bias
- Use informativeness criteria (e.g., prediction uncertainty, gradient magnitude) to select which examples to retain
- Replay buffer samples alongside new data during continual updates

**Key insight:** Not all past examples are equally valuable. Keeping high-uncertainty or high-gradient examples in the replay buffer is more effective than random sampling.

**Relevance to our task:**
- Experience replay is the most natural approach to prevent catastrophic forgetting
- For regression, "balanced" could mean balanced across subjects or across prediction difficulty
- Buffer size trade-off: too small = forgetting; too large = slow adaptation

**Source:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608024002624)

---

## 4. META-EEG: Meta-Learning for Zero-Calibration BCI (Han et al., Expert Systems with Applications 2024)

**Problem:** Inter-subject variability prevents direct transfer of MI-BCI models to new users.

**Method:**
- Gradient-based meta-learning (MAML-style) with intermittent freezing
- Train model to be easily adaptable: optimize for performance AFTER a few gradient steps on new subject data
- Intermittent freezing: periodically freeze layers to prevent overfitting during meta-training

**Protocol:**
```
Meta-training:
  For each episode:
    Sample support set (few examples from subject k)
    Inner loop: adapt model on support set (few gradient steps)
    Outer loop: evaluate on query set, update meta-parameters

Meta-testing (new subject):
    Take few labeled examples -> few gradient steps -> adapted model
```

**Key advantage:** The model is explicitly optimized to adapt quickly from minimal data. Zero-calibration means the meta-learned initialization already performs reasonably; few-shot calibration further improves it.

**Relevance to our task:**
- MAML-style meta-learning could work if we restructure training as episodes
- Each episode: sample a "target" subject from training set, adapt, evaluate
- At test time: adapt to new subject with few examples (if available)
- Without labels, combine with self-supervised inner loop

**Source:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417423024880) | [GitHub](https://github.com/MAILAB-korea/META-EEG)

---

## 5. FACE: Few-Shot Adapter with Cross-View Fusion (2025; arXiv 2503.18998)

**Problem:** Cross-subject EEG emotion recognition with very limited target-subject data.

**Method:**
- Pre-train a feature extractor on source subjects
- Design a lightweight adapter module that fuses multiple "views" of EEG (spectral, spatial, temporal)
- Fine-tune ONLY the adapter using K-shot examples from the target subject (K = 1, 5, 10)
- Cross-view fusion captures complementary information that aids fast adaptation

**Key insight:** Freezing the backbone and adapting only a small adapter prevents overfitting to the few target examples, similar to LoRA/adapter tuning in NLP.

**Relevance to our task:**
- Adapter-based fine-tuning is directly applicable to our FIR model
- We could freeze the FIR filters and add a lightweight subject-specific adapter layer
- Even 1-5 labeled windows from a new subject could shift the adapter

**Source:** [arXiv](https://arxiv.org/abs/2503.18998)

---

## 6. NeuroTTT: Test-Time Training for EEG Foundation Models (Wang et al., 2025; arXiv 2509.26301)

**Covered in detail in `test_time_adaptation.md`, but relevant online learning aspects:**

- Stage II performs test-time training on EACH unlabeled test sample via self-supervised loss
- Can be applied in a streaming fashion: adapt backbone as each test window arrives
- Self-supervised tasks (stopped-band prediction, amplitude scaling, temporal jigsaw) require only input data
- Entropy minimization via Tent (updating only normalization statistics) is sometimes more robust than full TTT

**Key finding for online use:** Tent-style adaptation (BN stats only) often outperforms full-model TTT on single samples because full-model updates on noisy individual EEG windows can destabilize the model.

**Source:** [arXiv](https://arxiv.org/abs/2509.26301)

---

## 7. Catastrophic Forgetting Mitigation Strategies

Relevant general continual learning methods applicable to EEG:

### Elastic Weight Consolidation (EWC)
- Compute Fisher Information Matrix after training on source subjects
- When adapting to new subject, penalize changes to parameters that were important for source subjects
- `L_total = L_new_subject + lambda * sum(F_i * (theta_i - theta_source_i)^2)`
- Prevents catastrophic forgetting of source knowledge while allowing adaptation

### Experience Replay
- Maintain a buffer of representative examples from training subjects
- During online adaptation, interleave buffer samples with new subject data
- Gradient from replay samples counteracts forgetting

### Progressive Neural Networks
- Add new columns/modules for each new subject
- Lateral connections allow knowledge transfer without overwriting
- No forgetting by design, but model grows with each subject

### Sparse Memory Fine-Tuning (2025)
- Leverage sparsely-updated memory layers
- Only a small subset of parameters change per new subject
- Reduces interference between subjects' learned representations

---

## 8. Proposed Protocol: Online Adaptation for Our Setup

### The Question
Can we train on subjects 1-11, do online adaptation on subject 12's data, then evaluate on 13-15?

### Analysis

**What we have at adaptation time (subject 12):**
- Input scalp EEG (46 channels): YES
- Output in-ear EEG (12 channels): YES (subject 12 is not in test set)
- This means we have LABELS for adaptation -- this is supervised online learning

**What we have at test time (subjects 13-15):**
- Input scalp EEG: YES
- Output in-ear EEG: NO (these are the test labels)

### Proposed Protocol

```
Phase 1: Pre-training (subjects 1-11)
    Train FIR model on all training subjects
    Store Fisher Information Matrix (for EWC)
    Save representative examples in replay buffer

Phase 2: Online adaptation (subject 12, supervised)
    For each batch of subject 12's data:
        1. Forward pass, compute regression loss (MSE + corr)
        2. Add EWC penalty to prevent forgetting subjects 1-11
        3. Optionally replay buffer samples from subjects 1-11
        4. Update model parameters (full or adapter-only)
        5. Evaluate on held-out portion of subject 12

Phase 3: Unsupervised test-time adaptation (subjects 13-15)
    For each test subject:
        Option A: Euclidean Alignment (subject-level covariance)
        Option B: BN statistics adaptation (Tent)
        Option C: Self-supervised TTT (channel masking task)
        Then: make final predictions with adapted model
```

### Why This Could Help

1. **Subject 12 as a bridge**: If subject 12 is "closer" in distribution to test subjects than subjects 1-11, adaptation on 12 could shift the model toward the test distribution
2. **More data is always better**: Adding subject 12 to training is straightforward and should help regardless
3. **Online protocol tests generalization**: If the model can rapidly adapt to subject 12 with few examples, it suggests the architecture supports fast personalization

### Why This Might NOT Help

1. **Subject 12 may not be representative** of subjects 13-15
2. **With only 15 subjects total**, the marginal value of one more training subject is limited
3. **EWC/replay overhead** for a simple FIR model may not be justified -- just retraining on subjects 1-12 might be equivalent
4. **The real bottleneck is test-time adaptation** (unsupervised), not adding one more training subject

### Simplified Alternative

The most practical approach may be simpler than full continual learning:

```
Step 1: Train on subjects 1-12 (just add subject 12 to training set)
Step 2: Apply subject-level Euclidean Alignment to test subjects
Step 3: Optionally do Tent-style BN adaptation on test subjects
```

This avoids the complexity of online learning while capturing most of the benefit.

---

## 9. CORAL / Deep CORAL for Online Alignment

**CORAL (Correlation Alignment)** aligns second-order statistics (covariance) between source and target domains:

```
Source covariance: C_s
Target covariance: C_t
CORAL loss: ||C_s - C_t||^2_F  (Frobenius norm)
```

**Deep CORAL** integrates this as a loss term in deep networks, aligning feature-space covariance between domains.

**Online variant:** As new subject data streams in, continuously update the target covariance estimate and re-align:

```python
# Online covariance update
C_t = (n * C_t + x_new @ x_new.T) / (n + 1)
n += 1
# Re-compute alignment transform
A = C_s^(1/2) @ C_t^(-1/2)
x_aligned = A @ x_new
```

**Relevance:** This is essentially a streaming version of Euclidean Alignment. As more test data arrives, the covariance estimate improves, and alignment becomes more accurate. This could be applied sample-by-sample as test data streams in.

**Source:** [Deep CORAL (arXiv)](https://arxiv.org/abs/1607.01719)

---

## 10. EEGPT and Foundation Model Adaptation (NeurIPS 2024)

**EEGPT** is a 10M-parameter pretrained transformer for universal EEG representation.

**Key features:**
- Mask-based dual self-supervised pretraining
- Hierarchical spatial-temporal processing
- Achieves SOTA on downstream tasks with linear probing

**Online adaptation pathway:**
1. Start with EEGPT pretrained backbone
2. Add a regression head for scalp-to-in-ear prediction
3. Fine-tune head on subjects 1-12
4. At test time, use TTT with EEGPT's built-in SSL objectives

**Limitation:** EEGPT expects specific channel configurations and may not directly support our 46-channel scalp + 12-channel in-ear setup without modification.

**Source:** [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) | [GitHub](https://github.com/BINE022/EEGPT)

---

## 11. Practical Recommendations for Our Project

### Immediate (Low complexity, high confidence)

1. **Add subject 12 to training set** -- simplest form of "adaptation," likely gives +0.001-0.005 r
2. **Subject-level Euclidean Alignment** on test subjects -- already recommended in `test_time_adaptation.md`, retry with proper implementation

### Medium-term (Moderate complexity)

3. **MAML-style meta-learning**: Restructure training as episodic leave-one-subject-out, optimize for fast adaptation. Could improve cross-subject generalization even without test-time adaptation.
4. **Adapter-based fine-tuning**: Add a small subject-specific adapter (linear layer) after the FIR backbone. Freeze FIR, train adapter on each subject. At test time, initialize adapter from nearest training subject.

### Longer-term (High complexity, potentially high reward)

5. **Self-supervised online TTT**: Train with auxiliary SSL task (channel masking). At test time, adapt backbone using SSL loss on unlabeled test data in a streaming fashion.
6. **Experience replay with EWC**: Full continual learning pipeline, but likely overkill for 15 subjects.

### What NOT to do

- **Don't use entropy minimization** for regression (no class probabilities)
- **Don't do full-model TTT on single windows** (too noisy, destabilizes the model)
- **Don't use batch-level Euclidean Alignment** (noisy covariance, already failed in iter018/042)
- **Don't expect online learning to break the 0.378 plateau** on 1-9 Hz data -- the bottleneck is likely the narrow frequency band, not the adaptation strategy

---

## Sources

- [T-TIME: Test-Time Information Maximization Ensemble](https://arxiv.org/abs/2412.07228) (IEEE TNNLS 2023)
- [Continual Online EEG Decoding Fine-Tuning Strategies](https://arxiv.org/abs/2502.06828) (2025)
- [Online Continual Decoding with Balanced Memory Buffer](https://www.sciencedirect.com/science/article/abs/pii/S0893608024002624) (Neural Networks 2024)
- [META-EEG: Meta-Learning for Zero-Calibration BCI](https://www.sciencedirect.com/science/article/pii/S0957417423024880) (Expert Sys. w/ Apps. 2024)
- [FACE: Few-Shot Adapter with Cross-View Fusion](https://arxiv.org/abs/2503.18998) (2025)
- [NeuroTTT: Test-Time Training for EEG Foundation Models](https://arxiv.org/abs/2509.26301) (2025)
- [EEGPT: Pretrained Transformer for Universal EEG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) (NeurIPS 2024)
- [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719) (ECCV 2016)
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) (PNAS 2017)
- [GOPSA: Geodesic Optimization for Predictive Shift Adaptation](https://arxiv.org/abs/2407.03878) (NeurIPS 2024)
- [Subject-Adaptive Transfer Learning Using Resting State EEG](https://arxiv.org/abs/2405.19346) (MICCAI 2024)
- [Harnessing Few-Shot Learning for EEG Classification (Survey)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1421922/full) (Frontiers 2024)
- [Inter- and Intra-Subject Variability in EEG (Survey)](https://arxiv.org/abs/2602.01019) (2026)
