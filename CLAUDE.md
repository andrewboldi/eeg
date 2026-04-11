# EEG Scalp-to-In-Ear Prediction — Autoresearch Project

## Work Philosophy: Delegator-First
**You are a PI (principal investigator), not a grad student.** Maximize productivity by delegating to subagents like PhD students:
- **Research agents**: Download papers, read TeX sources, extract architecture details, write summaries
- **Implementation agents**: Write model code, preprocessing scripts, benchmark runners
- **Analysis agents**: Analyze results, find patterns, debug failures, scan for code issues
- **Think before doing**: Spend time planning what to delegate, not doing grunt work yourself
- **Sequential GPU, parallel research**: Only one GPU task at a time (30GB RAM, 8GB VRAM), but spawn multiple research/analysis agents simultaneously
- **Every iteration**: delegate paper reading + code writing + analysis to separate agents

## Objective
**Maximize Pearson r** for predicting 12-channel in-ear EEG from 27-channel scalp EEG on real human recordings (Ear-SAAD dataset, 15 subjects).

## Fixed Benchmark (DO NOT MODIFY)
- **Test set**: Subjects 13, 14, 15 (LOSO — train on 1-12, evaluate on each held-out)
- **Metric**: Mean Pearson r across 3 test subjects × 12 in-ear channels
- **Data**: Ear-SAAD (Geirnaert et al. 2025), 1-9 Hz, 20 Hz sampling, 2s windows
- **Script**: `uv run python scripts/benchmark.py --baseline` (or `--model-fn models/your_model.py`)
- **Leaderboard**: `results/benchmark/leaderboard.jsonl`

### Current Best: r = 0.378 (Combined loss + corr validation, iter017/019)

## Autoresearch Loop Protocol

Each iteration follows this cycle:

### 1. RESEARCH — Search for ideas
- Search the web for recent papers on EEG prediction, time-series ML, spatial filtering
- Download arXiv TeX sources to `docs/external/` via `https://arxiv.org/src/<ID>`
  - Example: `wget -O docs/external/lora.tar.gz https://arxiv.org/src/2106.09685`
  - Extract and keep the .tex files for reference
- Read the downloaded papers for applicable techniques
- Check `results/benchmark/leaderboard.jsonl` for what's been tried

### 2. HYPOTHESIZE — Pick ONE idea
- State the hypothesis clearly (e.g., "Temporal attention over 10s context will capture slow EEG dynamics")
- Predict expected improvement with reasoning
- Identify risks and fallback plan

### 3. IMPLEMENT — Write the model
- Create `models/iter{NNN}_{name}.py` implementing `build_and_train()`
- Follow the template in `models/template.py`
- Keep changes minimal and focused on testing the hypothesis

### 4. EVALUATE — Run the benchmark
```bash
uv run python scripts/benchmark.py --model-fn models/iter{NNN}_{name}.py --name iter{NNN}_{name}
```
- Results auto-append to `results/benchmark/leaderboard.jsonl`
- Compare against current best r

### 5. ANALYZE — Write the report
- Create `docs/src/report_{NNN}.tex` documenting:
  - Hypothesis, method, results, comparison to baseline
  - What worked, what didn't, why
  - Concrete next ideas motivated by findings
- Build: `cd docs && make`

### 6. UPDATE — Record progress
- Update the leaderboard table below
- Update "Current Best" if improved
- Add new ideas to the research queue
- Commit and push everything

### 7. REPEAT

## Research Queue (prioritized by expected impact)
1. **Broadband prediction** (download raw BIDS data, 1-45 Hz at 256 Hz) — highest potential impact
2. **[GPU REQUIRED] Transformer architectures** (iter022/023) — need GPU for training
3. Multi-task learning (predict scalp+around-ear+in-ear jointly)
4. Subject-specific fine-tuning (few-shot adaptation with target-subject data)
5. Contrastive pre-training on scalp EEG, then fine-tune for in-ear
6. GEVD-based spatial pre-filtering (maximize SNR before temporal model)
7. Around-ear channel prediction (19 cEEGrid channels as targets)
8. Domain adaptation techniques for cross-subject transfer

## Key Findings from Iterations 013-036
- **Cross-subject variability is the bottleneck**, not model capacity
- Subject 14 consistently ~0.27 r; Subject 13 consistently ~0.46 r
- Longer FIR filters (11, 15 taps) don't help — 7 taps is sufficient for 1-9 Hz
- Combined MSE+corr loss + corr-based early stopping gives marginal improvement
- Band-specific splitting and Euclidean alignment hurt performance
- Residual learning and ensembles don't improve over direct FIR
- **InstanceNorm consistently hurts** (~-0.003 r) — removes useful amplitude dynamics
- MoE, noise augmentation, Huber loss all fail when combined with InstanceNorm
- Mixup helps marginally (+0.0003) when used WITHOUT InstanceNorm
- **Causal-only multi-lag models perform worse** than acausal FIR — acausal is critical
- PLS and OLS give nearly identical spatial filters for this data
- Pure correlation loss produces degenerate scale (terrible SNR)
- The 0.378 plateau appears to be a fundamental limit of 7-tap FIR + SGD on 1-9 Hz data
- **Ear-SAAD paper** uses Ledoit-Wolf shrinkage + 400ms temporal filter — testing in iter036

### Current Best: r = 0.378 (iter030: Mixup + FIR + combined loss + corr val)

## Leaderboard
| Iter | Model | Mean r | Std r | SNR (dB) | Key Idea |
|------|-------|--------|-------|----------|----------|
| 007 | closed_form_baseline | 0.366 | 0.072 | 0.59 | Linear spatial filter W*=R_YX @ inv(R_XX) |
| 008 | regularization_sweep | 0.366 | 0.074 | 0.59 | Tikhonov reg sweep (no improvement) |
| 009 | fir_spatio_temporal | **0.373** | 0.074 | 0.61 | FIR filter with CF center-tap init, 150 epochs |
| 010 | deep_temporal_conv | 0.372 | 0.076 | 0.62 | Depthwise-sep conv + residual (no improvement) |
| 011 | fir_channel_dropout | **0.376** | 0.076 | 0.61 | FIR + 15% channel dropout augmentation |
| 012 | cf_fir_ensemble | 0.375 | 0.076 | 0.61 | Weighted avg of CF + FIR (no improvement) |
| 013 | band_specific | 0.343 | 0.063 | 0.51 | Per-band CF spatial filters (worse) |
| 014 | long_fir_dropout | 0.375 | 0.076 | 0.61 | Longer FIR + dropout + warm restarts (no improvement) |
| 015 | correlation_loss | 0.373 | 0.076 | 0.61 | Corr loss + MSE validation (mismatch) |
| 016 | residual_fir | 0.375 | 0.077 | 0.61 | CF + learned FIR residual (no improvement) |
| 017 | corr_val | **0.378** | 0.078 | 0.61 | Combined MSE+corr loss + corr validation |
| 018 | euclidean_align | 0.369 | 0.072 | 0.60 | Per-batch whitening (worse — noisy batch cov) |
| 019 | swa_combined | **0.378** | 0.078 | 0.61 | SWA + combined loss (matches iter017) |
| 020 | instance_norm_fir | 0.375 | 0.077 | 0.63 | InstanceNorm hurts r despite better SNR |
| 021 | mixture_of_experts | 0.375 | 0.077 | 0.63 | 4 experts + InstanceNorm (no improvement) |
| 024 | mixup_inorm | 0.375 | 0.076 | 0.61 | Mixup + InstanceNorm (INorm masks benefit) |
| 025 | cosine_loss_schedule | 0.376 | 0.077 | 0.63 | MSE→corr annealing + INorm (no improvement) |
| 026 | noise_augment | 0.375 | 0.077 | 0.62 | Gaussian noise + INorm (no improvement) |
| 027 | pls_regression | 0.377 | 0.077 | 0.61 | PLS init ≈ CF init |
| 028 | huber_corr | 0.375 | 0.075 | 0.62 | Huber loss + INorm (lowest std but no improvement) |
| 030 | mixup_no_inorm | **0.378** | 0.077 | 0.61 | Mixup without INorm — new best |
| 031 | pure_corr | 0.375 | 0.077 | -5.96 | Pure corr loss — degenerate scale |
| 032 | high_dropout | 0.378 | 0.079 | 0.60 | 25% dropout (no improvement over 15%) |
| 034 | cca_ridge | 0.365 | 0.072 | 0.59 | Causal-only lags hurt performance |
| 035 | ledoit_wolf | 0.369 | 0.075 | 0.59 | LW shrinkage + causal lags (still worse) |

## Reading Papers
When referencing arXiv papers:
1. Download TeX source: `wget -O docs/external/{name}.tar.gz https://arxiv.org/src/{ID}`
2. Extract: `cd docs/external && mkdir {name} && tar xzf {name}.tar.gz -C {name}/`
3. Read the .tex files for key methods, architectures, loss functions
4. Cite in your report and note which ideas you adapted

## File Structure
```
models/                         # Model submissions (one per iteration)
  template.py                   # Copy this to create a new model
  iter008_*.py, iter009_*.py    # Each iteration's model
scripts/
  benchmark.py                  # FIXED benchmark (DO NOT MODIFY)
  real_data_experiment.py       # Data loading utilities
docs/
  src/report_NNN.tex            # LaTeX reports per iteration
  external/                     # Downloaded paper TeX sources
results/
  benchmark/                    # Benchmark results and leaderboard
  real_data/                    # Raw experiment outputs
```

## Running
```bash
uv sync                                                    # Install deps
uv run python scripts/benchmark.py --baseline              # Baseline
uv run python scripts/benchmark.py --model-fn models/X.py  # Test model
cd docs && make                                            # Build reports
```

## Key Constraints
- Data is 1-9 Hz at 20 Hz (very narrowband) — this limits what temporal models can learn
- 12 in-ear channels, some noisier than others (ELC, ERT often have artifacts)
- 15 subjects total, 3 held out for test — small dataset
- NaN values interpolated linearly; some channels are 100% NaN for some subjects
- Conv encoder barely beats closed-form (+0.005 in r) — the mapping is roughly linear in this band

## Important: Always Commit and Push
After every iteration, commit and push all changes (code, reports, CLAUDE.md updates, benchmark results).

## Iteration History (Synthetic Phase)
| Iter | Key Change | Status |
|------|-----------|--------|
| 001 | Baseline (combined loss) | All gradient models fail |
| 002 | MSE-only, grad clip, CF init | Linear matches CF |
| 003 | 20 subjects, 500 conv epochs | Conv converges |
| 004 | FIR center-tap CF init | ALL CONVERGED (r=0.887 synthetic) |
| 005 | Cross-subject LOSO | CF LOSO r=0.861+/-0.012 |
| 006 | Electrode subset selection | 5ch maintain r>=0.80 |
| 007 | Real data (Ear-SAAD) | CF r=0.343, Conv r=0.348 |
