# EEG Scalp-to-In-Ear Prediction — Autoresearch Project

## Objective
**Maximize Pearson r** for predicting 12-channel in-ear EEG from 27-channel scalp EEG on real human recordings (Ear-SAAD dataset, 15 subjects).

## Fixed Benchmark (DO NOT MODIFY)
- **Test set**: Subjects 13, 14, 15 (LOSO — train on 1-12, evaluate on each held-out)
- **Metric**: Mean Pearson r across 3 test subjects × 12 in-ear channels
- **Data**: Ear-SAAD (Geirnaert et al. 2025), 1-9 Hz, 20 Hz sampling, 2s windows
- **Script**: `uv run python scripts/benchmark.py --baseline` (or `--model-fn models/your_model.py`)
- **Leaderboard**: `results/benchmark/leaderboard.jsonl`

### Current Best: r = 0.373 (FIR spatio-temporal filter, iter009)

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
1. Subject-specific fine-tuning (few-shot adaptation with target-subject data)
2. Temporal context models (FIR/RNN/attention over longer windows)
3. Broadband prediction (download raw BIDS data, 1-45 Hz at 256 Hz)
4. Multi-task learning (predict scalp+around-ear+in-ear jointly)
5. Data augmentation (time-shift, noise injection, channel dropout)
6. Contrastive pre-training on scalp EEG, then fine-tune for in-ear
7. Frequency-band-specific models (separate delta/theta/alpha predictions)
8. Graph neural networks (model electrode spatial relationships)
9. Around-ear channel prediction (19 cEEGrid channels as targets)
10. Domain adaptation techniques for cross-subject transfer

## Leaderboard
| Iter | Model | Mean r | Std r | SNR (dB) | Key Idea |
|------|-------|--------|-------|----------|----------|
| 007 | closed_form_baseline | 0.366 | 0.072 | 0.59 | Linear spatial filter W*=R_YX @ inv(R_XX) |
| 008 | regularization_sweep | 0.366 | 0.074 | 0.59 | Tikhonov reg sweep (no improvement) |
| 009 | fir_spatio_temporal | **0.373** | 0.074 | 0.61 | FIR filter with CF center-tap init, 150 epochs |
| 010 | deep_temporal_conv | 0.372 | 0.076 | 0.62 | Depthwise-sep conv + residual, 100 epochs (no improvement) |

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
