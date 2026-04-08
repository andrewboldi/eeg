# EEG Scalp-to-In-Ear Prediction Project

## Project Goal
Predict 4-channel in-ear EEG from 21-channel scalp EEG (10-20 system), then extend to broader EEG research goals.

## Research Loop Protocol

This project follows an automated iterative research loop:

### Loop Steps
1. **Run**: Execute the full pipeline (`bash scripts/run_all.sh`) to train all models and run ablations
2. **Analyze**: Extract metrics from `results/` and compare against previous iterations
3. **Write**: Generate a LaTeX report in `docs/src/report_NNN.tex` documenting findings
4. **Build**: Run `make` in `docs/` to produce PDFs
5. **Diagnose**: Identify root causes of poor performance or regressions
6. **Improve**: Implement concrete fixes based on diagnosis (code changes to `src/`, `configs/`, `scripts/`)
7. **Check convergence**: If all models achieve target metrics, move to Extension phase
8. **Repeat**: Go to step 1

### Convergence Criteria
- All gradient-trained models achieve Pearson r >= 0.85
- SNR >= 5.0 dB for all models
- No negative correlations in any model
- Band power correlations positive across all 5 bands

### Extension Phase (post-convergence)
Only after convergence criteria are met:
1. Brainstorm new research directions extending the scalp-to-in-ear prediction goal
2. Implement ONE new idea per iteration
3. Document results in a new report
4. Each extension should build on solid, verified results

### Current State
- **Iteration**: 004
- **Status**: CONVERGED. All 4 models achieve r >= 0.85 and SNR >= 5.0 dB
- **Phase**: Extension Phase - ready for new research directions
- **Next extension ideas** (prioritized):
  1. ~~Cross-subject generalization~~ DONE (Iter 005): LOSO r=0.861+/-0.012
  2. ~~Electrode subset selection~~ DONE (Iter 006): 5 channels (Fp1,F7,F8,T7,T8) maintain r>=0.80
  3. ~~Real EEG data~~ DONE (Iter 007): Ear-SAAD dataset, CF r=0.343, LOSO r=0.326+/-0.092
  4. Broadband prediction (raw BIDS data at full bandwidth 1-45 Hz)
  5. Subject-specific fine-tuning (adapt pooled model with target-subject data)
  6. Around-ear channel prediction (include 19 cEEGrid channels)
  7. Auditory attention decoding (downstream BCI application)
  8. Online/streaming causal prediction

### Iteration History
| Iter | Key Change | FIR r | Conv r | Status |
|------|-----------|-------|--------|--------|
| 001 | Baseline (combined loss) | -0.21 | -0.04 | All gradient models fail |
| 002 | MSE-only, grad clip, CF init for linear | 0.695 | 0.815 | Linear matches CF |
| 003 | 20 subjects, 500 conv epochs | 0.695 | 0.876 | Conv converges |
| 004 | FIR center-tap CF init | 0.887 | 0.886 | ALL CONVERGED |
| 005 | Cross-subject LOSO eval | --- | --- | CF LOSO r=0.861+/-0.012 |
| 006 | Electrode subset selection | --- | --- | 5ch (Fp1,F7,F8,T7,T8) r=0.818 |
| 007 | Real EEG data (Ear-SAAD) | --- | 0.348 | CF r=0.343, LOSO r=0.326+/-0.092 |

## File Structure
- `docs/src/report_NNN.tex` - LaTeX source for iteration NNN
- `docs/report_NNN.pdf` - Built PDF
- `docs/Makefile` - Build system for docs
- `results/` - Pipeline outputs (metrics, checkpoints, logs, ablation)
- `configs/` - YAML configs for each model
- `src/` - Source code (models, data, losses, metrics, training)

## Running the Pipeline
```bash
uv sync                       # Install deps
bash scripts/run_all.sh       # Full pipeline
cd docs && make               # Build reports
```

## Key Findings (Converged at Iteration 004)
- All 4 architectures achieve r~0.887, SNR~6.5 dB on synthetic data
- Closed-form initialization is critical for FIR and linear models
- MSE-only loss outperforms all combined loss variants
- Gradient clipping (norm=1.0) essential for training stability
- More data (20 subjects) + more epochs (500) needed for Conv Encoder
- Architecture doesn't matter much for linear forward models; initialization does

## Extension Loop Protocol
After convergence, each extension iteration:
1. Pick ONE idea from the extension list above
2. Implement the minimum code changes needed
3. Run the pipeline and measure impact
4. Write a report documenting results
5. Update this file with findings
6. If the extension opens new questions, add them to the list
7. **Commit and push all changes** (code, configs, reports, CLAUDE.md updates)

## Important: Always Commit and Push
After every iteration or significant code change, commit and push to the remote repository. Do not let work accumulate uncommitted. Use descriptive commit messages summarizing what changed and why.
