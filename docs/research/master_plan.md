# Master Research Plan — Active Projects

## Current Best: r=0.638 (iter039 tiny deep, 46ch broadband)
## Target: r=0.80+

---

## GPU Queue (sequential, one at a time)
| Priority | Model | Status | Expected gain |
|----------|-------|--------|---------------|
| 1 | iter043 ensemble (CF+deep) | TRAINING fold 2/3 | +0.01-0.04 |
| 2 | iter044 residual CF | READY to run | +0.01-0.02 |
| 3 | iter045 long context (4s) | READY, needs 4s benchmark | +0.02-0.05 |
| 4 | iter046 calibrated output | AGENT WRITING | +0.02-0.05 |
| 5 | iter047 spatial PE (REVE-inspired) | AGENT WRITING | +0.03-0.08 |
| 6 | iter048 NeuroTTT SSL adapt | AGENT WRITING | +0.02-0.06 |
| 7 | iter049 adversarial subject-invariant | TODO | +0.02-0.04 |
| 8 | iter050 L1 loss (REVE-style) | TODO | +0.01-0.02 |
| 9 | iter051 per-channel output heads | TODO | +0.01-0.02 |
| 10 | iter052 cross-attention decoder | TODO | +0.01-0.03 |
| 11 | Optuna HPO sweep (50 trials) | AGENT WRITING | +0.02-0.05 |

## Research Agents (parallel, no GPU needed)
| # | Project | Status | Output file |
|---|---------|--------|-------------|
| 1 | REVE paper analysis | DONE | docs/research/reve_architecture.md |
| 2 | LUNA/EEG-X analysis | DONE | docs/research/luna_eegx_architecture.md |
| 3 | Test-time adaptation | DONE | docs/research/test_time_adaptation.md |
| 4 | Subject difficulty analysis | RUNNING | docs/research/subject_analysis.md |
| 5 | Large EEG datasets | DONE | docs/research/large_eeg_data.md |
| 6 | SOTA techniques survey | DONE | docs/research/sota_techniques.md |
| 7 | Download HBN-EEG (AWS S3) | TODO | data/raw/hbn_eeg/ |
| 8 | Download MOABB datasets | TODO | data/raw/moabb/ |
| 9 | Ear-SAAD original paper | TODO | docs/research/ear_saad_paper.md |
| 10 | Data augmentation survey | TODO | docs/research/augmentation_survey.md |
| 11 | Write results paper draft | TODO | docs/src/paper_draft.tex |
| 12 | Visualization of predictions | TODO | results/figures/ |
| 13 | KAN for spatial mixing | TODO | docs/research/kan_spatial.md |
| 14 | Frequency-domain approaches | TODO | docs/research/freq_domain.md |

## Key Strategic Insights
1. **Scaling is dead** — 55K params ≈ 7M params on test r. Cross-subject gap is the bottleneck.
2. **Normalization always hurts** — RevIN, EA, InstanceNorm all destroy useful subject-specific amplitude info.
3. **Around-ear channels are gold** — +0.15 r just from adding 19 cEEGrid channels as input.
4. **Subject variance is huge** — r ranges from 0.38 (Subject 8) to 0.94 (Subject 3) on CF baseline.
5. **REVE's 4D positional encoding** could teach the model spatial relationships between electrodes.
6. **NeuroTTT's SSL adaptation** is the most promising path for closing the cross-subject gap.
7. **Foundation model pretraining** (REVE/LUNA) needs large data — HBN-EEG is freely downloadable.

## Theoretical Ceiling Analysis
- Subject 3 gets r=0.94 with just CF → the signal IS there for some subjects
- The variance across subjects (std=0.14) suggests ~0.16 r is "on the table" via better adaptation
- If we brought worst subjects to median: mean r would go from 0.645 to ~0.75
- Foundation model pretraining could add another +0.05-0.10
- **Realistic target: r=0.75-0.85 within 20 iterations**

## Resource Constraints
- **GPU**: RTX 4060, 8GB VRAM — one training job at a time
- **RAM**: 30GB — don't run parallel preprocessing + training
- **Storage**: ~10GB free for additional datasets
- **Compute time**: ~10 min per 3-fold benchmark, ~4 hours for 50-trial Optuna
