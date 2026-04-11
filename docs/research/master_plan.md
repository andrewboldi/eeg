# Master Research Plan — Active Projects

## Current Best: r=0.638 (iter039 tiny deep, 46ch broadband)
## Target: r=0.80+

---

## Architecture Sweep Results (iter038-054, 46ch broadband)

15 models tested. **None beat the tiny baseline (iter039).**

| Model | mean_r | std_r | SNR (dB) | Key Idea |
|-------|--------|-------|----------|----------|
| closed_form_46ch | 0.577 | 0.119 | 1.81 | Linear CF baseline, 46ch |
| **iter039_tiny_46ch** | **0.638** | **0.111** | — | **Tiny deep net (BEST)** |
| iter041_subject_adapt | 0.587 | 0.119 | 1.86 | Subject-specific adaptation |
| iter042_ea_46ch | 0.574 | 0.096 | — | Euclidean alignment (worse) |
| iter043_ensemble_46ch | 0.606 | 0.123 | 2.10 | CF+deep ensemble |
| iter044_residual_cf_46ch | 0.610 | 0.120 | 2.12 | Residual on top of CF |
| iter045_long_context_4s | 0.606 | 0.120 | 2.10 | 4s context window |
| iter046_calibrated_46ch | 0.581 | 0.124 | 1.89 | Calibrated output scaling |
| iter047_spatial_pe_46ch | 0.589 | 0.117 | 1.95 | REVE-inspired spatial PE |
| iter048_neurottt_46ch | 0.597 | 0.117 | 2.01 | NeuroTTT SSL adaptation |
| iter049_adversarial_46ch | 0.608 | 0.117 | 2.12 | Adversarial subject-invariant |
| iter050_l1_loss_46ch | 0.605 | 0.122 | 2.09 | L1 loss (REVE-style) |
| iter051_perchannel_heads_46ch | 0.609 | 0.122 | 2.14 | Per-channel output heads |
| iter052_cross_attn_decoder_46ch | 0.596 | 0.116 | 2.00 | Cross-attention decoder |
| iter053_spectral_loss_46ch | 0.604 | 0.121 | 2.04 | Spectral domain loss |
| iter054_pretrained_finetune | 0.589 | 0.117 | 1.94 | Pretrained + finetune |

## Current Status
**Unified pretraining running** with 40 HBN + 15 Ear-SAAD subjects (55 total).

## Data Inventory
| Dataset | Subjects | Channels | Status |
|---------|----------|----------|--------|
| Ear-SAAD | 15 | 27 scalp + 19 cEEGrid + 12 in-ear | Downloaded, preprocessed |
| HBN-EEG | 138 | 128 (10-20 subset available) | Downloaded from S3 |
| Mobile BCI | 18 | 64 | Downloaded |
| EESM | 22 | 30 | Downloaded |
| MOABB | 50 | varies | Downloaded |
| **Total** | **243** | — | — |

## Research Documents
| File | Topic |
|------|-------|
| ceegrid_access.md | cEEGrid hardware access notes |
| ceegrid_datasets.md | cEEGrid dataset survey |
| channel_attention_eeg.md | Channel attention mechanisms |
| contrastive_eeg.md | Contrastive learning for EEG |
| da_regression.md | Domain adaptation for regression |
| disgcmae.md | DisGCMAE architecture analysis |
| ear_saad_paper.md | Ear-SAAD original paper analysis |
| eeg_augmentation.md | EEG augmentation survey |
| eeg_denoising.md | EEG denoising techniques |
| eesm_status.md | EESM dataset download status |
| fewshot_calibration.md | Few-shot calibration methods |
| flash_attention_cuda.md | Flash attention CUDA notes |
| freq_domain.md | Frequency-domain approaches |
| hbn_eeg_status.md | HBN-EEG download status |
| improvement_ideas.md | Brainstormed improvement ideas |
| kan_spatial.md | KAN for spatial mixing |
| large_eeg_data.md | Large EEG dataset survey |
| luna_eegx_architecture.md | LUNA/EEG-X architecture analysis |
| moabb_download_status.md | MOABB download status |
| mobile_bci_analysis.md | Mobile BCI dataset analysis |
| mobile_bci_status.md | Mobile BCI download status |
| multitask_pretrain.md | Multi-task pretraining plan |
| online_learning_eeg.md | Online learning for EEG |
| optimal_transport_eeg.md | Optimal transport domain adaptation |
| pretraining_plan.md | Foundation model pretraining plan |
| progress_summary.md | Overall progress summary |
| reve_architecture.md | REVE paper analysis |
| riemannian_eeg.md | Riemannian geometry for EEG |
| sota_architectures.md | SOTA architecture survey |
| sota_techniques.md | SOTA techniques survey |
| subject_analysis.md | Subject difficulty analysis |
| test_time_adaptation.md | Test-time adaptation methods |
| transfer_regression.md | Transfer learning for regression |
| tuh_access.md | TUH EEG corpus access notes |

## Key Findings from Architecture Sweep (15 models)

1. **Tiny model is king** — iter039 (r=0.638) beat every larger/fancier architecture tested.
2. **Scaling is dead** — 55K params performs the same as 7M params. Cross-subject gap is the bottleneck.
3. **Normalization always hurts** — RevIN, EA, InstanceNorm all destroy useful subject-specific amplitude info. EA (iter042) was the worst at 0.574.
4. **Around-ear channels are gold** — +0.15 r just from adding 19 cEEGrid channels as input (27ch -> 46ch).
5. **Subject variance is huge** — r ranges from 0.38 (Subject 8) to 0.94 (Subject 3) on CF baseline.
6. **Adversarial training is runner-up** — iter049 (0.608) came closest after tiny, but still 0.03 behind.
7. **Residual CF and per-channel heads tied for 3rd** — both at ~0.610, showing CF is a strong prior.
8. **Spatial PE, cross-attention, NeuroTTT all underperformed** — complex architectures add noise, not signal.
9. **Long context (4s) didn't help** — temporal information beyond 2s is not useful for this mapping.
10. **Pretrained finetune (iter054) disappointed** — only 0.589, suggesting pretraining needs more data/scale.

## Next Priorities (ranked by expected impact)

1. **Unified pretraining at scale** — Currently running: 40 HBN + 15 Ear-SAAD (55 subjects). If pretraining on 243 subjects doesn't break through 0.638, the ceiling is fundamental.
2. **Add remaining datasets to pretraining** — Incorporate Mobile BCI (18), EESM (22), MOABB (50) for 243 total subjects.
3. **Few-shot calibration** — Use 30-60s of target-subject data to adapt pretrained model (inspired by fewshot_calibration.md research).
4. **Test-time training (TTT)** — Self-supervised adaptation at inference using reconstruction objective.
5. **Channel attention with pretraining** — Let the model learn which input channels matter per-subject.
6. **Riemannian alignment** — Proper geometric alignment instead of naive Euclidean (which failed in iter042).
7. **Optimal transport domain adaptation** — Map subject-specific distributions to a common space.
8. **Write results paper** — Document the architecture sweep null result and the scaling law finding.

## Theoretical Ceiling Analysis
- Subject 3 gets r=0.94 with just CF -- the signal IS there for some subjects
- The variance across subjects (std=0.11) suggests ~0.16 r is "on the table" via better adaptation
- If we brought worst subjects to median: mean r would go from 0.638 to ~0.75
- Foundation model pretraining could add another +0.05-0.10
- **Realistic target: r=0.75-0.85 with pretraining + calibration**

## Resource Constraints
- **GPU**: RTX 4060, 8GB VRAM — one training job at a time
- **RAM**: 30GB — don't run parallel preprocessing + training
- **Storage**: ~10GB free for additional datasets
- **Compute time**: ~10 min per 3-fold benchmark, ~4 hours for 50-trial Optuna
- **Pretraining data**: 243 subjects across 5 datasets available
