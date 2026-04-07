# EEG Scalp-to-In-Ear Downsampling

Learn a mapping from 21-channel scalp EEG (10-20 system) to 4-channel in-ear EEG, enabling virtual in-ear channels from standard clinical montages.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full pipeline (synthetic data)
bash scripts/run_all.sh

# Or train individual models
uv run python -m src.train --config configs/base.yaml          # Closed-form baseline
uv run python -m src.train --config configs/model1_linear.yaml  # Linear spatial filter
uv run python -m src.train --config configs/model2_fir.yaml     # Spatio-temporal FIR
uv run python -m src.train --config configs/model3_conv.yaml    # Convolutional encoder

# Evaluate a trained model
uv run python -m src.evaluate --config configs/model2_fir.yaml --checkpoint results/checkpoints/fir_filter_best.pt

# Run hyperparameter sweeps
uv run python scripts/sweep_hparams.py --model model2 --n-trials 30

# Run ablation studies
uv run python scripts/ablation.py --study loss
```

## Models

| Model | Type | Parameters | Description |
|-------|------|-----------|-------------|
| Closed-form | Linear | 84 | Optimal W* = R_YX @ inv(R_XX), baseline |
| Linear Spatial | Linear | 84 | SGD-trained W, same solution as closed-form |
| FIR Filter | Linear | ~5K | 1D convolution with temporal context |
| Conv Encoder | Nonlinear | ~50-200K | Depthwise-separable convolutions with residual connections |

## Project Structure

```
configs/          # YAML configuration files
src/
  data/           # Download, preprocessing, dataset, synthetic generation
  models/         # All model architectures
  losses/         # MSE, spectral, band power, combined losses
  metrics/        # Pearson r, RMSE, SNR, coherence, band power correlation
  train.py        # Training loop
  evaluate.py     # Evaluation script
  visualize.py    # Plotting utilities
scripts/          # Pipeline, hyperparameter sweeps, ablation studies
notebooks/        # Analysis notebook
```

## Loss Functions

- **Time-domain MSE**: Standard reconstruction loss
- **Spectral loss**: Log-magnitude FFT matching
- **Band power loss**: Power matching in delta/theta/alpha/beta/gamma bands
- **Combined**: Weighted sum of all three

## Evaluation Metrics

- Pearson correlation, RMSE, relative RMSE, SNR (dB)
- Band power correlation, magnitude-squared coherence, spectral RMSE
