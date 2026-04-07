#!/usr/bin/env bash
# Full pipeline: generate data -> train all models -> evaluate -> visualize
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== EEG Scalp-to-In-Ear Downsampling Pipeline ==="
echo "Working directory: $PROJECT_DIR"

# Step 0: Ensure directories exist
mkdir -p data/raw data/processed data/splits
mkdir -p results/checkpoints results/logs results/figures

# Step 1: Train closed-form baseline
echo ""
echo "=== Step 1: Closed-form baseline ==="
python -m src.train --config configs/base.yaml 2>&1 | tee results/logs/closed_form.log

# Step 2: Train Model 1 (Linear Spatial Filter)
echo ""
echo "=== Step 2: Model 1 - Linear Spatial Filter ==="
python -m src.train --config configs/model1_linear.yaml 2>&1 | tee results/logs/linear_spatial.log

# Step 3: Train Model 2 (FIR Filter)
echo ""
echo "=== Step 3: Model 2 - Spatio-Temporal FIR Filter ==="
python -m src.train --config configs/model2_fir.yaml 2>&1 | tee results/logs/fir_filter.log

# Step 4: Train Model 3 (Convolutional Encoder)
echo ""
echo "=== Step 4: Model 3 - Convolutional Encoder ==="
python -m src.train --config configs/model3_conv.yaml 2>&1 | tee results/logs/conv_encoder.log

# Step 5: Run ablation study
echo ""
echo "=== Step 5: Ablation study ==="
python scripts/ablation.py 2>&1 | tee results/logs/ablation.log

echo ""
echo "=== Pipeline complete ==="
echo "Results saved in results/"
