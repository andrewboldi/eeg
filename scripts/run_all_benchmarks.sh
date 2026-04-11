#!/usr/bin/env bash
# Run all pending model benchmarks sequentially on 46ch broadband data.
# Usage: PYTHONPATH=. bash scripts/run_all_benchmarks.sh

set -e
export PYTHONPATH=/home/andrew/eeg
export CUDA_VISIBLE_DEVICES=0

BENCH="uv run python scripts/benchmark_broadband_46ch.py"
DATA2S="data/processed/broadband_46ch.h5"

echo "=== Starting benchmark sweep at $(date) ==="

# Standard 2s models (use default 46ch benchmark)
for model in \
    "iter044_residual_cf" \
    "iter046_calibrated" \
    "iter047_spatial_pe" \
    "iter048_neurottt" \
    "iter049_adversarial" \
    "iter050_l1_loss" \
    "iter051_perchannel_heads" \
    "iter052_cross_attn_decoder" \
    "iter053_spectral_loss"; do
    echo ""
    echo "=== Running $model at $(date) ==="
    $BENCH --model-fn "models/${model}.py" --name "${model}_46ch" || echo "FAILED: $model"
    echo "=== Finished $model at $(date) ==="
done

# 4s model needs special data path
echo ""
echo "=== Running iter045_long_context (4s data) at $(date) ==="
$BENCH --model-fn models/iter045_long_context.py --name iter045_long_context_4s --data data/processed/broadband_46ch_4s.h5 || echo "FAILED: iter045_long_context"

echo ""
echo "=== All benchmarks complete at $(date) ==="
echo "Results in results/benchmark/leaderboard_broadband_46ch.jsonl"
