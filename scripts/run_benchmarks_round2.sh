#!/usr/bin/env bash
# Run iter055-066 model benchmarks sequentially on 46ch broadband data.
# Usage: PYTHONPATH=. bash scripts/run_benchmarks_round2.sh

set -e
export PYTHONPATH=/home/andrew/eeg
export CUDA_VISIBLE_DEVICES=0

BENCH="uv run python scripts/benchmark_broadband_46ch.py"

echo "=== Starting round 2 benchmark sweep at $(date) ==="

for model in \
    "iter055_two_stage" \
    "iter056_heavy_augment" \
    "iter057_unified_pretrained" \
    "iter058_subject_mixup" \
    "iter059_within_subj_mixup" \
    "iter060_reptile" \
    "iter061_deep_coral" \
    "iter062_channel_attention" \
    "iter063_gopsa" \
    "iter064_target_denoise" \
    "iter065_highpass3hz" \
    "iter066_contrastive_finetune"; do
    echo ""
    echo "=== Running $model at $(date) ==="
    $BENCH --model-fn "models/${model}.py" --name "${model}_46ch" || echo "FAILED: $model"
    echo "=== Finished $model at $(date) ==="
done

echo ""
echo "=== All round 2 benchmarks complete at $(date) ==="
echo "Results in results/benchmark/leaderboard_broadband_46ch.jsonl"
