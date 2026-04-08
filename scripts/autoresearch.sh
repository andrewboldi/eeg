#!/usr/bin/env bash
# Autoresearch runner — kicks off Claude Code to run one iteration.
# Usage:
#   bash scripts/autoresearch.sh          # Run one iteration
#   bash scripts/autoresearch.sh --loop   # Run continuously (30 min intervals)

set -euo pipefail
cd "$(dirname "$0")/.."

PROMPT='You are the EEG autoresearch agent. Read CLAUDE.md for the full protocol and current state. Run ONE iteration of the autoresearch loop:

1. RESEARCH: Check results/benchmark/leaderboard.jsonl for what has been tried. Search the web for new EEG/time-series ML techniques. Download any relevant arXiv paper TeX sources to docs/external/.
2. HYPOTHESIZE: Pick the most promising idea from the research queue (or invent a new one based on literature). State your hypothesis clearly.
3. IMPLEMENT: Create models/iter{NNN}_{name}.py implementing build_and_train().
4. EVALUATE: Run `uv run python scripts/benchmark.py --model-fn models/iter{NNN}_{name}.py --name iter{NNN}_{name}`
5. ANALYZE: Write docs/src/report_{NNN}.tex. Build with `cd docs && make`.
6. UPDATE: Update CLAUDE.md leaderboard. Commit and push everything.

Current best: check CLAUDE.md. Your goal is to MAXIMIZE mean Pearson r on the fixed test set (subjects 13-15).

After completing this iteration, if the result improved, explain why. If not, diagnose what went wrong and what to try next. Do NOT ask for permission — just execute the full loop.'

if [[ "${1:-}" == "--loop" ]]; then
    echo "=== Autoresearch loop mode (30 min intervals) ==="
    while true; do
        echo "$(date): Starting iteration..."
        claude --print "$PROMPT" 2>&1 | tee -a results/autoresearch.log
        echo "$(date): Iteration complete. Sleeping 30 minutes..."
        sleep 1800
    done
else
    echo "=== Running one autoresearch iteration ==="
    claude --print "$PROMPT"
fi
