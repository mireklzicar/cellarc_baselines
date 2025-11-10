#!/usr/bin/env bash
set -euo pipefail

# Execute the configured LLM baselines (currently GPT-5) over the evaluation splits.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

declare -a LLM_MODELS=(
  "gpt-5.0-mini"
)

for model in "${LLM_MODELS[@]}"; do
  echo "=== Evaluating LLM baseline (OpenAI / ${model}) ==="
  python "${REPO_ROOT}/scripts/eval_llm.py" \
    llm.model="${model}" \
    eval.progress_bar=true \
    eval.max_episodes=null \
    "$@"
done
