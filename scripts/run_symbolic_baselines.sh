#!/usr/bin/env bash
set -euo pipefail

# Execute all symbolic baselines over the full evaluation splits.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

declare -a SYMBOLIC_BASELINES=(
  "copycat"
  "most_frequent"
  "de_bruijn"
  "random"
)

for baseline in "${SYMBOLIC_BASELINES[@]}"; do
  echo "=== Evaluating symbolic baseline: ${baseline} ==="
  python "${REPO_ROOT}/scripts/eval_symbolic.py" \
    baseline.name="${baseline}" \
    eval.progress_bar=true \
    eval.max_episodes=null
done
