#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="${SESSION_NAME:-llm_nano_smoke}"
WAIT_TOKEN="${SESSION_NAME}_done"
LOG_DIR="${REPO_ROOT}/outputs/llm/tmux_logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_${SESSION_NAME}.log"

COMMAND="python scripts/eval_llm.py \
  llm.model=gpt-5-nano-2025-08-07 \
  llm.api_style=responses \
  llm.reasoning_effort=low \
  llm.max_output_tokens=4096 \
  llm.request_timeout=120 \
  llm.stream=false \
  llm.use_async=true \
  eval.max_episodes=10 \
  eval.progress_bar=false \
  eval.prediction_log=outputs/llm/gpt5_nano_20250807_smoke_predictions.jsonl \
  eval.async_concurrency=8 \
  dataset.root=../cellarc/artifacts/datasets"

TMUX_CMD="cd ${REPO_ROOT} && START=\$(date +%s); ${COMMAND} |& tee ${LOG_PATH}; STATUS=\${PIPESTATUS[0]}; END=\$(date +%s); echo Runtime: \$((END-START)) seconds | tee -a ${LOG_PATH}; tmux wait-for -S ${WAIT_TOKEN}; exit \$STATUS"

tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${TMUX_CMD}'"
tmux wait-for "${WAIT_TOKEN}"
tmux kill-session -t "${SESSION_NAME}" >/dev/null 2>&1 || true

echo "Logs saved to ${LOG_PATH}"
