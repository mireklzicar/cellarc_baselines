#!/usr/bin/env bash
# Evaluate GPT-5 on the Hugging Face 100-episode evaluation splits.

set -euo pipefail

sanitize_for_filename() {
  local input="$1"
  echo "${input}" | sed 's/[^A-Za-z0-9._-]/_/g'
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# Toggle this to select the default model. Uncomment the GPT-5 line to evaluate the larger model.
MODEL="${MODEL:-gpt-5-2025-08-07}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"

SMOKE_TEST="${SMOKE_TEST:-false}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${REPO_ROOT}/outputs/hf_datasets}"
INCLUDE_METADATA="${INCLUDE_METADATA:-false}"
FORCE_REFRESH="${FORCE_REFRESH:-false}"
ASYNC_CONCURRENCY="${ASYNC_CONCURRENCY:-10}"
WORKER_POOL="${WORKER_POOL:-10}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-200000}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"

if [[ "${FORCE_REFRESH,,}" == "true" ]]; then
  FORCE_FLAG="True"
else
  FORCE_FLAG="False"
fi

if [[ "${INCLUDE_METADATA,,}" == "true" ]]; then
  META_FLAG="True"
else
  META_FLAG="False"
fi

export HF_CACHE_ROOT
export FORCE_FLAG
export META_FLAG

python - <<'PY'
import os
from pathlib import Path
from cellarc import download_benchmark

print(f"[run_hf_nano_eval] Syncing Hugging Face snapshot into {os.environ['HF_CACHE_ROOT']}")

root = Path(os.environ["HF_CACHE_ROOT"])
root.mkdir(parents=True, exist_ok=True)

download_benchmark(
    name="cellarc_100k",
    include_metadata=os.environ.get("META_FLAG", "False") == "True",
    root=root,
    force_download=os.environ.get("FORCE_FLAG", "False") == "True",
)
PY

DATASET_ROOT="${HF_CACHE_ROOT}"

if [[ "${SMOKE_TEST,,}" == "true" ]]; then
  MAX_EPISODES=10
  PROGRESS=true
  SMOKE_SUFFIX="-smoke"
else
  MAX_EPISODES=100
  PROGRESS=true
  SMOKE_SUFFIX=""
fi

MODEL_TAG="$(sanitize_for_filename "${MODEL}")"
REASONING_SUFFIX=""
if [[ -n "${REASONING_EFFORT}" && "${REASONING_EFFORT,,}" != "null" ]]; then
  REASONING_SUFFIX="-$(sanitize_for_filename "${REASONING_EFFORT}")"
fi

if [[ -z "${PREDICTION_LOG:-}" ]]; then
  PREDICTION_LOG="outputs/llm/${MODEL_TAG}${REASONING_SUFFIX}${SMOKE_SUFFIX}_predictions.jsonl"
fi
export PREDICTION_LOG

ABS_PREDICTION_LOG="$(python - <<'PY'
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
log_path = Path(os.environ["PREDICTION_LOG"]).expanduser()
if not log_path.is_absolute():
    log_path = repo_root / log_path
print(log_path.resolve())
PY
)"

LOG_DIR="$(dirname "${ABS_PREDICTION_LOG}")"
mkdir -p "${LOG_DIR}"

echo "[run_gpt_eval] Prediction log target ${ABS_PREDICTION_LOG}"

if [[ -f "${ABS_PREDICTION_LOG}" ]]; then
  read -r -p "[run_gpt_eval] Prediction log ${ABS_PREDICTION_LOG} exists. Overwrite? [y/N]: " OVERWRITE_LOG
  if [[ "${OVERWRITE_LOG}" =~ ^[Yy]$ ]]; then
    : > "${ABS_PREDICTION_LOG}"
  else
    echo "[run_gpt_eval] Existing prediction log preserved; aborting run."
    exit 1
  fi
fi

pushd "${REPO_ROOT}" >/dev/null

python scripts/eval_llm.py \
  llm.model="${MODEL}" \
  llm.api_style=responses \
  llm.reasoning_effort="${REASONING_EFFORT}" \
  llm.max_output_tokens="${MAX_OUTPUT_TOKENS}" \
  llm.request_timeout="${REQUEST_TIMEOUT}" \
  llm.stream=true \
  llm.use_async=false \
  dataset.root="${DATASET_ROOT}" \
  dataset.include_metadata="${INCLUDE_METADATA}" \
  splits="[test_interpolation_100,test_extrapolation_100]" \
  eval.max_episodes="${MAX_EPISODES}" \
  eval.progress_bar="${PROGRESS}" \
  eval.prediction_log="${PREDICTION_LOG}" \
  eval.async_concurrency="${ASYNC_CONCURRENCY}" \
  eval.num_workers="${WORKER_POOL}"

popd >/dev/null

ABS_DATASET_ROOT="$(python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["HF_CACHE_ROOT"]).expanduser().resolve()
print(root)
PY
)"

echo "[run_gpt_eval] Prediction log saved to ${ABS_PREDICTION_LOG}"
echo "[run_gpt_eval] Hugging Face dataset cache at ${ABS_DATASET_ROOT}"
