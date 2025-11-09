#!/usr/bin/env bash
# Evaluate GPT-5 on the Hugging Face 100-episode evaluation splits.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Toggle this to select the default model. Uncomment the GPT-5 line to evaluate the larger model.
MODEL="${MODEL:-gpt-5-2025-08-07}"

SMOKE_TEST="${SMOKE_TEST:-false}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${REPO_ROOT}/outputs/hf_datasets}"
INCLUDE_METADATA="${INCLUDE_METADATA:-false}"
FORCE_REFRESH="${FORCE_REFRESH:-false}"
ASYNC_CONCURRENCY="${ASYNC_CONCURRENCY:-5}"
WORKER_POOL="${WORKER_POOL:-5}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-200000}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"

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
  PREDICTION_LOG="${PREDICTION_LOG:-outputs/llm/gpt5_nano_hf100_smoke_predictions.jsonl}"
else
  MAX_EPISODES=100
  PROGRESS=true
  PREDICTION_LOG="${PREDICTION_LOG:-outputs/llm/gpt5_nano_hf100_predictions.jsonl}"
fi

if [[ "${PREDICTION_LOG}" = /* ]]; then
  LOG_DIR="$(dirname "${PREDICTION_LOG}")"
else
  LOG_DIR="$(dirname "${REPO_ROOT}/${PREDICTION_LOG}")"
fi
mkdir -p "${LOG_DIR}"

pushd "${REPO_ROOT}" >/dev/null

python scripts/eval_llm.py \
  llm.model="${MODEL}" \
  llm.api_style=responses \
  llm.reasoning_effort=high \
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
