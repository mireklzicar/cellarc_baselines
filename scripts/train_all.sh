#!/usr/bin/env bash
# Train every baseline architecture across the small, medium, and large size presets.

set -euo pipefail

SKIP_FIRST_RUN=false
if [[ "${1:-}" == "--skip-first-run" ]]; then
  SKIP_FIRST_RUN=true
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train.py"

ARCHES=(
  transformer
#  transformer_ar
  transformer_act
  cnn1d
#  nca1d
  tiny_recursive
  hrm
  rnn
#  rnn_ar
)

EMBEDDING_ARCHES=(
  transformer
#  transformer_ar
  transformer_act
  cnn1d
#  nca1d
  tiny_recursive
  hrm
  rnn
#  rnn_ar
)

MODES=(
  embedding
  incontext
)

supports_embedding() {
  local candidate="$1"
  for emb_arch in "${EMBEDDING_ARCHES[@]}"; do
    if [[ "${candidate}" == "${emb_arch}" ]]; then
      return 0
    fi
  done
  return 1
}

run_idx=0

train_band() {
  local size="$1"
  local mode="$2"
  echo "=== Starting ${size} models (mode=${mode}) ==="
  for arch in "${ARCHES[@]}"; do
    if [[ "${mode}" == "embedding" ]] && ! supports_embedding "${arch}"; then
      echo "--- Skipping '${arch}' (${size}) for mode=${mode} (unsupported) ---"
      continue
    fi

    if [[ "${SKIP_FIRST_RUN}" == "true" && ${run_idx} -eq 0 ]]; then
      echo "--- Skipping first run ${arch} (${size}) mode=${mode} ---"
      SKIP_FIRST_RUN=false
      ((run_idx+=1))
      continue
    fi

    echo "--- Training '${arch}' (${size}) mode=${mode} ---"
    python "${TRAIN_PY}" \
      --config-name train/default \
      model.architecture="${arch}" \
      model/size="${size}" \
      training.mode="${mode}" \
      trainer.checkpoints.enabled=true \
      logging.wandb.enabled=true \
      logging.wandb.project="cellarc100k_50e_${mode}_${size}" \
      logging.wandb.group="mode_${mode}" \
      logging.wandb.name="${arch}_${size}_${mode}"
    ((run_idx+=1))
  done
}

SIZES=("$@")
if [[ ${#SIZES[@]} -eq 0 ]]; then
  SIZES=(medium large small)
fi

for size in "${SIZES[@]}"; do
  for mode in "${MODES[@]}"; do
    train_band "${size}" "${mode}"
  done
done
