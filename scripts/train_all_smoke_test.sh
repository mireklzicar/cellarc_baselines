#!/usr/bin/env bash
# Train every baseline architecture across the small, medium, and large size presets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train.py"

ARCHES=(
  rnn
  rnn_ar
  transformer
  transformer_ar
  transformer_act
  cnn1d
  nca1d
  tiny_recursive
  hrm
)

EMBEDDING_ARCHES=(
  rnn
  rnn_ar
  transformer
  transformer_ar
  transformer_act
  cnn1d
  nca1d
  tiny_recursive
  hrm
)

MODES=(
  incontext
  embedding
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

train_band() {
  local size="$1"
  local mode="$2"
  echo "=== Starting ${size} models (mode=${mode}) ==="
  for arch in "${ARCHES[@]}"; do
    if [[ "${mode}" == "embedding" ]] && ! supports_embedding "${arch}"; then
      echo "--- Skipping '${arch}' (${size}) for mode=${mode} (unsupported) ---"
      continue
    fi

    echo "--- Training '${arch}' (${size}) mode=${mode} ---"
    python "${TRAIN_PY}" \
      --config-name train/smoke \
      model.architecture="${arch}" \
      model/size="${size}" \
      training.mode="${mode}" \
      trainer.checkpoints.enabled=false \
      logging.wandb.enabled=false \
      logging.wandb.project="cellarc100k_${mode}_baselines_${size}" \
      logging.wandb.group="mode_${mode}" \
      logging.wandb.name="${arch}_${size}_${mode}"
  done
}

SIZES=("$@")
if [[ ${#SIZES[@]} -eq 0 ]]; then
  SIZES=(small medium large)
fi

for mode in "${MODES[@]}"; do
  for size in "${SIZES[@]}"; do
    train_band "${size}" "${mode}"
  done
done
