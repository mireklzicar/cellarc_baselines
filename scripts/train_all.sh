#!/usr/bin/env bash
# Train every baseline architecture across the small, medium, and large size presets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train.py"

ARCHES=(
  rnn
  transformer
  cnn1d
  nca1d
  tiny_recursive
  hrm
  tape_rnn
  stack_rnn
)

train_band() {
  local size="$1"
  echo "=== Starting ${size} models ==="
  for arch in "${ARCHES[@]}"; do
    echo "--- Training '${arch}' (${size}) ---"
    python "${TRAIN_PY}" \
      --config-name train/single_epoch \
      model.architecture="${arch}" \
      model.size="${size}" \
      trainer.checkpoints.enabled=true \
      logging.wandb.enabled=true \
      logging.wandb.project=ca_tmp_new_2
  done
}

train_band small
train_band medium
train_band large
