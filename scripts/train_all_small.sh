#!/usr/bin/env bash
# Run the default training schedule for every small baseline architecture.

set -euo pipefail

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

for arch in "${ARCHES[@]}"; do
  echo "=== Training '${arch}' (small) ==="
  python "$(dirname "${BASH_SOURCE[0]}")/train.py" \
    --config-name train/single_epoch \
    model.architecture="${arch}" \
    logging.wandb.enabled=true \
    logging.wandb.project=ca_tmp_new
done
