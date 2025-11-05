#!/usr/bin/env bash
# Train tiny_recursive in four variants:
#   - embedding (puzzle-embedding), no TTA
#   - embedding, with TTA
#   - incontext, no TTA
#   - incontext, with TTA
# Sizes match the original default: medium and large.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train.py"

# tiny_recursive
ARCHES=( transformer )
SIZES=( medium large )

train_one() {
  local size="$1"      # small | medium | large
  local mode="$2"      # embedding | incontext
  local tta="$3"       # true | false

  for arch in "${ARCHES[@]}"; do
    echo "--- Training '${arch}' (${size}) mode=${mode} tta=${tta} ---"
    local run_name="${arch}_${mode}_tta_${tta}"
    python "${TRAIN_PY}" \
      --config-name train/single_epoch \
      model.architecture="${arch}" \
      model/size="${size}" \
      training.mode="${mode}" \
      eval.tta.enabled="${tta}" \
      trainer.checkpoints.enabled=true \
      logging.wandb.enabled=true \
      logging.wandb.project="single_trm_test_${size}_${mode}" \
      logging.wandb.group="trm_${mode}_$( [[ "${tta}" == "true" ]] && echo tta || echo notta )" \
      logging.wandb.name="${run_name}"
  done
}

run_band() {
  local mode="$1"   # embedding | incontext
  local tta="$2"    # true | false
  for size in "${SIZES[@]}"; do
    echo "=== Starting ${size} (${mode}, tta=${tta}) ==="
    train_one "${size}" "${mode}" "${tta}"
  done
}

# 1) Puzzle-embedding, no TTA (matches previous default behavior, just explicit)
run_band embedding false

# 2) Puzzle-embedding, with TTA
#run_band embedding true

# 3) In-context, no TTA
run_band incontext false

# 4) In-context, with TTA
#run_band incontext true
