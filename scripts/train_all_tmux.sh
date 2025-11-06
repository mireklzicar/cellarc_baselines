#!/usr/bin/env bash
# Parallel trainer: launch three tmux sessions pinned to CUDA 0..2 and
# randomly distribute the runs from train_all.sh across them.

set -euo pipefail

# Optional flags:
#   --skip-first-run     Skip the first generated run (keeps parity with train_all.sh)
#   --session-prefix STR Customize tmux session name prefix (default: train_all_gpu)

SKIP_FIRST_RUN=false
SESSION_PREFIX="train_all_gpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-first-run)
      SKIP_FIRST_RUN=true
      shift
      ;;
    --session-prefix)
      SESSION_PREFIX="$2"; shift 2
      ;;
    --)
      shift; break
      ;;
    -*)
      echo "Unknown option: $1" >&2; exit 1
      ;;
    *)
      break
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train.py"

# GPU list (default 0,1,2)
GPUS=(0 1 2)

# Architectures and modes mirror scripts/train_all.sh
ARCHES=(
  transformer
  transformer_act
  cnn1d
  tiny_recursive
  hrm
  rnn
)

EMBEDDING_ARCHES=(
  transformer
  transformer_act
  cnn1d
  tiny_recursive
  hrm
  rnn
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

# Build the list of sizes from remaining args, default to medium large small
SIZES=("$@")
if [[ ${#SIZES[@]} -eq 0 ]]; then
  SIZES=(large medium small)
fi

declare -a RUNS
run_idx=0

add_run() {
  local arch="$1" size="$2" mode="$3"
  # Hydrated command (single line) to execute inside tmux session
  local cmd
  cmd=(
    python -m scripts.train
    --config-name train/default
    "model.architecture=${arch}"
    "model/size=${size}"
    "training.mode=${mode}"
    trainer.checkpoints.enabled=true
    logging.wandb.enabled=true
    "logging.wandb.project=cellarc100k_50e_${mode}_${size}"
    "logging.wandb.group=mode_${mode}"
    "logging.wandb.name=${arch}_${size}_${mode}"
  )
  RUNS+=("${cmd[*]}")
}

for size in "${SIZES[@]}"; do
  for mode in "${MODES[@]}"; do
    for arch in "${ARCHES[@]}"; do
      if [[ "${mode}" == "embedding" ]] && ! supports_embedding "${arch}"; then
        continue
      fi
      if [[ "${SKIP_FIRST_RUN}" == "true" && ${run_idx} -eq 0 ]]; then
        # Maintain parity with train_all.sh: skip the first generated run
        ((run_idx+=1))
        continue
      fi
      add_run "${arch}" "${size}" "${mode}"
      ((run_idx+=1))
    done
  done
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "No runs were generated (check filters/flags)." >&2
  exit 1
fi

# Shuffle the run list to randomize distribution
shuffle_runs() {
  if command -v shuf >/dev/null 2>&1; then
    printf '%s\n' "${RUNS[@]}" | shuf
  else
    # Portable fallback: random key sort
    awk 'BEGIN{srand()} {print rand()"\t"$0}' | sort -k1,1n | cut -f2-
  fi
}

mapfile -t RUNS_SHUFFLED < <(shuffle_runs)

# Split runs round-robin across GPUs (balanced but randomized order)
declare -a RUNS_BY_GPU
for i in "${!RUNS_SHUFFLED[@]}"; do
  gpu_index=$(( i % ${#GPUS[@]} ))
  # Append with a real newline so each command is a line when read back
  RUNS_BY_GPU[$gpu_index]="${RUNS_BY_GPU[$gpu_index]-}${RUNS_SHUFFLED[$i]}"$'\n'
done

# Helper to assert session doesn't exist
ensure_session_absent() {
  local session_name="$1"
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session '${session_name}' already exists. Please kill it first: tmux kill-session -t ${session_name}" >&2
    exit 1
  fi
}

# Create tmux sessions and feed commands
SESSION_NAMES=()
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  session_name="${SESSION_PREFIX}${gpu}"
  SESSION_NAMES+=("${session_name}")
  ensure_session_absent "${session_name}"

  # Start a bash login shell to ensure predictable behavior for sent commands
  tmux new-session -d -s "${session_name}" bash -l >/dev/null
  # Verify session exists (fail fast if creation had issues)
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Failed to create tmux session '${session_name}'." >&2
    exit 1
  fi
  tmux send-keys -t "${session_name}" "export CUDA_VISIBLE_DEVICES=${gpu}" C-m
  tmux send-keys -t "${session_name}" "export PYTHONUNBUFFERED=1" C-m
  tmux send-keys -t "${session_name}" "echo 'Session ${session_name} using CUDA_VISIBLE_DEVICES=${gpu}'" C-m

  # Queue each run assigned to this GPU
  if [[ -n "${RUNS_BY_GPU[$idx]:-}" ]]; then
    # Iterate over newline-separated commands
    while IFS= read -r run_cmd; do
      [[ -z "${run_cmd}" ]] && continue
      tmux send-keys -t "${session_name}" "echo '[GPU ${gpu}] >> ${run_cmd}'" C-m
      tmux send-keys -t "${session_name}" "${run_cmd}" C-m
    done <<< "${RUNS_BY_GPU[$idx]}"
  else
    tmux send-keys -t "${session_name}" "echo 'No runs assigned to this session.'" C-m
  fi

  tmux send-keys -t "${session_name}" "echo 'All assigned runs finished for ${session_name}.'" C-m
done

echo "Launched tmux sessions:"
for s in "${SESSION_NAMES[@]}"; do
  echo "  - ${s}  (attach: tmux attach -t ${s})"
done
