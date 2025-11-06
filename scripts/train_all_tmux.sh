#!/usr/bin/env bash
# Parallel trainer: launch three tmux sessions pinned to CUDA 0..2 and
# randomly distribute the runs from train_all.sh across them.

set -euo pipefail

# Optional flags:
#   --skip-first-run     Skip the first generated run (keeps parity with train_all.sh)
#   --session-prefix STR Customize tmux session name prefix (default: train_all_gpu)

SKIP_FIRST_RUN=false
SKIP_COMPLETED=false
SESSION_PREFIX="train_all_gpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-first-run)
      SKIP_FIRST_RUN=true
      shift
      ;;
    --skip-completed|--resume)
      SKIP_COMPLETED=true
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
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure helper scripts are executable (idempotent)
chmod +x "${SCRIPT_DIR}/tmux_run_wrapper.sh" \
           "${SCRIPT_DIR}/tmux_queue_worker.sh" \
           "${SCRIPT_DIR}/tmux_runs_status.sh" 2>/dev/null || true

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

# Consider a run completed if its checkpoint/evaluation summary exists
is_run_completed() {
  local arch="$1" size="$2" mode="$3"
  local project="cellarc100k_50e_${mode}_${size}"
  local run_id="${arch}_${size}_${mode}"
  local checkpoint_dir="${REPO_ROOT}/outputs/${project}/${run_id}"
  [[ -f "${checkpoint_dir}/evaluation.json" ]]
}

# Build the list of sizes from remaining args, default to medium large small
SIZES=("$@")
if [[ ${#SIZES[@]} -eq 0 ]]; then
  SIZES=(medium large small)
fi

# Prune incomplete outputs: for requested sizes and both modes, remove run dirs
# that do not contain any checkpoint file (*.pt). Useful for crashed OOM runs.
prune_incomplete_outputs() {
  local -a sizes=("$@")
  local removed=0
  for mode in "${MODES[@]}"; do
    for size in "${sizes[@]}"; do
      local project="cellarc100k_50e_${mode}_${size}"
      local root="${REPO_ROOT}/outputs/${project}"
      [[ -d "${root}" ]] || continue
      while IFS= read -r -d '' run_dir; do
        # If no .pt files inside the run directory, remove it
        if ! compgen -G "${run_dir}"/*.pt > /dev/null; then
          echo "Pruning incomplete run dir: ${run_dir}"
          rm -rf -- "${run_dir}"
          ((removed+=1))
        fi
      done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -print0)
    done
  done
  if (( removed > 0 )); then
    echo "Pruned ${removed} incomplete output directorie(s)."
  fi
}

if [[ "${SKIP_COMPLETED}" == "true" ]]; then
  prune_incomplete_outputs "${SIZES[@]}"
fi

# Status/manifest directory for this runset
RUNSET_ID="$(date +%Y%m%d-%H%M%S)"
STATUS_ROOT="${REPO_ROOT}/outputs/tmux_runs"
STATUS_DIR="${STATUS_ROOT}/${RUNSET_ID}"
mkdir -p "${STATUS_DIR}"/{planned,in_progress,succeeded,failed,logs}
MANIFEST="${STATUS_DIR}/manifest.tsv"
echo -e "run_id\tarch\tsize\tmode\tgpu\tsession\tcommand" > "${MANIFEST}"

declare -a RUN_IDS RUN_ARCHES RUN_SIZES RUN_MODES RUN_CMDS
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
  local run_id="${arch}_${size}_${mode}"
  RUN_IDS+=("${run_id}")
  RUN_ARCHES+=("${arch}")
  RUN_SIZES+=("${size}")
  RUN_MODES+=("${mode}")
  RUN_CMDS+=("${cmd[*]}")
}

for size in "${SIZES[@]}"; do
  for mode in "${MODES[@]}"; do
    for arch in "${ARCHES[@]}"; do
      if [[ "${mode}" == "embedding" ]] && ! supports_embedding "${arch}"; then
        continue
      fi
      if [[ "${SKIP_COMPLETED}" == "true" ]] && is_run_completed "${arch}" "${size}" "${mode}"; then
        echo "--- Skipping completed '${arch}' (${size}) mode=${mode} ---"
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

if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
  echo "No runs were generated (check filters/flags)." >&2
  exit 1
fi

count=${#RUN_IDS[@]}

# Shuffle indices to randomize processing order
shuffle_indices() {
  local n=$1
  if command -v shuf >/dev/null 2>&1; then
    shuf -i 0-$((n-1))
  else
    # Fallback: simple seq (no randomization if shuf missing)
    seq 0 $((n-1))
  fi
}

mapfile -t IDX_SHUFFLED < <(shuffle_indices "$count")

# Write all planned runs to the status directory; workers will claim dynamically
for i in "${IDX_SHUFFLED[@]}"; do
  run_id="${RUN_IDS[$i]}"
  arch="${RUN_ARCHES[$i]}"
  size="${RUN_SIZES[$i]}"
  mode="${RUN_MODES[$i]}"
  run_cmd="${RUN_CMDS[$i]}"
  # Record planned entry and manifest row (GPU/session unknown yet)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_id}" "${arch}" "${size}" "${mode}" "-" "-" "${run_cmd}" >> "${MANIFEST}"
  printf "gpu=-\nsession=-\narch=%s\nsize=%s\nmode=%s\n" \
    "${arch}" "${size}" "${mode}" > "${STATUS_DIR}/planned/${run_id}.meta"
  printf "%s\n" "${run_cmd}" > "${STATUS_DIR}/planned/${run_id}.cmd"
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
  tmux send-keys -t "${session_name}" "cd '${REPO_ROOT}'" C-m
  tmux send-keys -t "${session_name}" "export CUDA_VISIBLE_DEVICES=${gpu}" C-m
  tmux send-keys -t "${session_name}" "export PYTHONUNBUFFERED=1" C-m
  tmux send-keys -t "${session_name}" "export RUNS_STATUS_DIR='${STATUS_DIR}'" C-m
  tmux send-keys -t "${session_name}" "export TMUX_SESSION_NAME='${session_name}'" C-m
  tmux send-keys -t "${session_name}" "echo 'Session ${session_name} using CUDA_VISIBLE_DEVICES=${gpu}'" C-m

  # Start worker loop that will claim and run jobs dynamically
  tmux send-keys -t "${session_name}" "echo '[GPU ${gpu}] Queue worker starting'" C-m
  tmux send-keys -t "${session_name}" "bash scripts/tmux_queue_worker.sh '${gpu}'" C-m

  tmux send-keys -t "${session_name}" "echo 'All assigned runs finished for ${session_name}.'" C-m
done

echo "Launched tmux sessions:"
for s in "${SESSION_NAMES[@]}"; do
  echo "  - ${s}  (attach: tmux attach -t ${s})"
done

mkdir -p "${STATUS_ROOT}"
ln -sfn "${STATUS_DIR}" "${STATUS_ROOT}/latest"
echo "Runset status directory: ${STATUS_DIR}"
echo "Summary: scripts/tmux_runs_status.sh  # uses outputs/tmux_runs/latest by default"
