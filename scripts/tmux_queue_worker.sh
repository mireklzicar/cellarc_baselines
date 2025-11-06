#!/usr/bin/env bash
# A worker loop that claims the next available planned run and executes it.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: tmux_queue_worker.sh <gpu_id> [poll_interval_seconds]" >&2
  exit 2
fi

GPU_ID="$1"
POLL_INTERVAL="${2:-2}"

if [[ -z "${RUNS_STATUS_DIR:-}" ]]; then
  echo "RUNS_STATUS_DIR must be exported in the environment." >&2
  exit 2
fi

echo "[worker gpu=${GPU_ID}] Starting queue worker in ${RUNS_STATUS_DIR}"

claim_next() {
  shopt -s nullglob
  local files=("${RUNS_STATUS_DIR}/planned"/*.cmd)
  if (( ${#files[@]} == 0 )); then
    return 1
  fi
  # Randomize order to reduce contention
  local shuffled
  if command -v shuf >/dev/null 2>&1; then
    mapfile -t shuffled < <(printf '%s\n' "${files[@]}" | shuf)
  else
    shuffled=("${files[@]}")
  fi
  for f in "${shuffled[@]}"; do
    local base
    base="$(basename "${f}")"
    local run_id="${base%.cmd}"
    local dest_cmd="${RUNS_STATUS_DIR}/in_progress/${run_id}.cmd"
    # Try to atomically move the cmd file into in_progress to claim it
    if mv "${f}" "${dest_cmd}" 2>/dev/null; then
      # Move meta if present (best-effort)
      local meta_src="${RUNS_STATUS_DIR}/planned/${run_id}.meta"
      local meta_dst="${RUNS_STATUS_DIR}/in_progress/${run_id}.meta"
      [[ -f "${meta_src}" ]] && mv "${meta_src}" "${meta_dst}" 2>/dev/null || true
      echo "[worker gpu=${GPU_ID}] Claimed ${run_id}"
      # Export for child processes
      export CLAIMED_RUN_ID="${run_id}"
      return 0
    fi
  done
  return 1
}

while true; do
  if claim_next; then
    run_id="${CLAIMED_RUN_ID}"
    if ! scripts/tmux_run_wrapper.sh "${run_id}" "${GPU_ID}"; then
      status=$?
      echo "[worker gpu=${GPU_ID}] Run ${run_id} failed with exit status ${status}; continuing." >&2
    fi
    unset CLAIMED_RUN_ID
    # Immediately try to claim another without sleeping
    continue
  fi
  # No jobs to claim; if queues are empty and no in-progress exists, exit
  shopt -s nullglob
  planned_left=("${RUNS_STATUS_DIR}/planned"/*.cmd)
  inprog_left=("${RUNS_STATUS_DIR}/in_progress"/*.cmd)
  if (( ${#planned_left[@]} == 0 && ${#inprog_left[@]} == 0 )); then
    echo "[worker gpu=${GPU_ID}] No jobs left. Exiting."
    break
  fi
  sleep "${POLL_INTERVAL}"
done
