#!/usr/bin/env bash
# Summarize tmux training runsets created by scripts/train_all_tmux.sh

set -euo pipefail

USAGE="Usage: $(basename "$0") [RUNSET_DIR|outputs/tmux_runs/latest] [--list]"

TARGET_DIR="${1:-outputs/tmux_runs/latest}"
LIST=false
if [[ "${2:-}" == "--list" ]]; then
  LIST=true
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Runset directory not found: ${TARGET_DIR}" >&2
  echo "${USAGE}" >&2
  exit 1
fi

count_meta() {
  local d="$1"
  shopt -s nullglob
  local files=("${TARGET_DIR}/${d}"/*.meta)
  echo ${#files[@]}
}

planned=$(count_meta planned)
inprog=$(count_meta in_progress)
succ=$(count_meta succeeded)
fail=$(count_meta failed)
total=$((planned+inprog+succ+fail))

echo "Runset: ${TARGET_DIR}"
echo "- Planned:     ${planned}"
echo "- In progress: ${inprog}"
echo "- Succeeded:   ${succ}"
echo "- Failed:      ${fail}"
echo "- Total:       ${total}"

if ${LIST}; then
  for st in planned in_progress succeeded failed; do
    echo "${st}:"
    shopt -s nullglob
    for f in "${TARGET_DIR}/${st}"/*.meta; do
      run_id="$(basename "${f}" .meta)"
      arch=$(grep -E '^arch=' -m1 "${f}" | cut -d= -f2- || true)
      size=$(grep -E '^size=' -m1 "${f}" | cut -d= -f2- || true)
      mode=$(grep -E '^mode=' -m1 "${f}" | cut -d= -f2- || true)
      gpu=$(grep -E '^gpu=' -m1 "${f}" | cut -d= -f2- || true)
      session=$(grep -E '^session=' -m1 "${f}" | cut -d= -f2- || true)
      start=$(grep -E '^start=' -m1 "${f}" | cut -d= -f2- || true)
      end=$(grep -E '^end=' -m1 "${f}" | cut -d= -f2- || true)
      exit_code=$(grep -E '^exit_code=' -m1 "${f}" | cut -d= -f2- || true)
      echo "  - ${run_id} (${arch}/${size}/${mode}) gpu=${gpu} sess=${session} start=${start} end=${end} exit=${exit_code}"
    done
  done
fi

echo "Logs: ${TARGET_DIR}/logs/<run_id>.log"
echo "Manifest: ${TARGET_DIR}/manifest.tsv"

