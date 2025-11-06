#!/usr/bin/env bash
# Wrapper to run a single training command and record status transitions.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: tmux_run_wrapper.sh <run_id> <gpu>" >&2
  exit 2
fi

RUN_ID="$1"
GPU="$2"

STATUS_DIR="${RUNS_STATUS_DIR:-}"
if [[ -z "${STATUS_DIR}" ]]; then
  echo "RUNS_STATUS_DIR is not set in environment." >&2
  exit 2
fi

PLANNED_META="${STATUS_DIR}/planned/${RUN_ID}.meta"
PLANNED_CMD_FILE="${STATUS_DIR}/planned/${RUN_ID}.cmd"
INPROG_META="${STATUS_DIR}/in_progress/${RUN_ID}.meta"
INPROG_CMD_FILE="${STATUS_DIR}/in_progress/${RUN_ID}.cmd"

CMD_SOURCE=""
if [[ -f "${INPROG_CMD_FILE}" ]]; then
  CMD_SOURCE="${INPROG_CMD_FILE}"
elif [[ -f "${PLANNED_CMD_FILE}" ]]; then
  CMD_SOURCE="${PLANNED_CMD_FILE}"
else
  echo "Command file missing for run_id=${RUN_ID}: ${INPROG_CMD_FILE} or ${PLANNED_CMD_FILE}" >&2
  exit 3
fi

mkdir -p "${STATUS_DIR}"/{in_progress,succeeded,failed,logs}

START_TS="$(date -Is)"
CMD_STR="$(<"${CMD_SOURCE}")"

# Mark in-progress (merge planned meta if available)
{
  if [[ -f "${PLANNED_META}" ]]; then
    cat "${PLANNED_META}"
  elif [[ -f "${INPROG_META}" ]]; then
    cat "${INPROG_META}"
  fi
  printf "gpu=%s\nsession=%s\nstart=%s\ncmd=%s\n" \
    "${GPU}" "${TMUX_SESSION_NAME:-unknown}" "${START_TS}" "${CMD_STR}"
} > "${INPROG_META}"

LOG_FILE="${STATUS_DIR}/logs/${RUN_ID}.log"
echo "[START ${START_TS}] ${RUN_ID} (GPU=${GPU})" | tee -a "${LOG_FILE}"

# Execute the command in a login shell to load user envs consistently
set +e
bash -lc "${CMD_STR}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

END_TS="$(date -Is)"
rm -f "${INPROG_META}" "${INPROG_CMD_FILE}"

if [[ ${EXIT_CODE} -eq 0 ]]; then
  DEST_DIR="succeeded"
else
  DEST_DIR="failed"
fi

DEST_META="${STATUS_DIR}/${DEST_DIR}/${RUN_ID}.meta"
{
  # Keep baseline fields from planned if present, but override gpu/session and add timings
  if [[ -f "${PLANNED_META}" ]]; then
    # Filter out potentially stale gpu/session
    grep -v -E '^(gpu|session)=' "${PLANNED_META}" || true
  fi
  printf "gpu=%s\nsession=%s\nstart=%s\nend=%s\nexit_code=%s\ncmd=%s\n" \
    "${GPU}" "${TMUX_SESSION_NAME:-unknown}" "${START_TS}" "${END_TS}" "${EXIT_CODE}" "${CMD_STR}"
} > "${DEST_META}"

echo "[END ${END_TS}] ${RUN_ID} (GPU=${GPU}) exit=${EXIT_CODE}" | tee -a "${LOG_FILE}"

exit "${EXIT_CODE}"
