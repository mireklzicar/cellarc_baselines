#!/usr/bin/env bash
# Wrapper to run a single training command and record status transitions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

PROJECT_NAME=""
RUN_NAME="${RUN_ID}"
declare -a CMD_ARGS
IFS_BACKUP="${IFS}"
IFS=' ' read -r -a CMD_ARGS <<< "${CMD_STR}"
IFS="${IFS_BACKUP}"
for arg in "${CMD_ARGS[@]}"; do
  case "${arg}" in
    logging.wandb.project=*)
      PROJECT_NAME="${arg#logging.wandb.project=}"
      ;;
    logging.wandb.name=*)
      RUN_NAME="${arg#logging.wandb.name=}"
      ;;
  esac
done
OUTPUT_DIR=""
if [[ -n "${PROJECT_NAME}" && -n "${RUN_NAME}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/outputs/${PROJECT_NAME}/${RUN_NAME}"
fi

# Execute the command in a login shell to load user envs consistently
set +e
bash -lc "${CMD_STR}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

END_TS="$(date -Is)"

META_SOURCE=""
if [[ -f "${INPROG_META}" ]]; then
  META_SOURCE="${INPROG_META}"
elif [[ -f "${PLANNED_META}" ]]; then
  META_SOURCE="${PLANNED_META}"
fi
SIZE_NAME=""
MODE_NAME=""
if [[ -n "${META_SOURCE}" ]]; then
  SIZE_NAME="$(awk -F= '$1=="size"{print $2; exit}' "${META_SOURCE}")"
  MODE_NAME="$(awk -F= '$1=="mode"{print $2; exit}' "${META_SOURCE}")"
fi

FAILURE_REASON=""
if [[ ${EXIT_CODE} -ne 0 ]]; then
  if grep -qiE "CUDA (error: )?out of memory" "${LOG_FILE}"; then
    FAILURE_REASON="cuda_oom"
    echo "[CUDA OOM] Detected CUDA out-of-memory for ${RUN_ID}." | tee -a "${LOG_FILE}"
    if [[ -z "${PROJECT_NAME}" || -z "${RUN_NAME}" ]]; then
      if [[ -n "${MODE_NAME}" && -n "${SIZE_NAME}" ]]; then
        PROJECT_NAME="cellarc100k_50e_${MODE_NAME}_${SIZE_NAME}"
        if [[ -z "${OUTPUT_DIR}" ]]; then
          OUTPUT_DIR="${REPO_ROOT}/outputs/${PROJECT_NAME}/${RUN_ID}"
        fi
      fi
    fi
    if [[ -n "${OUTPUT_DIR}" ]]; then
      if [[ -d "${OUTPUT_DIR}" ]]; then
        rm -rf -- "${OUTPUT_DIR}"
        echo "[CUDA OOM] Removed incomplete output directory ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
      else
        echo "[CUDA OOM] Output directory ${OUTPUT_DIR} not found; nothing to remove." | tee -a "${LOG_FILE}"
      fi
    else
      echo "[CUDA OOM] Unable to determine output directory for ${RUN_ID}." | tee -a "${LOG_FILE}"
    fi
  fi
fi

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

if [[ -n "${FAILURE_REASON}" ]]; then
  printf "failure_reason=%s\n" "${FAILURE_REASON}" >> "${DEST_META}"
fi

echo "[END ${END_TS}] ${RUN_ID} (GPU=${GPU}) exit=${EXIT_CODE}" | tee -a "${LOG_FILE}"

exit "${EXIT_CODE}"
