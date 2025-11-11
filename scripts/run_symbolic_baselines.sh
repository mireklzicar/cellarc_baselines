#!/usr/bin/env bash
set -euo pipefail

# Execute all symbolic baselines over the full evaluation splits.

usage() {
  cat <<'EOF'
Usage: scripts/run_symbolic_baselines.sh [--per-task-dir DIR] [--splits split_a,split_b]

Options:
  --per-task-dir DIR   Directory where per-task accuracy JSON files should be written.
  --splits LIST        Comma-separated list of splits to evaluate (defaults to config file).
  -h, --help           Show this help text.
EOF
}

PER_TASK_DIR=""
SPLITS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --per-task-dir)
      PER_TASK_DIR="${2:-}"
      shift 2
      ;;
    --splits)
      SPLITS_OVERRIDE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/outputs/symbolic"
SUMMARY_CSV="${RESULTS_DIR}/symbolic_baselines_summary.csv"

mkdir -p "${RESULTS_DIR}"

HYDRA_SPLITS_ARG=""
if [[ -n "${SPLITS_OVERRIDE}" ]]; then
  IFS=',' read -r -a USER_SPLITS <<< "${SPLITS_OVERRIDE}"
  if [[ ${#USER_SPLITS[@]} -eq 0 ]]; then
    echo "No splits provided to --splits option." >&2
    exit 1
  fi
  SPLIT_LIST="["
  for split_name in "${USER_SPLITS[@]}"; do
    if [[ -z "${split_name}" ]]; then
      continue
    fi
    SPLIT_LIST+="${split_name},"
  done
  SPLIT_LIST="${SPLIT_LIST%,}"
  SPLIT_LIST+="]"
  HYDRA_SPLITS_ARG="splits=${SPLIT_LIST}"
fi

declare -a SYMBOLIC_BASELINES=(
  "copycat"
  "most_frequent"
  "de_bruijn"
  "random"
)

for baseline in "${SYMBOLIC_BASELINES[@]}"; do
  echo "=== Evaluating symbolic baseline: ${baseline} ==="
  results_json="${RESULTS_DIR}/${baseline}_results.json"
  declare -a HYDRA_OVERRIDES=()
  if [[ -n "${HYDRA_SPLITS_ARG}" ]]; then
    HYDRA_OVERRIDES+=("${HYDRA_SPLITS_ARG}")
  fi
  if [[ -n "${PER_TASK_DIR}" ]]; then
    HYDRA_OVERRIDES+=("eval.per_task_results_dir=${PER_TASK_DIR}")
  fi
  cmd=(
    python "${REPO_ROOT}/scripts/eval_symbolic.py"
    baseline.name="${baseline}"
    eval.progress_bar=true
    eval.max_episodes=null
    eval.results_json_path="${results_json}"
  )
  if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
    cmd+=("${HYDRA_OVERRIDES[@]}")
  fi
  "${cmd[@]}"
done

echo "=== Writing symbolic baseline summary to ${SUMMARY_CSV} ==="
python - "${SUMMARY_CSV}" "${RESULTS_DIR}" "${SYMBOLIC_BASELINES[@]}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
baselines = sys.argv[3:]

if not baselines:
    raise SystemExit("No baselines provided for summary generation.")

split_names = None
rows = []

for baseline in baselines:
    results_path = results_dir / f"{baseline}_results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing results JSON for baseline '{baseline}': {results_path}")

    data = json.loads(results_path.read_text())
    splits = data.get("splits", [])
    if not splits:
        raise SystemExit(f"No split metrics found in {results_path}")

    current_split_names = [entry["split"] for entry in splits]
    if split_names is None:
        split_names = current_split_names
    elif split_names != current_split_names:
        raise SystemExit(
            f"Split mismatch for baseline '{baseline}'. Expected {split_names}, got {current_split_names}"
        )

    token_metrics = [f"{entry.get('token_accuracy', 0.0):.6f}" for entry in splits]
    rows.append((baseline, token_metrics))

if split_names is None:
    raise SystemExit("Unable to determine split names for summary CSV.")

header = ["model"] + [f"{split}_token_accuracy" for split in split_names]
with summary_path.open("w", encoding="utf-8") as handle:
    handle.write(",".join(header) + "\n")
    for baseline, token_metrics in rows:
        handle.write(",".join([baseline, *token_metrics]) + "\n")

print(f"Wrote summary CSV to {summary_path}")
PY
