#!/bin/bash
#SBATCH --job-name=gl-dev-diag
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=logs/eval_internal_dev_%j.out
#SBATCH --error=logs/eval_internal_dev_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
: "${EVAL_DATA:?prepared internal dev JSONL required}"
: "${EVAL_CHECKPOINT:?trained checkpoint required}"
: "${EVAL_OUTPUT:?fresh diagnostic output directory required}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-configs/eval_protocol_retrospective.json}"
EVAL_SHARDS="${EVAL_SHARDS:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
EVAL_RECORD_INDEX="${EVAL_RECORD_INDEX:-}"
MAX_REQUEUES="${MAX_REQUEUES:-3}"
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"

selection_args=()
if [[ -n "$EVAL_RECORD_INDEX" ]]; then
  [[ "$EVAL_RECORD_INDEX" =~ ^[0-9]+$ ]] || {
    echo "ERROR: EVAL_RECORD_INDEX must be a non-negative integer"
    exit 2
  }
  selection_args=(--record-index "$EVAL_RECORD_INDEX")
else
  [[ "$EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: EVAL_SHARDS must be a positive integer"
    exit 2
  }
  [[ "$EVAL_SHARD_INDEX" =~ ^[0-9]+$ ]] || {
    echo "ERROR: EVAL_SHARD_INDEX must be a non-negative integer"
    exit 2
  }
  if (( EVAL_SHARD_INDEX >= EVAL_SHARDS )); then
    echo "ERROR: EVAL_SHARD_INDEX=$EVAL_SHARD_INDEX is outside EVAL_SHARDS=$EVAL_SHARDS"
    exit 2
  fi
  selection_args=(--shard-index "$EVAL_SHARD_INDEX" --shard-count "$EVAL_SHARDS")
fi
[[ "$MAX_REQUEUES" =~ ^[0-9]+$ ]] || {
  echo "ERROR: MAX_REQUEUES must be a non-negative integer"
  exit 2
}
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] || {
  echo "ERROR: SLURM_RESTART_COUNT must be a non-negative integer"
  exit 2
}
if (( RESTART_COUNT > MAX_REQUEUES )); then
  echo "ERROR: restart count $RESTART_COUNT exceeds MAX_REQUEUES=$MAX_REQUEUES"
  exit 2
fi

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export TMPDIR="${TMPDIR:-$HOME/work/.tmp}"
export HF_HOME="${GUARDLENS_HF_HOME:-$HOME/work/hf_models}"
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" logs

cd "$PROJECT_DIR"
for path in "$EVAL_DATA" "$EVAL_DATA.manifest.json" "$EVAL_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path"; exit 2; }
done
if [[ -e "$EVAL_OUTPUT" ]]; then
  echo "Diagnostic output exists; evaluator will verify its manifest identity: $EVAL_OUTPUT"
fi
[[ -f "$EVAL_PROTOCOL" ]] || { echo "ERROR: missing protocol: $EVAL_PROTOCOL"; exit 2; }

EXPECTED_BRANCH="naacl-causal-localization-redesign"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CODE_SHA=$(git rev-parse HEAD)
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected git branch $EXPECTED_BRANCH, got $CURRENT_BRANCH"
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked working-tree changes detected; commit or stash them before a scientific run"
  exit 2
fi

EVAL_DATA_SHA256=$(sha256sum "$EVAL_DATA" | awk '{print $1}')
echo "Code SHA: $CODE_SHA"
echo "Checkpoint: $EVAL_CHECKPOINT"
echo "Internal dev: $EVAL_DATA"
echo "Output: $EVAL_OUTPUT"
echo "Protocol: $EVAL_PROTOCOL"
echo "External guards: DISABLED"
echo "Held-out test: NOT ACCESSED"
if [[ -n "$EVAL_RECORD_INDEX" ]]; then
  echo "Diagnostic record gate: zero-based index $EVAL_RECORD_INDEX"
else
  echo "Diagnostic shard: $EVAL_SHARD_INDEX/$EVAL_SHARDS"
fi
echo "Slurm restart: $RESTART_COUNT/$MAX_REQUEUES"

count_completed_work() {
  python - "$EVAL_OUTPUT/shards" "$EVAL_SHARD_INDEX" "$EVAL_SHARDS" "$EVAL_RECORD_INDEX" <<'PY'
import re
import sys
from pathlib import Path

directory = Path(sys.argv[1])
shard_index = int(sys.argv[2])
shard_count = int(sys.argv[3])
record_index = sys.argv[4]
completed = []
if directory.is_dir():
    for path in directory.iterdir():
        match = re.fullmatch(r"record-(\d{6})\.json", path.name)
        if match:
            completed.append(int(match.group(1)))
if record_index:
    print(int(int(record_index) in completed))
else:
    print(sum(index % shard_count == shard_index for index in completed))
PY
}

COMPLETED_AT_START=$(count_completed_work)
echo "Completed records for this allocation at start: $COMPLETED_AT_START"

requeue_requested=0
diagnostic_pid=""
request_requeue() {
  requeue_requested=1
  echo "Received Slurm pre-timeout signal; preserving completed diagnostic shards"
  if [[ -n "$diagnostic_pid" ]]; then
    kill -TERM "$diagnostic_pid" 2>/dev/null || true
  fi
}
trap request_requeue USR1

set +e
python -u -m eval_platform diagnose-dev \
  --data "$EVAL_DATA" \
  --data-sha256 "$EVAL_DATA_SHA256" \
  --checkpoint "$EVAL_CHECKPOINT" \
  --output "$EVAL_OUTPUT" \
  --protocol "$EVAL_PROTOCOL" \
  --lexicon configs/eval_surface_lexicon.json \
  --device cuda \
  "${selection_args[@]}" &
diagnostic_pid=$!
wait "$diagnostic_pid"
diagnostic_status=$?
if (( requeue_requested == 1 )); then
  wait "$diagnostic_pid" 2>/dev/null || true
fi
set -e
trap - USR1

if (( requeue_requested == 1 )); then
  COMPLETED_NOW=$(count_completed_work)
  echo "Completed records for this allocation at timeout: $COMPLETED_NOW"
  if (( COMPLETED_NOW <= COMPLETED_AT_START )); then
    echo "ERROR: no completed-record progress in this allocation; refusing automatic requeue"
    exit 2
  fi
  if (( RESTART_COUNT >= MAX_REQUEUES )); then
    echo "ERROR: diagnostic needs another allocation but MAX_REQUEUES=$MAX_REQUEUES is exhausted"
    exit 2
  fi
  [[ -f "$EVAL_OUTPUT/manifest.json" ]] || {
    echo "ERROR: cannot requeue because the diagnostic manifest was not created"
    exit 2
  }
  echo "Requeuing Slurm job $SLURM_JOB_ID after verified record progress"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if (( diagnostic_status != 0 )); then
  echo "ERROR: diagnostic process exited with status $diagnostic_status"
  exit "$diagnostic_status"
fi
