#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

MATRIX_ROOT="${MATRIX_ROOT:-${1:-}}"
: "${MATRIX_ROOT:?pass the completed four-candidate matrix root as argument 1 or MATRIX_ROOT}"
[[ -d "$MATRIX_ROOT" ]] || { echo "ERROR: matrix root missing: $MATRIX_ROOT"; exit 2; }

CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
FREEZE_DIR="${FREEZE_DIR:-$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a}"
REPORT_PATH="${REPORT_PATH:-$FREEZE_DIR/data_prep_freeze_report.json}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
DIAGNOSTIC_TIME="${DIAGNOSTIC_TIME:-24:00:00}"
DIAGNOSTIC_SMOKE_TIME="${DIAGNOSTIC_SMOKE_TIME:-01:00:00}"
DIAGNOSTIC_SMOKE_INDEX="${DIAGNOSTIC_SMOKE_INDEX:-136}"
EVAL_SHARDS="${EVAL_SHARDS:-4}"
REQUEUE_LIMIT="${MAX_REQUEUES:-3}"
[[ "$EVAL_SHARDS" =~ ^[1-4]$ ]] || { echo "ERROR: EVAL_SHARDS must be 1 through 4"; exit 2; }
[[ "$REQUEUE_LIMIT" =~ ^[0-9]+$ ]] || { echo "ERROR: MAX_REQUEUES must be non-negative"; exit 2; }
[[ "$DIAGNOSTIC_SMOKE_INDEX" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid diagnostic gate index"; exit 2; }

unset MAX_REQUEUES EVAL_RECORD_INDEX EVAL_SHARD_INDEX \
  BACKBONE_TRAINABLE_LAYERS BACKBONE_TURN_MICROBATCH ARCHITECTURE_MODE \
  ATTRIBUTION_FUSION INPUT_VIEW MODEL TURN_POOLING \
  RUN_ROOT PRECHECK_DIR OUTPUT TRAIN_RESUME \
  EVAL_DATA EVAL_CHECKPOINT EVAL_OUTPUT EVAL_PROTOCOL SMOKE_OUTPUT_DIR

BACKBONE="${BACKBONE:-answerdotai/ModernBERT-large}"
BACKBONE_REVISION="${BACKBONE_REVISION:-45bb4654a4d5aaff24dd11d4781fa46d39bf8c13}"
MAX_TURNS="${MAX_TURNS:-64}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-8}"
HEAD_LR="${HEAD_LR:-2e-4}"
BACKBONE_LR="${BACKBONE_LR:-2e-5}"
EPOCHS="${EPOCHS:-20}"
LOCALIZATION_RAMP_EPOCHS="${LOCALIZATION_RAMP_EPOCHS:-5}"
LENGTH_AUC_CEILING="${LENGTH_AUC_CEILING:-0.650}"
TRAIN_VARIANT="primary_plus_auxiliary"
TRAIN_PATH="$FREEZE_DIR/splits_primary_plus_train_auxiliary/train.jsonl"
DEV_PATH="$FREEZE_DIR/splits_primary_plus_train_auxiliary/dev.jsonl"
PREPARED_DEV="$MATRIX_ROOT/shared/internal-dev.jsonl"
CONTROL_PREFLIGHT_DIR="$MATRIX_ROOT/shared/preflight-top4-sibling-controls"
EXTENDED_COMPARISON="$MATRIX_ROOT/internal_dev_comparison_six_candidates.json"
[[ -f "$PREPARED_DEV" && -f "$PREPARED_DEV.manifest.json" ]] || {
  echo "ERROR: original prepared internal dev is missing"
  exit 2
}
[[ ! -e "$EXTENDED_COMPARISON" ]] || {
  echo "ERROR: six-candidate comparison already exists: $EXTENDED_COMPARISON"
  exit 2
}

base_names=(
  hierarchical_sibling_retro_frozen
  cross_token_sibling_retro_frozen
  cross_token_gated_retro_frozen
  cross_token_gated_retro_top4
)
for name in "${base_names[@]}"; do
  python - "$MATRIX_ROOT/diagnostics/$name/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("coverage") != {"failures": [], "scored": 364, "total": 364}:
    raise SystemExit(f"ERROR: incomplete base report: {sys.argv[1]}")
if not report.get("development_only") or report.get("held_out_test_accessed") is not False:
    raise SystemExit(f"ERROR: base report is not development-only: {sys.argv[1]}")
PY
done

names=(hierarchical_sibling_retro_top4 cross_token_sibling_retro_top4)
architectures=(hierarchical_turn cross_token)
protocol="configs/eval_protocol_retrospective.json"
common_export="CONDA_ENV=$CONDA_ENV,FREEZE_DIR=$FREEZE_DIR,REPORT_PATH=$REPORT_PATH,BACKBONE=$BACKBONE,BACKBONE_REVISION=$BACKBONE_REVISION,MAX_TURNS=$MAX_TURNS,MAX_TOKENS=$MAX_TOKENS,LENGTH_AUC_CEILING=$LENGTH_AUC_CEILING"
train_export="$common_export,TRAIN_VARIANT=$TRAIN_VARIANT,TRAIN_PATH=$TRAIN_PATH,DEV_PATH=$DEV_PATH,MODEL=guardlens,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUMULATION=$GRAD_ACCUMULATION,HEAD_LR=$HEAD_LR,BACKBONE_LR=$BACKBONE_LR,EPOCHS=$EPOCHS,LOCALIZATION_RAMP_EPOCHS=$LOCALIZATION_RAMP_EPOCHS,TURN_POOLING=attention,MAX_REQUEUES=$REQUEUE_LIMIT"

submitted_jobs=()
cancel_submitted_jobs() {
  status=$?
  trap - ERR INT TERM
  if (( ${#submitted_jobs[@]} > 0 )); then
    echo "Submission failed; cancelling jobs from this invocation: ${submitted_jobs[*]}" >&2
    scancel "${submitted_jobs[@]}" || true
  fi
  exit "$status"
}
trap cancel_submitted_jobs ERR INT TERM

preflight_dependency=()
if [[ -f "$CONTROL_PREFLIGHT_DIR/$TRAIN_VARIANT/retrospective/marker.json" ]]; then
  echo "control preflight already complete: $CONTROL_PREFLIGHT_DIR"
else
  [[ ! -e "$CONTROL_PREFLIGHT_DIR" ]] || {
    echo "ERROR: incomplete control preflight exists; archive it before resubmitting"
    exit 2
  }
  preflight_job=$(sbatch --parsable \
    --export="ALL,$common_export,PREPARED_DEV=$PREPARED_DEV,SHARED_PREFLIGHT_DIR=$CONTROL_PREFLIGHT_DIR" \
    preflight_naacl_top4_sibling_controls.slurm)
  preflight_job="${preflight_job%%;*}"
  submitted_jobs+=("$preflight_job")
  preflight_dependency=(--dependency="afterok:$preflight_job")
  echo "control preflight: $preflight_job"
fi

finalize_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  train_root="$MATRIX_ROOT/training/$name"
  diagnostic_output="$MATRIX_ROOT/diagnostics/$name"
  report="$diagnostic_output/report.json"
  if [[ -f "$report" ]]; then
    echo "complete $name: report already exists"
    continue
  fi

  # Even when a checkpoint already exists, do not start diagnostic work until
  # the current-code control preflight has passed.
  training_dependency=("${preflight_dependency[@]}")
  if [[ -f "$train_root/checkpoints/training_summary.json" && -f "$train_root/checkpoints/best.pt" ]]; then
    echo "training complete $name: reusing best.pt"
  else
    resume=0
    smoke_dependency=("${preflight_dependency[@]}")
    if [[ -f "$train_root/checkpoints/last.pt" ]]; then
      resume=1
      echo "resumable training checkpoint found $name"
    else
      [[ ! -e "$train_root" ]] || {
        echo "ERROR: incomplete non-resumable training directory: $train_root"
        exit 2
      }
      smoke_job=$(sbatch --parsable \
        --kill-on-invalid-dep=yes \
        "${preflight_dependency[@]}" \
        --export="ALL,$train_export,BACKBONE_TRAINABLE_LAYERS=4,BACKBONE_TURN_MICROBATCH=1,ARCHITECTURE_MODE=${architectures[$index]},ATTRIBUTION_FUSION=0,INPUT_VIEW=retrospective,SHARED_PREFLIGHT_DIR=$CONTROL_PREFLIGHT_DIR,SMOKE_OUTPUT_DIR=$MATRIX_ROOT/smoke/$name" \
        smoke_naacl_window.slurm)
      smoke_job="${smoke_job%%;*}"
      submitted_jobs+=("$smoke_job")
      smoke_dependency=(--dependency="afterok:$smoke_job")
      echo "smoke $name: $smoke_job"
    fi

    train_job=$(sbatch --parsable \
      --time="$TRAIN_TIME" \
      --kill-on-invalid-dep=yes \
      "${smoke_dependency[@]}" \
      --export="ALL,$train_export,BACKBONE_TRAINABLE_LAYERS=4,BACKBONE_TURN_MICROBATCH=1,ARCHITECTURE_MODE=${architectures[$index]},ATTRIBUTION_FUSION=0,INPUT_VIEW=retrospective,SHARED_PREFLIGHT_DIR=$CONTROL_PREFLIGHT_DIR,RUN_ROOT=$train_root,PRECHECK_DIR=$train_root/preflight,OUTPUT=$train_root/checkpoints,TRAIN_RESUME=$resume" \
      train_naacl.slurm)
    train_job="${train_job%%;*}"
    submitted_jobs+=("$train_job")
    training_dependency=(--dependency="afterok:$train_job")
    echo "train $name: $train_job"
  fi

  shard_count=0
  if [[ -d "$diagnostic_output/shards" ]]; then
    shard_count=$(find "$diagnostic_output/shards" -maxdepth 1 -type f -name 'record-*.json' | wc -l)
  fi
  if [[ -f "$diagnostic_output/manifest.json" && "$shard_count" -eq 364 ]]; then
    diagnostic_dependency=("${training_dependency[@]}")
    echo "diagnostic shards complete $name: finalizing directly"
  else
    gate_job=$(sbatch --parsable \
      --time="$DIAGNOSTIC_SMOKE_TIME" \
      --kill-on-invalid-dep=yes \
      --output="logs/eval_internal_dev_smoke_%j.out" \
      --error="logs/eval_internal_dev_smoke_%j.err" \
      "${training_dependency[@]}" \
      --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_CHECKPOINT=$train_root/checkpoints/best.pt,EVAL_OUTPUT=$diagnostic_output,EVAL_PROTOCOL=$protocol,EVAL_RECORD_INDEX=$DIAGNOSTIC_SMOKE_INDEX,MAX_REQUEUES=0" \
      eval_internal_dev.slurm)
    gate_job="${gate_job%%;*}"
    submitted_jobs+=("$gate_job")

    array_job=$(sbatch --parsable \
      --time="$DIAGNOSTIC_TIME" \
      --kill-on-invalid-dep=yes \
      --output="logs/eval_internal_dev_%A_%a.out" \
      --error="logs/eval_internal_dev_%A_%a.err" \
      --array="0-$((EVAL_SHARDS - 1))%$EVAL_SHARDS" \
      --dependency="afterok:$gate_job" \
      --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_CHECKPOINT=$train_root/checkpoints/best.pt,EVAL_OUTPUT=$diagnostic_output,EVAL_PROTOCOL=$protocol,EVAL_SHARDS=$EVAL_SHARDS,MAX_REQUEUES=$REQUEUE_LIMIT" \
      eval_internal_dev.slurm)
    array_job="${array_job%%;*}"
    submitted_jobs+=("$array_job")
    diagnostic_dependency=(--dependency="afterok:$array_job")
    echo "diagnostic gate/array $name: $gate_job/$array_job"
  fi

  finalize_job=$(sbatch --parsable \
    --kill-on-invalid-dep=yes \
    "${diagnostic_dependency[@]}" \
    --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_OUTPUT=$diagnostic_output,EVAL_SHARDS=$EVAL_SHARDS" \
    eval_internal_dev_finalize.slurm)
  finalize_job="${finalize_job%%;*}"
  finalize_jobs+=("$finalize_job")
  submitted_jobs+=("$finalize_job")
  echo "finalize $name: $finalize_job"
done

compare_dependency=()
if (( ${#finalize_jobs[@]} > 0 )); then
  dependency_ids=$(IFS=:; echo "${finalize_jobs[*]}")
  compare_dependency=(--dependency="afterok:$dependency_ids")
fi
compare_job=$(sbatch --parsable \
  --kill-on-invalid-dep=yes \
  "${compare_dependency[@]}" \
  --export="ALL,CONDA_ENV=$CONDA_ENV,MATRIX_ROOT=$MATRIX_ROOT" \
  compare_internal_dev_extended.slurm)
compare_job="${compare_job%%;*}"
submitted_jobs+=("$compare_job")

trap - ERR INT TERM
echo "extended comparison: $compare_job"
echo "matrix root: $MATRIX_ROOT"
echo "new candidates: ${names[*]}"
echo "existing four candidates rerun: NO"
echo "internal diagnostic shards per new candidate: $EVAL_SHARDS"
echo "diagnostic record gate: index $DIAGNOSTIC_SMOKE_INDEX with limit $DIAGNOSTIC_SMOKE_TIME"
echo "held-out test accessed: NO"
echo "external evaluation submitted: NO"
