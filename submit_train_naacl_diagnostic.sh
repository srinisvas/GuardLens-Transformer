#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
FREEZE_DIR="${FREEZE_DIR:-$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a}"
REPORT_PATH="${REPORT_PATH:-$FREEZE_DIR/data_prep_freeze_report.json}"
BASE_OUTPUT="${BASE_OUTPUT:-$HOME/work/results/guardlens_naacl_redesign/diagnostic-matrices}"
MATRIX_ID="${MATRIX_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MATRIX_ROOT="${MATRIX_ROOT:-$BASE_OUTPUT/$MATRIX_ID}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
DIAGNOSTIC_TIME="${DIAGNOSTIC_TIME:-24:00:00}"
DIAGNOSTIC_SMOKE_TIME="${DIAGNOSTIC_SMOKE_TIME:-01:00:00}"
DIAGNOSTIC_SMOKE_INDEX="${DIAGNOSTIC_SMOKE_INDEX:-136}"
EVAL_SHARDS="${EVAL_SHARDS:-4}"
MAX_REQUEUES="${MAX_REQUEUES:-3}"
[[ "$EVAL_SHARDS" =~ ^[1-4]$ ]] || {
  echo "ERROR: EVAL_SHARDS must be an integer from 1 to 4"
  exit 2
}
[[ "$MAX_REQUEUES" =~ ^[0-9]+$ ]] || {
  echo "ERROR: MAX_REQUEUES must be a non-negative integer"
  exit 2
}
[[ "$DIAGNOSTIC_SMOKE_INDEX" =~ ^[0-9]+$ ]] || {
  echo "ERROR: DIAGNOSTIC_SMOKE_INDEX must be a non-negative integer"
  exit 2
}
# Do not let an interactive shell's single-record/shard override leak into the
# jobs below. Each selection is passed explicitly.
unset EVAL_RECORD_INDEX EVAL_SHARD_INDEX
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
SHARED_PREFLIGHT_DIR="$MATRIX_ROOT/shared/preflight"

# Atomically reserve the complete matrix namespace before submitting any work.
mkdir -p "$BASE_OUTPUT"
if ! mkdir "$MATRIX_ROOT"; then
  echo "ERROR: matrix root already exists: $MATRIX_ROOT"
  exit 2
fi

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

names=(
  hierarchical_sibling_retro_frozen
  cross_token_sibling_retro_frozen
  cross_token_gated_retro_frozen
  cross_token_gated_retro_top4
)
architectures=(
  hierarchical_turn
  cross_token
  cross_token
  cross_token
)
fusions=(0 0 1 1)
views=(retrospective retrospective retrospective retrospective)
trainable_layers=(0 0 0 4)
backbone_microbatches=(8 8 8 1)
protocols=(
  configs/eval_protocol_retrospective.json
  configs/eval_protocol_retrospective.json
  configs/eval_protocol_retrospective.json
  configs/eval_protocol_retrospective.json
)

common_export="CONDA_ENV=$CONDA_ENV,FREEZE_DIR=$FREEZE_DIR,REPORT_PATH=$REPORT_PATH,BACKBONE=$BACKBONE,BACKBONE_REVISION=$BACKBONE_REVISION,MAX_TURNS=$MAX_TURNS,MAX_TOKENS=$MAX_TOKENS,LENGTH_AUC_CEILING=$LENGTH_AUC_CEILING"
train_export="$common_export,TRAIN_VARIANT=$TRAIN_VARIANT,TRAIN_PATH=$TRAIN_PATH,DEV_PATH=$DEV_PATH,MODEL=guardlens,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUMULATION=$GRAD_ACCUMULATION,HEAD_LR=$HEAD_LR,BACKBONE_LR=$BACKBONE_LR,EPOCHS=$EPOCHS,LOCALIZATION_RAMP_EPOCHS=$LOCALIZATION_RAMP_EPOCHS,TURN_POOLING=attention,MAX_REQUEUES=$MAX_REQUEUES"

preflight_job=$(sbatch --parsable \
  --export="ALL,$common_export,PREPARED_DEV=$PREPARED_DEV,SHARED_PREFLIGHT_DIR=$SHARED_PREFLIGHT_DIR" \
  preflight_naacl_matrix.slurm)
preflight_job="${preflight_job%%;*}"
submitted_jobs+=("$preflight_job")

smoke_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  job=$(sbatch --parsable \
    --kill-on-invalid-dep=yes \
    --dependency="afterok:$preflight_job" \
    --export="ALL,$train_export,BACKBONE_TRAINABLE_LAYERS=${trainable_layers[$index]},BACKBONE_TURN_MICROBATCH=${backbone_microbatches[$index]},ARCHITECTURE_MODE=${architectures[$index]},ATTRIBUTION_FUSION=${fusions[$index]},INPUT_VIEW=${views[$index]},SHARED_PREFLIGHT_DIR=$SHARED_PREFLIGHT_DIR,SMOKE_OUTPUT_DIR=$MATRIX_ROOT/smoke/$name" \
    smoke_naacl_window.slurm)
  job="${job%%;*}"
  smoke_jobs+=("$job")
  submitted_jobs+=("$job")
  echo "smoke $name: $job"
done

# All four exact-shape smokes must pass before spending four full GPU jobs.
smoke_dependency=$(IFS=:; echo "${smoke_jobs[*]}")
train_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  job=$(sbatch --parsable \
    --time="$TRAIN_TIME" \
    --kill-on-invalid-dep=yes \
    --dependency="afterok:$smoke_dependency" \
    --export="ALL,$train_export,BACKBONE_TRAINABLE_LAYERS=${trainable_layers[$index]},BACKBONE_TURN_MICROBATCH=${backbone_microbatches[$index]},ARCHITECTURE_MODE=${architectures[$index]},ATTRIBUTION_FUSION=${fusions[$index]},INPUT_VIEW=${views[$index]},SHARED_PREFLIGHT_DIR=$SHARED_PREFLIGHT_DIR,RUN_ROOT=$MATRIX_ROOT/training/$name,PRECHECK_DIR=$MATRIX_ROOT/training/$name/preflight,OUTPUT=$MATRIX_ROOT/training/$name/checkpoints,TRAIN_RESUME=0" \
    train_naacl.slurm)
  job="${job%%;*}"
  train_jobs+=("$job")
  submitted_jobs+=("$job")
  echo "train $name: $job"
done

finalize_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  diagnostic_output="$MATRIX_ROOT/diagnostics/$name"
  diagnostic_smoke_job=$(sbatch --parsable \
    --time="$DIAGNOSTIC_SMOKE_TIME" \
    --kill-on-invalid-dep=yes \
    --output="logs/eval_internal_dev_smoke_%j.out" \
    --error="logs/eval_internal_dev_smoke_%j.err" \
    --dependency="afterok:${train_jobs[$index]}" \
    --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_CHECKPOINT=$MATRIX_ROOT/training/$name/checkpoints/best.pt,EVAL_OUTPUT=$diagnostic_output,EVAL_PROTOCOL=${protocols[$index]},EVAL_RECORD_INDEX=$DIAGNOSTIC_SMOKE_INDEX,MAX_REQUEUES=0" \
    eval_internal_dev.slurm)
  diagnostic_smoke_job="${diagnostic_smoke_job%%;*}"
  submitted_jobs+=("$diagnostic_smoke_job")

  diagnostic_job=$(sbatch --parsable \
    --time="$DIAGNOSTIC_TIME" \
    --kill-on-invalid-dep=yes \
    --output="logs/eval_internal_dev_%A_%a.out" \
    --error="logs/eval_internal_dev_%A_%a.err" \
    --array="0-$((EVAL_SHARDS - 1))%$EVAL_SHARDS" \
    --dependency="afterok:$diagnostic_smoke_job" \
    --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_CHECKPOINT=$MATRIX_ROOT/training/$name/checkpoints/best.pt,EVAL_OUTPUT=$diagnostic_output,EVAL_PROTOCOL=${protocols[$index]},EVAL_SHARDS=$EVAL_SHARDS,MAX_REQUEUES=$MAX_REQUEUES" \
    eval_internal_dev.slurm)
  diagnostic_job="${diagnostic_job%%;*}"
  submitted_jobs+=("$diagnostic_job")

  finalize_job=$(sbatch --parsable \
    --kill-on-invalid-dep=yes \
    --dependency="afterok:$diagnostic_job" \
    --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_OUTPUT=$diagnostic_output,EVAL_SHARDS=$EVAL_SHARDS" \
    eval_internal_dev_finalize.slurm)
  finalize_job="${finalize_job%%;*}"
  finalize_jobs+=("$finalize_job")
  submitted_jobs+=("$finalize_job")
  echo "diagnostic gate/array/finalize $name: $diagnostic_smoke_job/$diagnostic_job/$finalize_job"
done

finalize_dependency=$(IFS=:; echo "${finalize_jobs[*]}")
compare_job=$(sbatch --parsable \
  --kill-on-invalid-dep=yes \
  --dependency="afterok:$finalize_dependency" \
  --export="ALL,CONDA_ENV=$CONDA_ENV,MATRIX_ROOT=$MATRIX_ROOT" \
  compare_internal_dev_matrix.slurm)
compare_job="${compare_job%%;*}"
submitted_jobs+=("$compare_job")

trap - ERR INT TERM
echo "shared CPU preflight: $preflight_job"
echo "compare matrix: $compare_job"
echo "matrix root: $MATRIX_ROOT"
echo "internal diagnostic shards per candidate: $EVAL_SHARDS"
echo "diagnostic record gate: index $DIAGNOSTIC_SMOKE_INDEX with limit $DIAGNOSTIC_SMOKE_TIME"
echo "automatic requeues per training/diagnostic job: $MAX_REQUEUES"
echo "held-out test accessed: NO"
echo "external evaluation submitted: NO"
