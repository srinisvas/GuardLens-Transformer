#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

BASE_OUTPUT="${BASE_OUTPUT:-$HOME/work/results/guardlens_naacl_redesign/diagnostic-matrices}"
MATRIX_ID="${MATRIX_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MATRIX_ROOT="${MATRIX_ROOT:-$BASE_OUTPUT/$MATRIX_ID}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
DIAGNOSTIC_TIME="${DIAGNOSTIC_TIME:-12:00:00}"
PREPARED_DEV="$MATRIX_ROOT/shared/internal-dev.jsonl"
SHARED_PREFLIGHT_DIR="$MATRIX_ROOT/shared/preflight"

if [[ -e "$MATRIX_ROOT" ]]; then
  echo "ERROR: matrix root already exists: $MATRIX_ROOT"
  exit 2
fi

names=(
  mean_frozen_aux
  attention_frozen_aux
  attention_top4_aux
  attention_top4_primary
)
train_variants=(
  primary_plus_auxiliary
  primary_plus_auxiliary
  primary_plus_auxiliary
  primary
)
poolings=(mean attention attention attention)
trainable_layers=(0 0 4 4)
turn_microbatches=(8 8 1 1)

preflight_job=$(sbatch --parsable \
  --export="ALL,PREPARED_DEV=$PREPARED_DEV,SHARED_PREFLIGHT_DIR=$SHARED_PREFLIGHT_DIR,INPUT_VIEW=pre_response,LENGTH_AUC_CEILING=0.650" \
  preflight_naacl_matrix.slurm)
preflight_job="${preflight_job%%;*}"

smoke_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  job=$(sbatch --parsable \
    --dependency="afterok:$preflight_job" \
    --export="ALL,TRAIN_VARIANT=${train_variants[$index]},TURN_POOLING=${poolings[$index]},BACKBONE_TRAINABLE_LAYERS=${trainable_layers[$index]},BACKBONE_TURN_MICROBATCH=${turn_microbatches[$index]},BACKBONE_LR=2e-5,INPUT_VIEW=pre_response,SHARED_PREFLIGHT_DIR=$SHARED_PREFLIGHT_DIR,SMOKE_OUTPUT_DIR=$MATRIX_ROOT/smoke/$name" \
    smoke_naacl_window.slurm)
  job="${job%%;*}"
  smoke_jobs+=("$job")
  echo "smoke $name: $job"
done

smoke_dependency=$(IFS=:; echo "${smoke_jobs[*]}")
train_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  job=$(sbatch --parsable \
    --time="$TRAIN_TIME" \
    --dependency="afterok:$smoke_dependency" \
    --export="ALL,TRAIN_VARIANT=${train_variants[$index]},TURN_POOLING=${poolings[$index]},BACKBONE_TRAINABLE_LAYERS=${trainable_layers[$index]},BACKBONE_TURN_MICROBATCH=${turn_microbatches[$index]},BACKBONE_LR=2e-5,INPUT_VIEW=pre_response,SHARED_PREFLIGHT_DIR=$SHARED_PREFLIGHT_DIR,RUN_ROOT=$MATRIX_ROOT/training/$name" \
    train_naacl.slurm)
  job="${job%%;*}"
  train_jobs+=("$job")
  echo "train $name: $job"
done

train_dependency=$(IFS=:; echo "${train_jobs[*]}")
diagnostic_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  job=$(sbatch --parsable \
    --time="$DIAGNOSTIC_TIME" \
    --dependency="afterok:$preflight_job:$train_dependency" \
    --export="ALL,EVAL_DATA=$PREPARED_DEV,EVAL_CHECKPOINT=$MATRIX_ROOT/training/$name/checkpoints/best.pt,EVAL_OUTPUT=$MATRIX_ROOT/diagnostics/$name" \
    eval_internal_dev.slurm)
  job="${job%%;*}"
  diagnostic_jobs+=("$job")
  echo "diagnose $name: $job"
done

diagnostic_dependency=$(IFS=:; echo "${diagnostic_jobs[*]}")
compare_job=$(sbatch --parsable \
  --dependency="afterok:$diagnostic_dependency" \
  --export="ALL,MATRIX_ROOT=$MATRIX_ROOT" \
  compare_internal_dev_matrix.slurm)
compare_job="${compare_job%%;*}"

echo "shared CPU preflight: $preflight_job"
echo "compare matrix: $compare_job"
echo "matrix root: $MATRIX_ROOT"
echo "held-out test accessed: NO"
echo "external evaluation submitted: NO"
