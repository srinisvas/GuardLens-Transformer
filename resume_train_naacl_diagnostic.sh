#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

MATRIX_ROOT="${MATRIX_ROOT:-${1:-}}"
: "${MATRIX_ROOT:?pass the existing matrix root as argument 1 or MATRIX_ROOT}"
[[ -d "$MATRIX_ROOT" ]] || { echo "ERROR: matrix root missing: $MATRIX_ROOT"; exit 2; }

CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
FREEZE_DIR="${FREEZE_DIR:-$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a}"
REPORT_PATH="${REPORT_PATH:-$FREEZE_DIR/data_prep_freeze_report.json}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
DIAGNOSTIC_TIME="${DIAGNOSTIC_TIME:-24:00:00}"
MAX_REQUEUES="${MAX_REQUEUES:-3}"
[[ "$MAX_REQUEUES" =~ ^[0-9]+$ ]] || {
  echo "ERROR: MAX_REQUEUES must be a non-negative integer"
  exit 2
}
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
[[ -f "$PREPARED_DEV" ]] || { echo "ERROR: prepared dev missing: $PREPARED_DEV"; exit 2; }
[[ -d "$SHARED_PREFLIGHT_DIR" ]] || { echo "ERROR: shared preflight missing: $SHARED_PREFLIGHT_DIR"; exit 2; }
comparison="$MATRIX_ROOT/internal_dev_comparison.json"
[[ ! -e "$comparison" ]] || { echo "ERROR: comparison already exists: $comparison"; exit 2; }

names=(
  hierarchical_sibling_retro_frozen
  cross_token_sibling_retro_frozen
  cross_token_gated_retro_frozen
  cross_token_gated_retro_top4
)
architectures=(hierarchical_turn cross_token cross_token cross_token)
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

# Validate the whole matrix before submitting any replacement work.
for name in "${names[@]}"; do
  train_root="$MATRIX_ROOT/training/$name"
  report="$MATRIX_ROOT/diagnostics/$name/report.json"
  if [[ -f "$report" ]]; then
    continue
  fi
  if [[ -f "$train_root/checkpoints/training_summary.json" && -f "$train_root/checkpoints/best.pt" ]]; then
    continue
  fi
  [[ -f "$train_root/checkpoints/last.pt" ]] || {
    echo "ERROR: $name has neither a completed training run nor a resumable last.pt"
    exit 2
  }
done

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

common_export="CONDA_ENV=$CONDA_ENV,FREEZE_DIR=$FREEZE_DIR,REPORT_PATH=$REPORT_PATH,BACKBONE=$BACKBONE,BACKBONE_REVISION=$BACKBONE_REVISION,MAX_TURNS=$MAX_TURNS,MAX_TOKENS=$MAX_TOKENS,LENGTH_AUC_CEILING=$LENGTH_AUC_CEILING"
train_export="$common_export,TRAIN_VARIANT=$TRAIN_VARIANT,TRAIN_PATH=$TRAIN_PATH,DEV_PATH=$DEV_PATH,MODEL=guardlens,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUMULATION=$GRAD_ACCUMULATION,HEAD_LR=$HEAD_LR,BACKBONE_LR=$BACKBONE_LR,EPOCHS=$EPOCHS,LOCALIZATION_RAMP_EPOCHS=$LOCALIZATION_RAMP_EPOCHS,TURN_POOLING=attention,MAX_REQUEUES=$MAX_REQUEUES"
expected_records=$(wc -l < "$PREPARED_DEV")
finalize_jobs=()

for index in "${!names[@]}"; do
  name="${names[$index]}"
  train_root="$MATRIX_ROOT/training/$name"
  diagnostic_output="$MATRIX_ROOT/diagnostics/$name"
  if [[ -f "$diagnostic_output/report.json" ]]; then
    echo "complete $name: report already exists"
    continue
  fi

  dependency=()
  training_complete=1
  if [[ ! -f "$train_root/checkpoints/training_summary.json" || ! -f "$train_root/checkpoints/best.pt" ]]; then
    training_complete=0
    train_job=$(sbatch --parsable \
      --time="$TRAIN_TIME" \
      --export="ALL,$train_export,BACKBONE_TRAINABLE_LAYERS=${trainable_layers[$index]},BACKBONE_TURN_MICROBATCH=${backbone_microbatches[$index]},ARCHITECTURE_MODE=${architectures[$index]},ATTRIBUTION_FUSION=${fusions[$index]},INPUT_VIEW=${views[$index]},SHARED_PREFLIGHT_DIR=$SHARED_PREFLIGHT_DIR,RUN_ROOT=$train_root,PRECHECK_DIR=$train_root/preflight,OUTPUT=$train_root/checkpoints,TRAIN_RESUME=1" \
      train_naacl.slurm)
    train_job="${train_job%%;*}"
    submitted_jobs+=("$train_job")
    dependency=(--dependency="afterok:$train_job")
    echo "resume train $name: $train_job"
  fi

  shard_count=0
  if [[ -d "$diagnostic_output/shards" ]]; then
    shard_count=$(find "$diagnostic_output/shards" -maxdepth 1 -type f -name 'record-*.json' | wc -l)
  fi
  if [[ "$training_complete" == "1" && -f "$diagnostic_output/manifest.json" && "$shard_count" -eq "$expected_records" ]]; then
    diagnostic_dependency=("${dependency[@]}")
    echo "diagnostic shards complete $name: using CPU finalizer directly"
  else
    diagnostic_job=$(sbatch --parsable \
      --time="$DIAGNOSTIC_TIME" \
      "${dependency[@]}" \
      --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_CHECKPOINT=$train_root/checkpoints/best.pt,EVAL_OUTPUT=$diagnostic_output,EVAL_PROTOCOL=${protocols[$index]},MAX_REQUEUES=$MAX_REQUEUES" \
      eval_internal_dev.slurm)
    diagnostic_job="${diagnostic_job%%;*}"
    submitted_jobs+=("$diagnostic_job")
    diagnostic_dependency=(--dependency="afterok:$diagnostic_job")
    echo "resume diagnose $name: $diagnostic_job"
  fi

  finalize_job=$(sbatch --parsable \
    "${diagnostic_dependency[@]}" \
    --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_OUTPUT=$diagnostic_output" \
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
  "${compare_dependency[@]}" \
  --export="ALL,CONDA_ENV=$CONDA_ENV,MATRIX_ROOT=$MATRIX_ROOT" \
  compare_internal_dev_matrix.slurm)
compare_job="${compare_job%%;*}"
submitted_jobs+=("$compare_job")

trap - ERR INT TERM
echo "compare matrix: $compare_job"
echo "matrix root: $MATRIX_ROOT"
echo "automatic requeues per training/diagnostic job: $MAX_REQUEUES"
echo "held-out test accessed: NO"
echo "external evaluation submitted: NO"
