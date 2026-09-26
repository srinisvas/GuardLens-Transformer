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
  EVAL_DATA EVAL_CHECKPOINT EVAL_OUTPUT EVAL_PROTOCOL SMOKE_OUTPUT_DIR \
  SHARED_PREFLIGHT_DIR EXPECTED_ARCHITECTURE_MODE EXPECTED_ATTRIBUTION_FUSION \
  EXPECTED_BACKBONE_TRAINABLE_LAYERS EXPECTED_INPUT_VIEW EXPECTED_TRAIN_VARIANT

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

EXPECTED_BRANCH="naacl-causal-localization-redesign"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CURRENT_SHA=$(git rev-parse HEAD)
[[ "$CURRENT_BRANCH" == "$EXPECTED_BRANCH" ]] || {
  echo "ERROR: expected git branch $EXPECTED_BRANCH, got $CURRENT_BRANCH"
  exit 2
}
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked working-tree changes detected; commit or stash them before submission"
  exit 2
fi

validate_internal_artifact() {
  local manifest=$1
  local report=$2
  local architecture=$3
  local fusion=$4
  local layers=$5
  python - "$manifest" "$report" "$PREPARED_DEV" "$architecture" "$fusion" "$layers" "$TRAIN_VARIANT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, report_path, data_path, architecture, fusion, layers, variant = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
data_sha = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
detector = manifest.get("detector", {})
config = detector.get("config", {})
expected = {
    "stage": (manifest.get("stage"), "internal_dev_causal_diagnostic"),
    "development_only": (manifest.get("development_only"), True),
    "held_out_test_accessed": (manifest.get("held_out_test_accessed"), False),
    "dataset_sha256": (manifest.get("dataset_sha256"), data_sha),
    "protocol.view": (manifest.get("protocol", {}).get("view"), "retrospective"),
    "detector.input_view": (detector.get("input_view"), "retrospective"),
    "detector.architecture_mode": (detector.get("architecture_mode"), architecture),
    "detector.use_attribution_fusion": (
        detector.get("use_attribution_fusion"), bool(int(fusion))
    ),
    "detector.config.backbone_trainable_layers": (
        config.get("backbone_trainable_layers"), int(layers)
    ),
    "detector.config.train_variant": (config.get("train_variant"), variant),
}

failures = [f"{key}: observed={observed!r}, expected={wanted!r}"
            for key, (observed, wanted) in expected.items() if observed != wanted]
if failures:
    raise SystemExit(
        f"ERROR: invalid internal diagnostic manifest {manifest_path}: " + "; ".join(failures)
    )
if report_path != "-":
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("manifest_id") != manifest.get("manifest_id"):
        raise SystemExit(f"ERROR: report/manifest mismatch: {report_path}")
    if report.get("coverage") != {"failures": [], "scored": 364, "total": 364}:
        raise SystemExit(f"ERROR: incomplete internal report: {report_path}")
    if not report.get("development_only") or report.get("held_out_test_accessed") is not False:
        raise SystemExit(f"ERROR: report is not development-only: {report_path}")
PY
}

validate_training_artifact() {
  local summary=$1
  python - "$summary" "$TRAIN_PATH" "$DEV_PATH" "$CURRENT_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary_path, train_path, dev_path, code_sha = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

expected = {
    "status": (summary.get("status"), "completed"),
    "best_checkpoint_phase": (summary.get("best_checkpoint_phase"), 2),
    "code_sha": (summary.get("code_sha"), code_sha),
    "data_sha256.train": (
        summary.get("data_sha256", {}).get("train"), sha256(train_path)
    ),
    "data_sha256.dev": (
        summary.get("data_sha256", {}).get("dev"), sha256(dev_path)
    ),
    "held_out_test_accessed": (
        summary.get("held_out_test_accessed"), False
    ),
}
failures = [
    f"{key}: observed={observed!r}, expected={wanted!r}"
    for key, (observed, wanted) in expected.items()
    if observed != wanted
]
if failures:
    raise SystemExit(
        f"ERROR: invalid completed training summary {summary_path}: "
        + "; ".join(failures)
    )
if summary.get("best_checkpoint") not in {
    "best_localization.pt", "best_joint.pt"
}:
    raise SystemExit(
        "ERROR: completed training summary does not select a joint-phase checkpoint"
    )
PY
}

base_names=(
  hierarchical_sibling_retro_frozen
  cross_token_sibling_retro_frozen
  cross_token_gated_retro_frozen
  cross_token_gated_retro_top4
)
base_architectures=(hierarchical_turn cross_token cross_token cross_token)
base_fusions=(0 0 1 1)
base_layers=(0 0 0 4)
for index in "${!base_names[@]}"; do
  name="${base_names[$index]}"
  validate_internal_artifact \
    "$MATRIX_ROOT/diagnostics/$name/manifest.json" \
    "$MATRIX_ROOT/diagnostics/$name/report.json" \
    "${base_architectures[$index]}" "${base_fusions[$index]}" "${base_layers[$index]}"
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
  python -m guardlens.data.preflight_marker verify \
    --marker "$CONTROL_PREFLIGHT_DIR/$TRAIN_VARIANT/retrospective/marker.json" \
    --variant "$TRAIN_VARIANT" \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --code-sha "$CURRENT_SHA" \
    --input-view retrospective \
    --backbone "$BACKBONE" \
    --backbone-revision "$BACKBONE_REVISION" \
    --max-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --length-auc-ceiling "$LENGTH_AUC_CEILING"
  echo "control preflight already complete: $CONTROL_PREFLIGHT_DIR"
else
  [[ ! -e "$CONTROL_PREFLIGHT_DIR" ]] || {
    echo "ERROR: incomplete control preflight exists; archive it before resubmitting"
    exit 2
  }
  preflight_job=$(sbatch --parsable \
    --export="ALL,$common_export,MATRIX_ROOT=$MATRIX_ROOT,PREPARED_DEV=$PREPARED_DEV,SHARED_PREFLIGHT_DIR=$CONTROL_PREFLIGHT_DIR" \
    preflight_naacl_top4_sibling_controls.slurm)
  preflight_job="${preflight_job%%;*}"
  submitted_jobs+=("$preflight_job")
  preflight_dependency=(--dependency="afterok:$preflight_job")
  echo "control preflight: $preflight_job"
fi

states=()
smoke_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  train_root="$MATRIX_ROOT/training/$name"
  diagnostic_output="$MATRIX_ROOT/diagnostics/$name"
  report="$diagnostic_output/report.json"
  if [[ -f "$report" ]]; then
    validate_internal_artifact "$diagnostic_output/manifest.json" "$report" \
      "${architectures[$index]}" 0 4
    states[$index]=complete
    echo "complete $name: report already exists"
    continue
  fi
  if [[ -f "$train_root/checkpoints/training_summary.json" && -f "$train_root/checkpoints/best.pt" ]]; then
    validate_training_artifact \
      "$train_root/checkpoints/training_summary.json"
    states[$index]=trained
    echo "training complete $name: reusing best.pt"
  elif [[ -f "$train_root/checkpoints/last.pt" ]]; then
    states[$index]=resume
    echo "resumable training checkpoint found $name"
  else
    [[ ! -e "$train_root" ]] || {
      echo "ERROR: incomplete non-resumable training directory: $train_root"
      exit 2
    }
    [[ ! -e "$MATRIX_ROOT/smoke/$name" ]] || {
      echo "ERROR: stale smoke output exists; archive it before resubmitting: $MATRIX_ROOT/smoke/$name"
      exit 2
    }
    states[$index]=fresh
    smoke_job=$(sbatch --parsable \
      --kill-on-invalid-dep=yes \
      "${preflight_dependency[@]}" \
      --export="ALL,$train_export,BACKBONE_TRAINABLE_LAYERS=4,BACKBONE_TURN_MICROBATCH=1,ARCHITECTURE_MODE=${architectures[$index]},ATTRIBUTION_FUSION=0,INPUT_VIEW=retrospective,SHARED_PREFLIGHT_DIR=$CONTROL_PREFLIGHT_DIR,SMOKE_OUTPUT_DIR=$MATRIX_ROOT/smoke/$name" \
      smoke_naacl_window.slurm)
    smoke_job="${smoke_job%%;*}"
    smoke_jobs[$index]="$smoke_job"
    submitted_jobs+=("$smoke_job")
    echo "smoke $name: $smoke_job"
  fi
done

fresh_training_dependency=("${preflight_dependency[@]}")
if (( ${#smoke_jobs[@]} > 0 )); then
  smoke_dependency_ids=$(IFS=:; echo "${smoke_jobs[*]}")
  fresh_training_dependency=(--dependency="afterok:$smoke_dependency_ids")
fi

finalize_jobs=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  [[ "${states[$index]}" != complete ]] || continue
  train_root="$MATRIX_ROOT/training/$name"
  diagnostic_output="$MATRIX_ROOT/diagnostics/$name"
  expected_export="EXPECTED_ARCHITECTURE_MODE=${architectures[$index]},EXPECTED_ATTRIBUTION_FUSION=0,EXPECTED_BACKBONE_TRAINABLE_LAYERS=4,EXPECTED_INPUT_VIEW=retrospective,EXPECTED_TRAIN_VARIANT=$TRAIN_VARIANT"

  # Even when a checkpoint already exists, do not start diagnostic work until
  # the current-code control preflight has passed.
  training_dependency=("${preflight_dependency[@]}")
  if [[ "${states[$index]}" != trained ]]; then
    resume=0
    train_dependency=("${fresh_training_dependency[@]}")
    if [[ "${states[$index]}" == resume ]]; then
      resume=1
      train_dependency=("${preflight_dependency[@]}")
    fi
    train_job=$(sbatch --parsable \
      --time="$TRAIN_TIME" \
      --kill-on-invalid-dep=yes \
      "${train_dependency[@]}" \
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
  if [[ -f "$diagnostic_output/manifest.json" ]]; then
    validate_internal_artifact "$diagnostic_output/manifest.json" - \
      "${architectures[$index]}" 0 4
  elif (( shard_count > 0 )); then
    echo "ERROR: diagnostic shards exist without a manifest: $diagnostic_output"
    exit 2
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
      --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_CHECKPOINT=$train_root/checkpoints/best.pt,EVAL_OUTPUT=$diagnostic_output,EVAL_PROTOCOL=$protocol,EVAL_RECORD_INDEX=$DIAGNOSTIC_SMOKE_INDEX,MAX_REQUEUES=0,$expected_export" \
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
      --export="ALL,CONDA_ENV=$CONDA_ENV,EVAL_DATA=$PREPARED_DEV,EVAL_CHECKPOINT=$train_root/checkpoints/best.pt,EVAL_OUTPUT=$diagnostic_output,EVAL_PROTOCOL=$protocol,EVAL_SHARDS=$EVAL_SHARDS,MAX_REQUEUES=$REQUEUE_LIMIT,$expected_export" \
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
