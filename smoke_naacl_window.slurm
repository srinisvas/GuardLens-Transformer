#!/bin/bash
#SBATCH --job-name=gl_naacl_smoke
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=logs/naacl_window_smoke_%j.out
#SBATCH --error=logs/naacl_window_smoke_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
FREEZE_DIR="${FREEZE_DIR:-$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a}"
REPORT_PATH="${REPORT_PATH:-$FREEZE_DIR/data_prep_freeze_report.json}"
TRAIN_VARIANT="${TRAIN_VARIANT:-primary_plus_auxiliary}"
SHARED_PREFLIGHT_DIR="${SHARED_PREFLIGHT_DIR:-}"
LENGTH_AUC_CEILING="${LENGTH_AUC_CEILING:-0.650}"
MAX_TURNS="${MAX_TURNS:-64}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BACKBONE="${BACKBONE:-answerdotai/ModernBERT-large}"
BACKBONE_REVISION="${BACKBONE_REVISION:-45bb4654a4d5aaff24dd11d4781fa46d39bf8c13}"
BACKBONE_TURN_MICROBATCH="${BACKBONE_TURN_MICROBATCH:-8}"
BACKBONE_TRAINABLE_LAYERS="${BACKBONE_TRAINABLE_LAYERS:-0}"
ARCHITECTURE_MODE="${ARCHITECTURE_MODE:-hierarchical_turn}"
ATTRIBUTION_FUSION="${ATTRIBUTION_FUSION:-0}"
TURN_POOLING="${TURN_POOLING:-attention}"
INPUT_VIEW="${INPUT_VIEW:-retrospective}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$}}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$HOME/work/results/guardlens_naacl_redesign/smoke/$SMOKE_RUN_ID}"

case "$TRAIN_VARIANT" in
  primary)
    TRAIN_PATH="${TRAIN_PATH:-$FREEZE_DIR/splits_primary/train.jsonl}"
    DEV_PATH="${DEV_PATH:-$FREEZE_DIR/splits_primary/dev.jsonl}"
    VERIFY_VARIANT="primary"
    ;;
  primary_plus_auxiliary)
    TRAIN_PATH="${TRAIN_PATH:-$FREEZE_DIR/splits_primary_plus_train_auxiliary/train.jsonl}"
    DEV_PATH="${DEV_PATH:-$FREEZE_DIR/splits_primary_plus_train_auxiliary/dev.jsonl}"
    VERIFY_VARIANT="primary_plus_auxiliary"
    ;;
  *)
    echo "ERROR: TRAIN_VARIANT must be primary or primary_plus_auxiliary"
    exit 2
    ;;
esac

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
if [[ -e "$SMOKE_OUTPUT_DIR" ]]; then
  echo "ERROR: smoke output already exists: $SMOKE_OUTPUT_DIR"
  exit 2
fi
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$SMOKE_OUTPUT_DIR" logs
cd "$PROJECT_DIR"
echo "HF_HOME:      $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"

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
export GUARDLENS_CODE_SHA="$CODE_SHA"
echo "Code branch: $CURRENT_BRANCH"
echo "Code SHA:    $CODE_SHA"


for path in "$REPORT_PATH" "$TRAIN_PATH" "$DEV_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path"; exit 2; }
done

if [[ -n "$SHARED_PREFLIGHT_DIR" ]]; then
  python -m guardlens.data.preflight_marker verify \
    --marker "$SHARED_PREFLIGHT_DIR/$VERIFY_VARIANT/$INPUT_VIEW/marker.json" \
    --variant "$VERIFY_VARIANT" \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --code-sha "$CODE_SHA" \
    --input-view "$INPUT_VIEW" \
    --backbone "$BACKBONE" \
    --backbone-revision "$BACKBONE_REVISION" \
    --max-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --length-auc-ceiling "$LENGTH_AUC_CEILING"
else
  python -m guardlens.data.verify_freeze \
    --report "$REPORT_PATH" \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --variant "$VERIFY_VARIANT" \
    --output "$SMOKE_OUTPUT_DIR/freeze_verification.json"

  python -m guardlens.data.audit_representation \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --backbone "$BACKBONE" \
    --backbone-revision "$BACKBONE_REVISION" \
    --max-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --input-view "$INPUT_VIEW" \
    --output "$SMOKE_OUTPUT_DIR/representation_coverage.json"
fi

echo "========================================================"
echo " GuardLens joint architecture GPU smoke"
echo " Train: $TRAIN_PATH"
echo " Variant: $TRAIN_VARIANT"
echo " Backbone: $BACKBONE"
echo " Revision: $BACKBONE_REVISION"
echo " Input: $INPUT_VIEW"
echo " Pooling: $TURN_POOLING"
echo " Architecture: $ARCHITECTURE_MODE"
echo " Attribution fusion: $ATTRIBUTION_FUSION"
echo " Trainable backbone layers: $BACKBONE_TRAINABLE_LAYERS"
echo " max_turns=$MAX_TURNS max_tokens=$MAX_TOKENS batch=$BATCH_SIZE"
echo "========================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

fusion_args=()
if [[ "$ATTRIBUTION_FUSION" == "1" ]]; then
  fusion_args+=(--attribution-fusion)
elif [[ "$ATTRIBUTION_FUSION" != "0" ]]; then
  echo "ERROR: ATTRIBUTION_FUSION must be 0 or 1"
  exit 2
fi

python -m guardlens.evaluation.smoke_training_window \
  --train "$TRAIN_PATH" \
  --train-variant "$TRAIN_VARIANT" \
  --backbone "$BACKBONE" \
  --backbone-revision "$BACKBONE_REVISION" \
  --backbone-turn-microbatch "$BACKBONE_TURN_MICROBATCH" \
  --backbone-trainable-layers "$BACKBONE_TRAINABLE_LAYERS" \
  --architecture-mode "$ARCHITECTURE_MODE" \
  --turn-pooling "$TURN_POOLING" \
  --input-view "$INPUT_VIEW" \
  --max-turns "$MAX_TURNS" \
  --max-tokens "$MAX_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --seed 42 \
  "${fusion_args[@]}"
