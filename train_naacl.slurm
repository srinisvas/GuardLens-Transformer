#!/bin/bash
#SBATCH --job-name=guardlens-naacl
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=logs/train_naacl_%j.out
#SBATCH --error=logs/train_naacl_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
FREEZE_DIR="${FREEZE_DIR:-$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a}"
TRAIN_VARIANT="${TRAIN_VARIANT:-primary_plus_auxiliary}"
BASE_OUTPUT="${BASE_OUTPUT:-$HOME/work/results/guardlens_naacl_redesign}"
MAX_TURNS="${MAX_TURNS:-64}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BACKBONE="${BACKBONE:-answerdotai/ModernBERT-large}"
BACKBONE_REVISION="${BACKBONE_REVISION:-45bb4654a4d5aaff24dd11d4781fa46d39bf8c13}"
BACKBONE_TURN_MICROBATCH="${BACKBONE_TURN_MICROBATCH:-8}"
BACKBONE_TRAINABLE_LAYERS="${BACKBONE_TRAINABLE_LAYERS:-0}"
TURN_POOLING="${TURN_POOLING:-attention}"
INPUT_VIEW="${INPUT_VIEW:-pre_response}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-8}"
HEAD_LR="${HEAD_LR:-2e-4}"
BACKBONE_LR="${BACKBONE_LR:-2e-5}"
EPOCHS="${EPOCHS:-20}"
LOCALIZATION_RAMP_EPOCHS="${LOCALIZATION_RAMP_EPOCHS:-5}"
MODEL="${MODEL:-guardlens}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$}}"

REPORT_PATH="${REPORT_PATH:-$FREEZE_DIR/data_prep_freeze_report.json}"
SHARED_PREFLIGHT_DIR="${SHARED_PREFLIGHT_DIR:-}"
LENGTH_AUC_CEILING="${LENGTH_AUC_CEILING:-0.650}"

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

RUN_ROOT="${RUN_ROOT:-$BASE_OUTPUT/$TRAIN_VARIANT/$MODEL/$RUN_ID}"
PRECHECK_DIR="${PRECHECK_DIR:-$RUN_ROOT/preflight}"
OUTPUT="${OUTPUT:-$RUN_ROOT/checkpoints}"

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
if [[ -e "$RUN_ROOT" || -e "$PRECHECK_DIR" || -e "$OUTPUT" ]]; then
  echo "ERROR: run output already exists; choose a fresh RUN_ID"
  echo "RUN_ROOT=$RUN_ROOT"
  exit 2
fi
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$PRECHECK_DIR" logs
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
echo "Run ID:      $RUN_ID"
echo "Run root:    $RUN_ROOT"


for path in "$REPORT_PATH" "$TRAIN_PATH" "$DEV_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path"; exit 2; }
done

if [[ "$MODEL" != "guardlens" ]]; then
  echo "ERROR: canonical launcher currently permits MODEL=guardlens only."
  echo "Baseline training will be re-enabled after the evaluation/baseline migration."
  exit 2
fi

echo "========================================================"
echo " GuardLens causal-localization training"
echo " Variant: $TRAIN_VARIANT"
echo " Train:   $TRAIN_PATH"
echo " Dev:     $DEV_PATH"
echo " Test:    NOT ACCESSED"
echo " Model:   $MODEL"
echo " Backbone:$BACKBONE"
echo " Revision:$BACKBONE_REVISION"
echo " Input:   $INPUT_VIEW"
echo " Pooling: $TURN_POOLING"
echo " Trainable backbone layers: $BACKBONE_TRAINABLE_LAYERS"
echo " Learning rates: heads=$HEAD_LR backbone=$BACKBONE_LR"
echo " Window:  turns=$MAX_TURNS hard_token_ceiling=$MAX_TOKENS"
echo " Batch:   micro=$BATCH_SIZE accumulation=$GRAD_ACCUMULATION backbone_turn_microbatch=$BACKBONE_TURN_MICROBATCH"
echo " Schedule: phase1=5 localization_ramp_epochs=$LOCALIZATION_RAMP_EPOCHS"
echo "========================================================"

if [[ -n "$SHARED_PREFLIGHT_DIR" ]]; then
  python -m guardlens.data.preflight_marker verify \
    --marker "$SHARED_PREFLIGHT_DIR/$VERIFY_VARIANT/marker.json" \
    --variant "$VERIFY_VARIANT" \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --code-sha "$CODE_SHA" \
    --input-view "$INPUT_VIEW"
  LENGTH_REPORT="$SHARED_PREFLIGHT_DIR/$VERIFY_VARIANT/length_probe_dev.json"
else
  python -m guardlens.data.verify_freeze \
    --report "$REPORT_PATH" \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --variant "$VERIFY_VARIANT" \
    --output "$PRECHECK_DIR/freeze_verification.json"

  python -m guardlens.data.audit_representation \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --backbone "$BACKBONE" \
    --backbone-revision "$BACKBONE_REVISION" \
    --max-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --input-view "$INPUT_VIEW" \
    --output "$PRECHECK_DIR/representation_coverage.json"

  python -m guardlens.evaluation.eval_length_probe_preflight \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --input-view "$INPUT_VIEW" \
    --output "$PRECHECK_DIR/length_probe_dev.json"
  LENGTH_REPORT="$PRECHECK_DIR/length_probe_dev.json"
fi

python3 - "$LENGTH_REPORT" "$LENGTH_AUC_CEILING" <<'PY'
import json, sys
path = sys.argv[1]
ceiling = float(sys.argv[2])
r = json.load(open(path, encoding="utf-8"))
auc = float(r["dev"]["roc_auc"])
bal = float(r["dev"]["balanced_accuracy"])
f1 = float(r["dev"]["f1"])
print(
    f"Length shortcut diagnostic: dev AUC={auc:.3f}, "
    f"balanced_acc={bal:.3f}, F1={f1:.3f}"
)
if auc > ceiling:
    raise SystemExit(
        f"ERROR: dev length-only AUC {auc:.3f} exceeds frozen ceiling {ceiling:.3f}; "
        "repair the data before using a GPU for training"
    )
print(f"Length shortcut gate passed: AUC <= {ceiling:.3f}")
PY

python -m guardlens.train \
  --train-path "$TRAIN_PATH" \
  --dev-path "$DEV_PATH" \
  --train-variant "$TRAIN_VARIANT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --backbone "$BACKBONE" \
  --backbone-revision "$BACKBONE_REVISION" \
  --backbone-turn-microbatch "$BACKBONE_TURN_MICROBATCH" \
  --backbone-trainable-layers "$BACKBONE_TRAINABLE_LAYERS" \
  --turn-pooling "$TURN_POOLING" \
  --input-view "$INPUT_VIEW" \
  --batch-size "$BATCH_SIZE" \
  --grad-accumulation "$GRAD_ACCUMULATION" \
  --lr "$HEAD_LR" \
  --backbone-lr "$BACKBONE_LR" \
  --epochs "$EPOCHS" \
  --phase1-epochs 5 \
  --localization-ramp-epochs "$LOCALIZATION_RAMP_EPOCHS" \
  --max-turns "$MAX_TURNS" \
  --max-tokens "$MAX_TOKENS" \
  --seed 42 \
  --device cuda

echo "========================================================"
echo "Training complete"
echo "Checkpoint: $OUTPUT/best.pt"
echo "Held-out test accessed: NO"
echo "========================================================"
