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
FREEZE_DIR="${FREEZE_DIR:-$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze}"
TRAIN_VARIANT="${TRAIN_VARIANT:-primary}"
BASE_OUTPUT="${BASE_OUTPUT:-$HOME/work/results/guardlens_naacl_redesign/checkpoints}"
PRECHECK_DIR="${PRECHECK_DIR:-$HOME/work/results/guardlens_naacl_redesign/preflight}"
MAX_LENGTH_AUC="${MAX_LENGTH_AUC:-0.65}"
MAX_TURNS="${MAX_TURNS:-48}"
MAX_TOKENS="${MAX_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-8}"
EPOCHS="${EPOCHS:-20}"
MODEL="${MODEL:-guardlens}"

REPORT_PATH="${REPORT_PATH:-$FREEZE_DIR/data_prep_freeze_report.json}"

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
export HF_HOME="${HF_HOME:-$HOME/work/hf_models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$TMPDIR" "$BASE_OUTPUT" "$PRECHECK_DIR" logs
cd "$PROJECT_DIR"

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
echo " Window:  turns=$MAX_TURNS hard_token_ceiling=$MAX_TOKENS"
echo " Batch:   micro=$BATCH_SIZE accumulation=$GRAD_ACCUMULATION"
echo "========================================================"

python -m guardlens.data.verify_freeze \
  --report "$REPORT_PATH" \
  --train "$TRAIN_PATH" \
  --dev "$DEV_PATH" \
  --variant "$VERIFY_VARIANT" \
  --output "$PRECHECK_DIR/freeze_verification.json"

python -m guardlens.data.audit_representation \
  --train "$TRAIN_PATH" \
  --dev "$DEV_PATH" \
  --backbone microsoft/deberta-v3-base \
  --max-turns "$MAX_TURNS" \
  --max-tokens "$MAX_TOKENS" \
  --output "$PRECHECK_DIR/representation_coverage.json"

python -m guardlens.evaluation.eval_length_probe_preflight \
  --train "$TRAIN_PATH" \
  --dev "$DEV_PATH" \
  --output "$PRECHECK_DIR/length_probe_dev.json"

python3 - "$PRECHECK_DIR/length_probe_dev.json" "$MAX_LENGTH_AUC" <<'PY'
import json, sys
path, limit = sys.argv[1], float(sys.argv[2])
r = json.load(open(path, encoding="utf-8"))
auc = float(r["dev"]["roc_auc"])
bal = float(r["dev"]["balanced_accuracy"])
print(f"Length shortcut preflight: dev AUC={auc:.3f}, balanced_acc={bal:.3f}, gate={limit:.3f}")
if auc > limit:
    raise SystemExit(
        f"ERROR: dev length-only AUC {auc:.3f} exceeds {limit:.3f}"
    )
PY

OUTPUT="$BASE_OUTPUT/$TRAIN_VARIANT/$MODEL"
mkdir -p "$OUTPUT"

python -m guardlens.train \
  --train-path "$TRAIN_PATH" \
  --dev-path "$DEV_PATH" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --backbone microsoft/deberta-v3-base \
  --batch-size "$BATCH_SIZE" \
  --grad-accumulation "$GRAD_ACCUMULATION" \
  --lr 2e-4 \
  --epochs "$EPOCHS" \
  --phase1-epochs 5 \
  --max-turns "$MAX_TURNS" \
  --max-tokens "$MAX_TOKENS" \
  --seed 42

echo "========================================================"
echo "Training complete"
echo "Checkpoint: $OUTPUT/best.pt"
echo "Held-out test accessed: NO"
echo "========================================================"
