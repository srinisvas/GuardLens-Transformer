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
TRAIN_PATH="${TRAIN_PATH:-$FREEZE_DIR/splits_primary/train.jsonl}"
DEV_PATH="${DEV_PATH:-$FREEZE_DIR/splits_primary/dev.jsonl}"
MAX_TURNS="${MAX_TURNS:-64}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BACKBONE="${BACKBONE:-answerdotai/ModernBERT-large}"
BACKBONE_REVISION="${BACKBONE_REVISION:-45bb4654a4d5aaff24dd11d4781fa46d39bf8c13}"
BACKBONE_TURN_MICROBATCH="${BACKBONE_TURN_MICROBATCH:-8}"

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

python -m guardlens.data.verify_freeze \
  --report "$REPORT_PATH" \
  --train "$TRAIN_PATH" \
  --dev "$DEV_PATH" \
  --variant primary

echo "========================================================"
echo " GuardLens joint architecture GPU smoke"
echo " Train: $TRAIN_PATH"
echo " Backbone: $BACKBONE"
echo " Revision: $BACKBONE_REVISION"
echo " max_turns=$MAX_TURNS max_tokens=$MAX_TOKENS batch=$BATCH_SIZE"
echo "========================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

python -m guardlens.evaluation.smoke_training_window \
  --train "$TRAIN_PATH" \
  --backbone "$BACKBONE" \
  --backbone-revision "$BACKBONE_REVISION" \
  --backbone-turn-microbatch "$BACKBONE_TURN_MICROBATCH" \
  --max-turns "$MAX_TURNS" \
  --max-tokens "$MAX_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --seed 42
