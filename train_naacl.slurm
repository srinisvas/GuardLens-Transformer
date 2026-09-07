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

# Retrain the unchanged GuardLens architecture and the existing baselines on
# the validity-repaired NAACL splits. Old v11 checkpoints are never touched.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
SPLIT_DIR="${SPLIT_DIR:-$HOME/staging/dataset_gen_output/naacl_splits}"
BASE_OUTPUT="${BASE_OUTPUT:-$HOME/work/results/guardlens_naacl/checkpoints}"

TRAIN_PATH="$SPLIT_DIR/train.jsonl"
DEV_PATH="$SPLIT_DIR/dev.jsonl"
TEST_PATH="$SPLIT_DIR/test.jsonl"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export TMPDIR="${TMPDIR:-$HOME/work/.tmp}"
export HF_HOME="${HF_HOME:-$HOME/work/hf_models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$TMPDIR" "$BASE_OUTPUT" logs

cd "$PROJECT_DIR"

for path in "$TRAIN_PATH" "$DEV_PATH" "$TEST_PATH"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing split: $path"
        exit 2
    fi
done

echo "========================================================"
echo " GuardLens NAACL validity-repair retraining"
echo " Train:  $TRAIN_PATH"
echo " Dev:    $DEV_PATH"
echo " Test:   $TEST_PATH"
echo " Output: $BASE_OUTPUT"
echo "========================================================"

# Keep the architecture/baseline suite unchanged so differences are attributable
# to repaired data/supervision rather than a model redesign.
MODELS=("guardlens" "guardlens_no_fusion" "guardlens_no_cf" "turn_level" "conversation_deberta")
EPOCHS=(25 20 20 20 20)
BATCH=(4 4 4 4 2)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    EP="${EPOCHS[$i]}"
    BS="${BATCH[$i]}"
    OUTPUT="$BASE_OUTPUT/$MODEL"
    mkdir -p "$OUTPUT"

    EXTRA_FLAGS=()
    if [[ "$MODEL" == "conversation_deberta" || "$MODEL" == "turn_level" ]]; then
        EXTRA_FLAGS+=("--no-pivot-head")
    fi

    echo ""
    echo "Training $MODEL at $(date)"
    python -m guardlens.train \
        --train-path "$TRAIN_PATH" \
        --dev-path "$DEV_PATH" \
        --test-path "$TEST_PATH" \
        --output "$OUTPUT" \
        --model "$MODEL" \
        --backbone microsoft/deberta-v3-base \
        --batch-size "$BS" \
        --lr 2e-4 \
        --epochs "$EP" \
        --max-turns 32 \
        --max-tokens 192 \
        --seed 42 \
        "${EXTRA_FLAGS[@]}"
done

echo "NAACL retraining complete: $BASE_OUTPUT"
