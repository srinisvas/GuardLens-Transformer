#!/bin/bash
#SBATCH --job-name=guardlens-all
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err

# ============================================================
# Train GuardLens + all baselines/ablations for the paper.
#
# Uses v11 pre-split dataset with interactive adversarial
# conversations + separate clean benign pool.
#
# Runs sequentially on one GPU:
#   1. GuardLens (full model)              ~3-4 hours
#   2. GuardLens-NoFusion (ablation)       ~2-3 hours
#   3. GuardLens-NoCF (ablation)           ~2 hours (no phase 3)
#   4. Turn-level classifier (baseline)    ~2 hours
#   5. ConversationDeBERTa (baseline)      ~1.5 hours
#
# Total: ~10-13 hours. Fits in 18h walltime.
#
# Usage:
#   mkdir -p logs
#   sbatch train_all.slurm
# ============================================================

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"

# v11 pre-split paths
SPLIT_DIR="$HOME/staging/dataset_gen_output/splits"
TRAIN_PATH="${SPLIT_DIR}/train.jsonl"
DEV_PATH="${SPLIT_DIR}/dev.jsonl"
TEST_PATH="${SPLIT_DIR}/test.jsonl"

BASE_OUTPUT="$HOME/work/results/guardlens_v11/checkpoints"

# ---- Environment ----
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR"

cd "$PROJECT_DIR"

echo "========================================================"
echo "  GuardLens v11: Full experiment suite"
echo "  Job:       ${SLURM_JOB_ID:-interactive}"
echo "  Node:      $(hostname)"
echo "  Train:     $TRAIN_PATH"
echo "  Dev:       $DEV_PATH"
echo "  Test:      $TEST_PATH"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify data exists
for f in "$TRAIN_PATH" "$DEV_PATH" "$TEST_PATH"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing data file: $f"
        exit 1
    fi
    echo "  $(wc -l < "$f") records in $(basename $f)"
done

# Models and their configs
MODELS=("guardlens" "guardlens_no_fusion" "guardlens_no_cf" "turn_level" "conversation_deberta")
EPOCHS=(25 20 20 20 20)
BATCH=(4 4 4 4 2)        # conversation_deberta needs smaller batch (longer sequences)
MAX_TURNS=(32 32 32 32 32)  # All models see all turns; flat baseline flattens after
MAX_TOKENS=(192 192 192 192 192)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    EP="${EPOCHS[$i]}"
    BS="${BATCH[$i]}"
    MT="${MAX_TURNS[$i]}"
    MK="${MAX_TOKENS[$i]}"
    OUTPUT="$BASE_OUTPUT/$MODEL"
    mkdir -p "$OUTPUT"

    echo ""
    echo "========================================================"
    echo "  Training: $MODEL"
    echo "  Epochs: $EP, Batch: $BS, MaxTurns: $MT"
    echo "  Output: $OUTPUT"
    echo "  Start:  $(date +%H:%M:%S)"
    echo "========================================================"

    EXTRA_FLAGS=""
    if [ "$MODEL" = "conversation_deberta" ]; then
        EXTRA_FLAGS="--no-pivot-head"
    fi
    if [ "$MODEL" = "turn_level" ]; then
        EXTRA_FLAGS="--no-pivot-head"
    fi

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
        --max-turns "$MT" \
        --max-tokens "$MK" \
        --seed 42 \
        $EXTRA_FLAGS \
    && echo "  $MODEL: SUCCESS ($(date +%H:%M:%S))" \
    || echo "  $MODEL: FAILED (exit code $?, $(date +%H:%M:%S))"

    echo ""
done

echo ""
echo "========================================================"
echo "  All experiments complete."
echo "  Results in: $BASE_OUTPUT/"
echo "========================================================"

# Print summary table
echo ""
printf "  %-25s | %-8s %-8s %-8s %-8s %-8s\n" "Model" "F1" "Acc" "AttrF1" "PivAcc" "Thresh"
printf "  %-25s-+-%-8s-%-8s-%-8s-%-8s-%-8s\n" "-------------------------" "--------" "--------" "--------" "--------" "--------"

for MODEL in "${MODELS[@]}"; do
    RESULT="$BASE_OUTPUT/$MODEL/test_results.json"
    if [[ -f "$RESULT" ]]; then
        python3 -c "
import json
r = json.load(open('$RESULT'))
f1 = f'{r.get(\"f1\", 0):.4f}'
acc = f'{r.get(\"accuracy\", 0):.4f}'
af1 = f'{r.get(\"attr_f1\", 0):.4f}'
piv = f'{r.get(\"pivot_accuracy\", 0):.4f}'
thr = f'{r.get(\"threshold\", 0.5):.2f}'
print(f'  {\"$MODEL\":<25} | {f1:<8} {acc:<8} {af1:<8} {piv:<8} {thr:<8}')
" 2>/dev/null || printf "  %-25s | parse error\n" "$MODEL"
    else
        printf "  %-25s | No results (failed?)\n" "$MODEL"
    fi
done

# Transfer tier breakdown for main model
MAIN_RESULT="$BASE_OUTPUT/guardlens/test_results.json"
if [[ -f "$MAIN_RESULT" ]]; then
    echo ""
    echo "  GuardLens transfer tier accuracy:"
    python3 -c "
import json
r = json.load(open('$MAIN_RESULT'))
tiers = r.get('transfer_tier_accuracy', {})
for k, v in sorted(tiers.items()):
    print(f'    {k}: {v:.4f}')
benign = r.get('benign_accuracy', {})
if benign:
    print('  Benign accuracy:')
    for k, v in sorted(benign.items()):
        print(f'    {k}: {v:.4f}')
sup = r.get('supervision_tier_accuracy', {})
if sup:
    print('  Supervision tier accuracy:')
    for k, v in sorted(sup.items()):
        print(f'    {k}: {v:.4f}')
" 2>/dev/null
fi
