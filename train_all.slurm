#!/bin/bash
#SBATCH --job-name=guardlens-all
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err

# ============================================================
# Train GuardLens + all baselines/ablations for the paper.
#
# Runs sequentially on one GPU:
#
#   1. GuardLens (full model)         ~2-3 hours
#   2. GuardLens-NoFusion (ablation)  ~2 hours
#   3. GuardLens-NoCF (ablation)      ~1.5 hours (no phase 3)
#   4. Turn-level classifier          ~1.5 hours
#
# Total: ~7-9 hours. Fits in 10h walltime.
#
# Usage:
#   mkdir -p logs
#   sbatch train_all.slurm
# ============================================================

set -uo pipefail  # No -e: we want to continue even if one model fails

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/dataset_gen"
DATA="$HOME/work/results/dataset_gen/semantic_multiturn_v10_augmented.jsonl"
BASE_OUTPUT="$HOME/work/results/dataset_gen/checkpoints"

# ---- Environment ----
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
mkdir -p "$TMPDIR"

cd "$PROJECT_DIR"

echo "========================================================"
echo "  GuardLens: Full experiment suite"
echo "  Job:    ${SLURM_JOB_ID:-interactive}"
echo "  Node:   $(hostname)"
echo "  Data:   $DATA"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODELS=("guardlens" "guardlens_no_fusion" "guardlens_no_cf" "turn_level" "conversation_deberta")
EPOCHS=(25 20 20 20 20)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    EP="${EPOCHS[$i]}"
    OUTPUT="$BASE_OUTPUT/$MODEL"
    mkdir -p "$OUTPUT"

    echo ""
    echo "========================================================"
    echo "  Training: $MODEL (${EP} epochs)"
    echo "  Output:   $OUTPUT"
    echo "  Start:    $(date +%H:%M:%S)"
    echo "========================================================"

    python -m guardlens.train \
        --data "$DATA" \
        --output "$OUTPUT" \
        --model "$MODEL" \
        --backbone microsoft/deberta-v3-base \
        --batch-size 8 \
        --lr 2e-4 \
        --epochs "$EP" \
        --seed 42 \
    && echo "  $MODEL: SUCCESS" \
    || echo "  $MODEL: FAILED (exit code $?)"

    echo "  End: $(date +%H:%M:%S)"
done

echo ""
echo "========================================================"
echo "  All experiments complete."
echo "  Results in: $BASE_OUTPUT/"
echo "========================================================"

# Print summary table
echo ""
echo "  Model                  | Test results"
echo "  -----------------------|-------------"
for MODEL in "${MODELS[@]}"; do
    RESULT="$BASE_OUTPUT/$MODEL/test_results.json"
    if [[ -f "$RESULT" ]]; then
        F1=$(python3 -c "import json; r=json.load(open('$RESULT')); print(f'{r[\"f1\"]:.4f}')" 2>/dev/null || echo "N/A")
        ACC=$(python3 -c "import json; r=json.load(open('$RESULT')); print(f'{r[\"accuracy\"]:.4f}')" 2>/dev/null || echo "N/A")
        AF1=$(python3 -c "import json; r=json.load(open('$RESULT')); print(f'{r.get(\"attr_f1\",0):.4f}')" 2>/dev/null || echo "N/A")
        printf "  %-23s | F1=%s Acc=%s AttrF1=%s\n" "$MODEL" "$F1" "$ACC" "$AF1"
    else
        printf "  %-23s | No results (failed?)\n" "$MODEL"
    fi
done
