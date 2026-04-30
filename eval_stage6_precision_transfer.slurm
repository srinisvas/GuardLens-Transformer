#!/bin/bash
#SBATCH --job-name=gl-eval-precision
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_stage6_%j.out
#SBATCH --error=logs/eval_stage6_%j.err

# Stage 6: Hard negative attribution precision + minimality curve + ShieldGemma transfer
# Two-phase: GuardLens first (attribution), then ShieldGemma (scoring).
# They never load simultaneously -- GuardLens is freed before ShieldGemma loads.
# Run after Stage 1 completes.
# Submit: sbatch eval_stage6_precision_transfer.slurm

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
RESULTS_DIR="$HOME/work/results/dataset_gen"
CKPT_DIR="$RESULTS_DIR/checkpoints"
DATA="$RESULTS_DIR/semantic_multiturn_v10_augmented.jsonl"
OUT_DIR="$RESULTS_DIR/results"

CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
mkdir -p "$TMPDIR"

cd "$PROJECT_DIR"

echo "========================================================"
echo "  GuardLens Eval Stage 6: Precision + Transfer"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install bitsandbytes if needed for 8-bit ShieldGemma
python3 -c "import bitsandbytes" 2>/dev/null || {
    echo "Installing bitsandbytes..."
    pip install bitsandbytes --quiet
}

# -------------------------------------------------------
# 6a. Hard negative attribution precision + minimality curve
# -------------------------------------------------------
echo ""
echo "=== 6a. Attribution Precision + Minimality Curve ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_attribution_precision \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/attribution_precision.json" \
    --device cuda \
    --batch-size 8 \
    --k-fracs 0.02 0.05 0.08 0.10 0.12 0.15 0.18 0.20 0.25 0.30 \
    --methods guardlens surface_risk integrated_gradients grad_x_input random \
&& echo "6a: SUCCESS" || echo "6a: FAILED (exit $?)"

echo "6a end: $(date)"

# -------------------------------------------------------
# 6b. Cross-model transfer (two-phase, ShieldGemma-9B)
# Phase 1: GuardLens extracts attributions -> variants saved to disk -> GuardLens freed
# Phase 2: ShieldGemma-9B loaded in 8-bit (~9GB) -> scores all variants
# -------------------------------------------------------
echo ""
echo "=== 6b. Cross-Model Transfer (ShieldGemma-9B, 8-bit) ==="
echo "Start: $(date)"

# Full test set (50 samples)
python -m guardlens.evaluation.eval_cross_model_transfer \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/cross_model_transfer_full.json" \
    --llamaguard-model google/shieldgemma-9b \
    --quantization 8bit \
    --n-samples 50 \
    --k-frac 0.15 \
    --variants-cache "$OUT_DIR/transfer_variants_full.jsonl" \
    --device cuda \
&& echo "6b-full: SUCCESS" || echo "6b-full: FAILED (exit $?)"

echo "6b-full end: $(date)"

# Implicit subset (30 samples -- key result)
python -m guardlens.evaluation.eval_cross_model_transfer \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/cross_model_transfer_implicit.json" \
    --llamaguard-model google/shieldgemma-9b \
    --quantization 8bit \
    --subset implicit \
    --n-samples 30 \
    --k-frac 0.15 \
    --variants-cache "$OUT_DIR/transfer_variants_implicit.jsonl" \
    --device cuda \
&& echo "6b-implicit: SUCCESS" || echo "6b-implicit: FAILED (exit $?)"

echo "6b-implicit end: $(date)"

echo ""
echo "========================================================"
echo "  Stage 6 complete: $(date)"
for f in attribution_precision.json cross_model_transfer_full.json \
          cross_model_transfer_implicit.json; do
    [ -f "$OUT_DIR/$f" ] && echo "  OK: $f" || echo "  MISSING: $f"
done
echo "========================================================"
