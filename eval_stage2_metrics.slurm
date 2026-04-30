#!/bin/bash
#SBATCH --job-name=guardlens_eval_metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_metrics_%j.out
#SBATCH --error=logs/eval_metrics_%j.err

set -euo pipefail
mkdir -p logs

echo "========================================================"
echo "  GuardLens Eval Stage 2: External Eval + Subset Analysis"
echo "  Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Start: $(date)"
echo "========================================================"

source ~/.bashrc
conda activate guardlens 2>/dev/null || source ~/work/venv/guardlens/bin/activate

RESULTS_DIR=~/work/results/dataset_gen
CKPT_DIR=$RESULTS_DIR/checkpoints
DATA=$RESULTS_DIR/semantic_multiturn_v10_augmented.jsonl
OUT_DIR=$RESULTS_DIR/results
mkdir -p $OUT_DIR

cd ~/work/GuardLens-Transformer

# -------------------------------------------------------
# 2a. Implicit vs Explicit Subset Analysis
# -------------------------------------------------------
echo ""
echo "=== 2a. Implicit / Explicit / Hard-Negative Subset Analysis ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_implicit_explicit \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/implicit_explicit_eval.json" \
    --methods guardlens surface_risk integrated_gradients grad_x_input attention random \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda

echo "2a complete: $(date)"

# -------------------------------------------------------
# 2b. External Evaluator -- Full Test Set
# -------------------------------------------------------
echo ""
echo "=== 2b. External Evaluator (full test set) ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_external \
    --data "$DATA" \
    --attribution-checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --external-checkpoints \
        "$CKPT_DIR/conversation_deberta/best.pt" \
        "$CKPT_DIR/turn_level/best.pt" \
    --output "$OUT_DIR/external_eval_full.json" \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda

echo "2b complete: $(date)"

# -------------------------------------------------------
# 2c. External Evaluator -- Implicit Subset Only
# -------------------------------------------------------
echo ""
echo "=== 2c. External Evaluator (implicit trigger subset) ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_external \
    --data "$DATA" \
    --attribution-checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --external-checkpoints \
        "$CKPT_DIR/conversation_deberta/best.pt" \
    --output "$OUT_DIR/external_eval_implicit.json" \
    --top-k 0.05 0.10 0.15 0.20 \
    --subset implicit \
    --batch-size 4 \
    --device cuda

echo "2c complete: $(date)"

# -------------------------------------------------------
# 2d. External Evaluator -- Explicit Subset Only
# -------------------------------------------------------
echo ""
echo "=== 2d. External Evaluator (explicit trigger subset) ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_external \
    --data "$DATA" \
    --attribution-checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --external-checkpoints \
        "$CKPT_DIR/conversation_deberta/best.pt" \
    --output "$OUT_DIR/external_eval_explicit.json" \
    --top-k 0.05 0.10 0.15 0.20 \
    --subset explicit \
    --batch-size 4 \
    --device cuda

echo "2d complete: $(date)"

echo ""
echo "========================================================"
echo "  Stage 2 complete: $(date)"
echo "  Outputs:"
for f in implicit_explicit_eval.json external_eval_full.json \
          external_eval_implicit.json external_eval_explicit.json; do
    if [ -f "$OUT_DIR/$f" ]; then
        echo "    OK: $OUT_DIR/$f"
    else
        echo "    MISSING: $OUT_DIR/$f"
    fi
done
echo "========================================================"
