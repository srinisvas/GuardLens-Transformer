#!/bin/bash
#SBATCH --job-name=gl-eval-cross
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_stage3_%j.out
#SBATCH --error=logs/eval_stage3_%j.err

# Stage 3: Cross-dataset generalisation (AdvBench / HarmBench / JailbreakBench)
# Run after Stage 1 completes.
# Submit: sbatch eval_stage3_cross_dataset.slurm

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
RESULTS_DIR="$HOME/work/results/dataset_gen"
CKPT_DIR="$RESULTS_DIR/checkpoints"
DATA="$RESULTS_DIR/semantic_multiturn_v10_augmented.jsonl"
OUT_DIR="$RESULTS_DIR/results"
SEEDS="$RESULTS_DIR/eval_data/cross_dataset_combined.jsonl"

CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
mkdir -p "$TMPDIR"

cd "$PROJECT_DIR"

echo "========================================================"
echo "  GuardLens Eval Stage 3: Cross-Dataset Generalisation"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$SEEDS" ]; then
    echo "ERROR: $SEEDS not found. Run Stage 1 first."
    exit 1
fi
echo "Seeds: $(wc -l < "$SEEDS") behaviors"

echo ""
echo "=== 3a. GuardLens ==="
echo "Start: $(date)"
python -m guardlens.evaluation.eval_cross_dataset \
    --seeds "$SEEDS" \
    --benign-pool "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/cross_dataset_guardlens.json" \
    --n-seeds 300 --n-benign 300 \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 --device cuda \
&& echo "3a: SUCCESS" || echo "3a: FAILED (exit $?)"

echo ""
echo "=== 3b. GuardLens-NoCF ==="
echo "Start: $(date)"
python -m guardlens.evaluation.eval_cross_dataset \
    --seeds "$SEEDS" \
    --benign-pool "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens_no_cf/best.pt" \
    --output "$OUT_DIR/cross_dataset_guardlens_no_cf.json" \
    --n-seeds 300 --n-benign 300 \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 --device cuda \
&& echo "3b: SUCCESS" || echo "3b: FAILED (exit $?)"

echo ""
echo "=== 3c. ConversationDeBERTa ==="
echo "Start: $(date)"
python -m guardlens.evaluation.eval_cross_dataset \
    --seeds "$SEEDS" \
    --benign-pool "$DATA" \
    --checkpoint "$CKPT_DIR/conversation_deberta/best.pt" \
    --output "$OUT_DIR/cross_dataset_conversation_deberta.json" \
    --n-seeds 300 --n-benign 300 \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 --device cuda \
&& echo "3c: SUCCESS" || echo "3c: FAILED (exit $?)"

echo ""
echo "========================================================"
echo "  Stage 3 complete: $(date)"
for f in cross_dataset_guardlens.json cross_dataset_guardlens_no_cf.json \
          cross_dataset_conversation_deberta.json; do
    [ -f "$OUT_DIR/$f" ] && echo "  OK: $f" || echo "  MISSING: $f"
done
echo "========================================================"
