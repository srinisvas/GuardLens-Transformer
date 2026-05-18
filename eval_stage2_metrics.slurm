#!/bin/bash
#SBATCH --job-name=gl-eval-metrics
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_stage2_%j.out
#SBATCH --error=logs/eval_stage2_%j.err

# Stage 2: External evaluator + implicit/explicit subset analysis
# Run after Stage 1 completes.
# Submit: sbatch eval_stage2_metrics.slurm

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints"
TEST_DATA="$HOME/staging/dataset_gen_output/splits/test.jsonl"
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
echo "  GuardLens Eval Stage 2: External Eval + Subset Analysis"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -------------------------------------------------------
# 2a. Implicit / Explicit / Hard-Negative Subset Analysis
# -------------------------------------------------------
echo ""
echo "=== 2a. Implicit/Explicit/Hard-Negative Subset Analysis ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_implicit_explicit \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/implicit_explicit_eval.json" \
    --methods guardlens surface_risk attention random \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda \
&& echo "2a: SUCCESS" || echo "2a: FAILED (exit $?)"

python -m guardlens.evaluation.eval_implicit_explicit \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/implicit_explicit_eval_grad.json" \
    --methods integrated_gradients grad_x_input \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 1 \
    --device cuda \
&& echo "2a-grad: SUCCESS" || echo "2a-grad: FAILED (exit $?)"

echo "2a end: $(date)"

# -------------------------------------------------------
# 2b. External Evaluator -- Full Test Set
# -------------------------------------------------------
echo ""
echo "=== 2b. External Evaluator (full test set) ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_external \
    --test-path "$TEST_DATA" \
    --attribution-checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --external-checkpoints \
        "$CKPT_DIR/conversation_deberta/best.pt" \
        "$CKPT_DIR/turn_level/best.pt" \
    --output "$OUT_DIR/external_eval_full.json" \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda \
&& echo "2b: SUCCESS" || echo "2b: FAILED (exit $?)"

echo "2b end: $(date)"

# -------------------------------------------------------
# 2c. External Evaluator -- Implicit Subset
# -------------------------------------------------------
echo ""
echo "=== 2c. External Evaluator (implicit subset) ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_external \
    --test-path "$TEST_DATA" \
    --attribution-checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --external-checkpoints "$CKPT_DIR/conversation_deberta/best.pt" \
    --output "$OUT_DIR/external_eval_contextual.json" \
    --top-k 0.05 0.10 0.15 0.20 \
    --subset contextual_pivot \
    --batch-size 4 \
    --device cuda \
&& echo "2c: SUCCESS" || echo "2c: FAILED (exit $?)"

echo "2c end: $(date)"

# -------------------------------------------------------
# 2d. External Evaluator -- Explicit Subset
# -------------------------------------------------------
echo ""
echo "=== 2d. External Evaluator (explicit subset) ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_external \
    --test-path "$TEST_DATA" \
    --attribution-checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --external-checkpoints "$CKPT_DIR/conversation_deberta/best.pt" \
    --output "$OUT_DIR/external_eval_lexical.json" \
    --top-k 0.05 0.10 0.15 0.20 \
    --subset lexical_pivot \
    --batch-size 4 \
    --device cuda \
&& echo "2d: SUCCESS" || echo "2d: FAILED (exit $?)"

echo "2d end: $(date)"

# ---- 2e: Boundary stress test (issue 13) ----
echo ""
echo "2e: Boundary stress test..."

BOUNDARY1="$RESULTS_DIR/old_benign_boundary_or_unused.jsonl"
BOUNDARY2="$RESULTS_DIR/benign_boundary.jsonl"

BOUNDARY_FILES=""
[ -f "$BOUNDARY1" ] && BOUNDARY_FILES="$BOUNDARY_FILES $BOUNDARY1"
[ -f "$BOUNDARY2" ] && BOUNDARY_FILES="$BOUNDARY_FILES $BOUNDARY2"

if [ -n "$BOUNDARY_FILES" ]; then
    python -m guardlens.evaluation.eval_boundary_stress \
        --boundary-files $BOUNDARY_FILES \
        --checkpoint "$CKPT_DIR/guardlens/best.pt" \
        --output "$OUT_DIR/boundary_stress.json" \
        --device cuda \
    && echo "2e: SUCCESS" || echo "2e: FAILED"
else
    echo "2e: SKIPPED (no boundary files found)"
fi

echo "2e end: $(date)"

echo ""
echo "========================================================"
echo "  Stage 2 complete: $(date)"
echo "  Outputs in: $OUT_DIR"
for f in implicit_explicit_eval.json external_eval_full.json \
          external_eval_contextual.json external_eval_lexical.json \
          boundary_stress.json; do
    [ -f "$OUT_DIR/$f" ] && echo "  OK: $f" || echo "  MISSING: $f"
done
echo "========================================================"
