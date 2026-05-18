#!/bin/bash
#SBATCH --job-name=gl-eval-para
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_stage4_%j.out
#SBATCH --error=logs/eval_stage4_%j.err

# Stage 4: Paraphrase robustness
# Uses HuggingFace backend (no vLLM needed -- single GPU).
# Run after Stage 1 completes.
# Submit: sbatch eval_stage4_paraphrase.slurm

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
echo "  GuardLens Eval Stage 4: Paraphrase Robustness"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Using HF backend with a small-medium model that fits alongside GuardLens.
# Qwen2.5-7B-Instruct is a good paraphraser and fits in ~16GB alongside
# GuardLens (~3GB), total ~19GB on a 40/80GB A100.
PARA_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "=== 4a. Full adversarial test set ==="
echo "Start: $(date)"
python -m guardlens.evaluation.eval_paraphrase \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/paraphrase_eval_full.json" \
    --n-samples 150 \
    --paraphrase-model hf \
    --vllm-model "$PARA_MODEL" \
    --top-k 0.15 \
    --save-examples 10 \
    --device cuda \
&& echo "4a: SUCCESS" || echo "4a: FAILED (exit $?)"

echo ""
echo "=== 4b. Implicit trigger subset ==="
echo "Start: $(date)"
python -m guardlens.evaluation.eval_paraphrase \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/paraphrase_eval_implicit.json" \
    --n-samples 80 \
    --subset contextual_pivot \
    --paraphrase-model hf \
    --vllm-model "$PARA_MODEL" \
    --top-k 0.15 \
    --save-examples 5 \
    --device cuda \
&& echo "4b: SUCCESS" || echo "4b: FAILED (exit $?)"

echo ""
echo "=== 4c. Explicit trigger subset ==="
echo "Start: $(date)"
python -m guardlens.evaluation.eval_paraphrase \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/paraphrase_eval_explicit.json" \
    --n-samples 80 \
    --subset lexical_pivot \
    --paraphrase-model hf \
    --vllm-model "$PARA_MODEL" \
    --top-k 0.15 \
    --save-examples 5 \
    --device cuda \
&& echo "4c: SUCCESS" || echo "4c: FAILED (exit $?)"

echo ""
echo "========================================================"
echo "  Stage 4 complete: $(date)"
for f in paraphrase_eval_full.json paraphrase_eval_implicit.json \
          paraphrase_eval_explicit.json; do
    [ -f "$OUT_DIR/$f" ] && echo "  OK: $f" || echo "  MISSING: $f"
done
echo "========================================================"
