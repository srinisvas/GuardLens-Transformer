#!/bin/bash
#SBATCH --job-name=gl-eval-core
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_core_%j.out
#SBATCH --error=logs/eval_core_%j.err

# ============================================================
# eval_core.slurm — Core evaluation (Table 1 + Table 2)
#
# Runs:
#   1. Causal attribution eval (GuardLens + all baselines)
#      → Deviation Drop, Flip Rate, Necessity, Sufficiency
#      → Per-supervision-tier breakdown
#      → Per-transfer-tier breakdown
#      → LaTeX table output
#
#   2. Classification eval for all 5 models
#      → F1, Acc, Precision, Recall with dev-tuned threshold
#      → Per-family, per-transfer-tier, per-benign-status accuracy
#
# This produces the primary paper tables.
#
# Usage:
#   sbatch eval_core.slurm
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints"
TEST_DATA="$HOME/staging/dataset_gen_output/splits/test.jsonl"
OUT_DIR="$RESULTS_DIR/results"

CONDA_ENV="$HOME/work/conda_envs/guardlens_train"

# ---- Environment ----
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1
export TMPDIR="$HOME/work/.tmp"
mkdir -p "$TMPDIR" "$OUT_DIR"

cd "$SLURM_SUBMIT_DIR"

echo "========================================================"
echo "  GuardLens Core Evaluation"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  Test data: $TEST_DATA"
echo "  Checkpoints: $CKPT_DIR"
echo "  Output: $OUT_DIR"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify inputs
if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data not found: $TEST_DATA"
    exit 1
fi

MODELS=("guardlens" "guardlens_no_fusion" "guardlens_no_cf" "turn_level" "conversation_deberta")

for m in "${MODELS[@]}"; do
    if [ ! -f "$CKPT_DIR/$m/best.pt" ]; then
        echo "WARNING: Missing checkpoint: $CKPT_DIR/$m/best.pt"
    fi
done

# ============================================================
# Part 1: Causal Attribution Eval (GuardLens main model)
# ============================================================

echo ""
echo "========================================================"
echo "  Part 1: Causal Attribution Evaluation"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

echo ""
echo "=== 1a. Full causal eval with all methods ==="
python -m guardlens.evaluation.eval_causal \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/causal_eval_results.json" \
    --methods guardlens attention integrated_gradients grad_x_input surface_risk random \
    --top-k 0.05 0.10 0.15 0.20 \
    --tier-eval \
    --transfer-eval \
    --batch-size 4 \
    --device cuda \
&& echo "1a: SUCCESS" || echo "1a: FAILED (exit $?)"

echo "1a end: $(date +%H:%M:%S)"

# ============================================================
# Part 2: Classification Eval (all 5 models)
# ============================================================

echo ""
echo "========================================================"
echo "  Part 2: Classification Evaluation (all models)"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

for MODEL in "${MODELS[@]}"; do
    CKPT="$CKPT_DIR/$MODEL/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "  $MODEL: SKIPPED (no checkpoint)"
        continue
    fi

    echo ""
    echo "=== Classification: $MODEL ==="
    python -m guardlens.evaluate \
        --test-path "$TEST_DATA" \
        --checkpoint "$CKPT" \
        --batch-size 8 \
        --device cuda \
    > "$OUT_DIR/classification_${MODEL}.json" 2>&1 \
    && echo "  $MODEL: SUCCESS" || echo "  $MODEL: FAILED (exit $?)"
done

echo ""
echo "Part 2 end: $(date +%H:%M:%S)"

# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================================"
echo "  Core Evaluation Complete: $(date +%H:%M:%S)"
echo "========================================================"

# Print causal eval summary if available
if [ -f "$OUT_DIR/causal_eval_results.json" ]; then
    echo ""
    echo "  Causal eval results:"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/causal_eval_results.json'))
full = r.get('full_test', {})
for method, data in full.items():
    dd = data.get('deviation_drops', {}).get('15%', 0)
    flip = data.get('flip_rates', {}).get('flip@15%', 0)
    print(f'    {method:<25} DD@15%={dd:.3f}  Flip@15%={flip:.3f}')
" 2>/dev/null || echo "  (could not parse results)"
fi

# Print classification summary
echo ""
echo "  Classification results:"
printf "  %-25s | %-8s %-8s %-8s %-8s\n" "Model" "F1" "Acc" "AttrF1" "Thresh"
printf "  %-25s-+-%-8s-%-8s-%-8s-%-8s\n" "-------------------------" "--------" "--------" "--------" "--------"

for MODEL in "${MODELS[@]}"; do
    RESULT="$OUT_DIR/classification_${MODEL}.json"
    if [ -f "$RESULT" ]; then
        python3 -c "
import json, sys
try:
    r = json.load(open('$RESULT'))
    f1 = f'{r.get(\"f1\", 0):.4f}'
    acc = f'{r.get(\"accuracy\", 0):.4f}'
    af1 = f'{r.get(\"attr_f1\", 0):.4f}'
    thr = f'{r.get(\"threshold\", 0.5):.2f}'
    print(f'  {\"$MODEL\":<25} | {f1:<8} {acc:<8} {af1:<8} {thr:<8}')
except:
    print(f'  {\"$MODEL\":<25} | parse error')
" 2>/dev/null || printf "  %-25s | parse error\n" "$MODEL"
    else
        printf "  %-25s | no results\n" "$MODEL"
    fi
done

# Transfer tier breakdown for main model
if [ -f "$OUT_DIR/causal_eval_results.json" ]; then
    echo ""
    echo "  Per-transfer-tier attribution (GuardLens):"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/causal_eval_results.json'))
te = r.get('transfer_eval', {})
for tier, data in te.items():
    gl = data.get('guardlens', {})
    dd = gl.get('deviation_drops', {}).get('15%', 0)
    flip = gl.get('flip_rates', {}).get('flip@15%', 0)
    print(f'    {tier:<25} DD@15%={dd:.3f}  Flip@15%={flip:.3f}')
" 2>/dev/null

    echo ""
    echo "  Per-supervision-tier attribution (GuardLens):"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/causal_eval_results.json'))
te = r.get('tier_eval', {})
for tier, data in te.items():
    gl = data.get('guardlens', {})
    dd = gl.get('deviation_drops', {}).get('15%', 0)
    flip = gl.get('flip_rates', {}).get('flip@15%', 0)
    print(f'    {tier:<25} DD@15%={dd:.3f}  Flip@15%={flip:.3f}')
" 2>/dev/null
fi

echo ""
echo "  Output files:"
for f in "$OUT_DIR"/causal_eval_results.json "$OUT_DIR"/causal_eval_results.tex \
         "$OUT_DIR"/classification_*.json; do
    [ -f "$f" ] && echo "    OK: $(basename $f)" || echo "    MISSING: $(basename $f)"
done

echo ""
echo "========================================================"