#!/bin/bash
#SBATCH --job-name=gl-eval-full
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_full_%j.out
#SBATCH --error=logs/eval_full_%j.err

# ============================================================
# eval_full.slurm — Complete evaluation from scratch
#
# Runs all evaluations in one job with correct checkpoints:
#   1. Surface risk FPR comparison (no GPU, fast)
#   2. Classification eval (all 5 models, best_detection.pt)
#   3. Core causal eval (best_attribution.pt, all methods)
#   4. Boundary stress test
#   5. MHJ external eval (if available)
#   6. Implicit/explicit subset analysis
#
# Uses:
#   best_detection.pt   → classification metrics
#   best_attribution.pt → causal attribution metrics
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints"
TEST_DATA="$HOME/staging/dataset_gen_output/splits/test.jsonl"
FINAL_DATA="$HOME/staging/dataset_gen_output/final_dataset.jsonl"
BOUNDARY_DATA="$HOME/staging/dataset_gen_output/benign_boundary.jsonl"
MHJ_DATA="$HOME/staging/dataset_gen_output/mhj_external_test.jsonl"
OUT_DIR="$RESULTS_DIR/results"

CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1
export TMPDIR="$HOME/work/.tmp"
mkdir -p "$TMPDIR" "$OUT_DIR" logs

cd "$SLURM_SUBMIT_DIR"

echo "========================================================"
echo "  GuardLens Full Evaluation (from scratch)"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Preflight
for f in "$TEST_DATA" "$FINAL_DATA"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"; exit 1
    fi
    echo "  $(wc -l < "$f") records in $(basename $f)"
done

echo ""
echo "  Checkpoints:"
for m in guardlens guardlens_no_fusion guardlens_no_cf turn_level conversation_deberta; do
    for ckpt in best_detection.pt best_attribution.pt best.pt; do
        f="$CKPT_DIR/$m/$ckpt"
        [ -f "$f" ] && echo "    OK: $m/$ckpt" || true
    done
done


# ============================================================
# 1. Surface Risk FPR Comparison (no GPU, fast)
# ============================================================

echo ""
echo "========================================================"
echo "  1. Surface Risk FPR Comparison"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

BOUNDARY_FLAG=""
[ -f "$BOUNDARY_DATA" ] && BOUNDARY_FLAG="--boundary-path $BOUNDARY_DATA"

python -m guardlens.evaluation.eval_surface_risk_fpr \
    --test-path "$TEST_DATA" \
    $BOUNDARY_FLAG \
    --output "$OUT_DIR/surface_risk_fpr.json" \
&& echo "1: SUCCESS" || echo "1: FAILED"


# ============================================================
# 2. Classification Eval (all models, best_detection.pt)
# ============================================================

echo ""
echo "========================================================"
echo "  2. Classification Evaluation (all models)"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

MODELS=("guardlens" "guardlens_no_fusion" "guardlens_no_cf" "turn_level" "conversation_deberta")

for MODEL in "${MODELS[@]}"; do
    # Use best_detection.pt, fall back to best.pt
    CKPT="$CKPT_DIR/$MODEL/best_detection.pt"
    [ ! -f "$CKPT" ] && CKPT="$CKPT_DIR/$MODEL/best.pt"

    if [ ! -f "$CKPT" ]; then
        echo "  $MODEL: SKIPPED (no checkpoint)"
        continue
    fi

    echo ""
    echo "  === $MODEL ==="
    python -m guardlens.evaluate \
        --test-path "$TEST_DATA" \
        --checkpoint "$CKPT" \
        --output "$OUT_DIR/classification_${MODEL}.json" \
        --batch-size 4 \
        --device cuda \
    2>"$OUT_DIR/classification_${MODEL}.stderr" \
    && echo "  $MODEL: SUCCESS" || echo "  $MODEL: FAILED"
done


# ============================================================
# 3. Core Causal Attribution Eval (best_attribution.pt)
# ============================================================

echo ""
echo "========================================================"
echo "  3. Core Causal Attribution Evaluation"
echo "  Using: best_attribution.pt"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

ATTR_CKPT="$CKPT_DIR/guardlens/best_attribution.pt"
[ ! -f "$ATTR_CKPT" ] && ATTR_CKPT="$CKPT_DIR/guardlens/best.pt"

python -m guardlens.evaluation.eval_causal \
    --test-path "$TEST_DATA" \
    --checkpoint "$ATTR_CKPT" \
    --output "$OUT_DIR/causal_eval_results.json" \
    --methods guardlens attention integrated_gradients grad_x_input surface_risk random \
    --top-k 0.05 0.10 0.15 0.20 \
    --tier-eval \
    --transfer-eval \
    --batch-size 4 \
    --device cuda \
&& echo "3: SUCCESS" || echo "3: FAILED"

echo "3 end: $(date +%H:%M:%S)"


# ============================================================
# 4. Boundary Stress Test
# ============================================================

echo ""
echo "========================================================"
echo "  4. Boundary Stress Test"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

if [ -f "$BOUNDARY_DATA" ]; then
    python -m guardlens.evaluation.eval_boundary_stress \
        --boundary-files "$BOUNDARY_DATA" \
        --checkpoint "$CKPT_DIR/guardlens/best_detection.pt" \
        --output "$OUT_DIR/boundary_stress.json" \
        --device cuda \
    && echo "4: SUCCESS" || echo "4: FAILED"
else
    echo "4: SKIPPED (no boundary data)"
fi


# ============================================================
# 5. MHJ External Eval
# ============================================================

echo ""
echo "========================================================"
echo "  5. MHJ External Evaluation"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

if [ -f "$MHJ_DATA" ]; then
    echo "  MHJ records: $(wc -l < "$MHJ_DATA")"

    # Classification
    python -m guardlens.evaluate \
        --test-path "$MHJ_DATA" \
        --checkpoint "$CKPT_DIR/guardlens/best_detection.pt" \
        --output "$OUT_DIR/mhj_classification.json" \
        --batch-size 4 \
        --device cuda \
    2>"$OUT_DIR/mhj_classification.stderr"

    if [ -f "$OUT_DIR/mhj_classification.json" ]; then
        python3 -c "
import json
r = json.load(open('$OUT_DIR/mhj_classification.json'))
print(f'  MHJ: F1={r.get(\"f1\",0):.4f} Acc={r.get(\"accuracy\",0):.4f} R={r.get(\"recall\",0):.4f}')
" 2>/dev/null
    fi

    # Causal eval
    python -m guardlens.evaluation.eval_causal \
        --test-path "$MHJ_DATA" \
        --checkpoint "$ATTR_CKPT" \
        --output "$OUT_DIR/mhj_causal_eval.json" \
        --methods guardlens surface_risk random \
        --top-k 0.05 0.10 0.15 0.20 \
        --batch-size 4 \
        --device cuda \
    && echo "5: SUCCESS" || echo "5: FAILED"
else
    echo "5: SKIPPED (no MHJ data at $MHJ_DATA)"
    echo "  To generate: python mhj_loader.py --input <mhj.jsonl> --output $MHJ_DATA"
fi


# ============================================================
# 6. Implicit/Explicit Subset Analysis (best_attribution.pt)
# ============================================================

echo ""
echo "========================================================"
echo "  6. Implicit/Explicit Subset Analysis"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_implicit_explicit \
    --test-path "$TEST_DATA" \
    --checkpoint "$ATTR_CKPT" \
    --output "$OUT_DIR/implicit_explicit_eval.json" \
    --methods guardlens surface_risk integrated_gradients attention random \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda \
&& echo "6: SUCCESS" || echo "6: FAILED"


# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================================"
echo "  Full Evaluation Complete: $(date)"
echo "========================================================"

echo ""
echo "  Output files:"
for f in surface_risk_fpr.json \
         classification_guardlens.json classification_guardlens_no_fusion.json \
         classification_guardlens_no_cf.json classification_turn_level.json \
         classification_conversation_deberta.json \
         causal_eval_results.json causal_eval_results.tex \
         boundary_stress.json \
         mhj_classification.json mhj_causal_eval.json \
         implicit_explicit_eval.json implicit_explicit_eval.tex; do
    if [ -f "$OUT_DIR/$f" ]; then
        echo "    OK: $f"
    else
        echo "    MISSING: $f"
    fi
done

# Print classification comparison
echo ""
echo "  Classification Comparison:"
printf "  %-25s | %-8s %-8s %-8s %-8s %-8s\n" "Model" "F1" "Acc" "AttrF1" "PivMal" "Thresh"
printf "  %-25s-+-%-8s-%-8s-%-8s-%-8s-%-8s\n" "-------------------------" "--------" "--------" "--------" "--------" "--------"

for MODEL in guardlens guardlens_no_fusion guardlens_no_cf turn_level conversation_deberta; do
    RESULT="$OUT_DIR/classification_${MODEL}.json"
    if [ -f "$RESULT" ]; then
        python3 -c "
import json
r = json.load(open('$RESULT'))
f1 = f'{r.get(\"f1\",0):.4f}'
acc = f'{r.get(\"accuracy\",0):.4f}'
af1 = f'{r.get(\"attr_f1\",0):.4f}'
piv = f'{r.get(\"pivot_accuracy_malicious\",0):.4f}'
thr = f'{r.get(\"threshold\",0.5):.2f}'
print(f'  {\"$MODEL\":<25} | {f1:<8} {acc:<8} {af1:<8} {piv:<8} {thr:<8}')
" 2>/dev/null || printf "  %-25s | parse error\n" "$MODEL"
    else
        printf "  %-25s | no results\n" "$MODEL"
    fi
done

# Print causal eval summary
if [ -f "$OUT_DIR/causal_eval_results.json" ]; then
    echo ""
    echo "  Causal Attribution (best_attribution.pt):"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/causal_eval_results.json'))
full = r.get('full_test', {})
for method, data in full.items():
    dd = data.get('deviation_drops', {}).get('15%', 0)
    flip = data.get('flip_rates', {}).get('flip@15%', 0)
    tf1 = data.get('token_f1', 0)
    print(f'    {method:<25} DD@15%={dd:.3f}  Flip@15%={flip:.3f}  TokF1={tf1:.3f}')
" 2>/dev/null
fi

echo ""
echo "========================================================"
