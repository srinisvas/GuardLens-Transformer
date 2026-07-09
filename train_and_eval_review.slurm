#!/bin/bash
#SBATCH --job-name=gl-train-review
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=logs/train_review_%j.out
#SBATCH --error=logs/train_review_%j.err

# ============================================================
# Retrain all 5 models + run reviewer response experiments.
#
# Training (~10-13h, same pattern as train_all.slurm):
#   1. GuardLens (full)                ~3-4 hours
#   2. GuardLens-NoFusion (ablation)   ~2-3 hours
#   3. GuardLens-NoCF (ablation)       ~2 hours
#   4. Turn-level classifier           ~2 hours
#   5. ConversationDeBERTa             ~1.5 hours
#
# Evaluation (~2.5h):
#   6. NoCF specificity decomposition
#   7. Top-k turn hit rate with random floor
#   8. External evaluator comparison
#
# Prerequisites:
#   - eval_nocf_decomposition.py and eval_topk_turn_hitrate.py
#     copied into guardlens/evaluation/
#
# Usage:
#   mkdir -p logs
#   sbatch train_and_eval_review.slurm
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
REVIEW_OUT="$HOME/work/results/guardlens_v11/results/review_response"

# ---- Environment setup ----
# If the conda env doesn't exist, create it. This requires
# network access (pip install). If compute nodes lack internet,
# run the creation block on a login node first:
#
#   CONDA_BASE=$(conda info --base)
#   source "$CONDA_BASE/etc/profile.d/conda.sh"
#   conda create -p $HOME/work/conda_envs/guardlens_train python=3.10 -y
#   conda activate $HOME/work/conda_envs/guardlens_train
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
#   pip install transformers numpy scipy
#
set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -d "$CONDA_ENV" ]; then
    echo "  Conda env not found at $CONDA_ENV — creating..."
    conda create -p "$CONDA_ENV" python=3.10 -y \
        || { echo "ERROR: conda create failed"; exit 1; }
    conda activate "$CONDA_ENV"

    echo "  Installing PyTorch (CUDA 11.8)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118 \
        || { echo "ERROR: PyTorch install failed"; exit 1; }

    echo "  Installing transformers, numpy, scipy..."
    pip install transformers numpy scipy \
        || { echo "ERROR: pip install failed"; exit 1; }

    echo "  Conda env created successfully"
else
    conda activate "$CONDA_ENV"
fi
set -u

# Verify conda activated correctly
echo "  Python: $(which python) ($(python --version 2>&1))"
if python -c "import torch; import transformers; import dataclasses" 2>/dev/null; then
    echo "  Conda env OK"
    python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
else
    echo "ERROR: Conda env broken or failed to activate."
    echo "  Got: $(which python)"
    echo "  Expected: $CONDA_ENV"
    echo "  Try recreating manually on a login node (see comments above)."
    exit 1
fi

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
mkdir -p "$TMPDIR" "$REVIEW_OUT" logs

# Download backbone if not cached. Must happen before setting HF_HUB_OFFLINE=1.
BACKBONE="microsoft/deberta-v3-base"
if python -c "
from transformers import AutoModel, AutoTokenizer
import os
cache = os.environ.get('TRANSFORMERS_CACHE', '')
# Try loading offline first
os.environ['HF_HUB_OFFLINE'] = '1'
try:
    AutoTokenizer.from_pretrained('$BACKBONE')
    AutoModel.from_pretrained('$BACKBONE')
    print('cached')
except Exception:
    print('missing')
" 2>/dev/null | grep -q "cached"; then
    echo "  HF backbone cached: $BACKBONE"
else
    echo "  Downloading HF backbone: $BACKBONE"
    python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('$BACKBONE')
AutoModel.from_pretrained('$BACKBONE')
print('  Download complete')
" || { echo "ERROR: Failed to download $BACKBONE (no network?)"; exit 1; }
fi

export HF_HUB_OFFLINE=1

cd "$PROJECT_DIR"

echo "========================================================"
echo "  GuardLens: Retrain + Reviewer Response Experiments"
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


# ============================================================
# TRAINING (same pattern as train_all.slurm)
# ============================================================

MODELS=("guardlens" "guardlens_no_fusion" "guardlens_no_cf" "turn_level" "conversation_deberta")
EPOCHS=(25 20 20 20 20)
BATCH=(4 4 4 4 2)
MAX_TURNS=(32 32 32 32 32)
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

# ---- Training summary table ----
echo ""
echo "========================================================"
echo "  Training Summary"
echo "========================================================"

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

# ---- Checkpoint inventory ----
echo ""
echo "  Checkpoints produced:"
for MODEL in "${MODELS[@]}"; do
    files=$(ls "$BASE_OUTPUT/$MODEL"/*.pt 2>/dev/null | xargs -I{} basename {} | tr '\n' '  ')
    if [ -n "$files" ]; then
        echo "    $MODEL: $files"
    else
        echo "    $MODEL: (none)"
    fi
done


# ============================================================
# REVIEWER RESPONSE EXPERIMENTS
# ============================================================

# Resolve checkpoints
GL_ATTR=$(ls "$BASE_OUTPUT/guardlens/best_attribution.pt" 2>/dev/null \
       || ls "$BASE_OUTPUT/guardlens/best.pt" 2>/dev/null \
       || echo "")

NOCF_ATTR=$(ls "$BASE_OUTPUT/guardlens_no_cf/best_attribution.pt" 2>/dev/null \
         || ls "$BASE_OUTPUT/guardlens_no_cf/best.pt" 2>/dev/null \
         || echo "")

EXT_DEBERTA=$(ls "$BASE_OUTPUT/conversation_deberta/best_detection.pt" 2>/dev/null \
           || ls "$BASE_OUTPUT/conversation_deberta/best.pt" 2>/dev/null \
           || echo "")

EXT_TURN=$(ls "$BASE_OUTPUT/turn_level/best_detection.pt" 2>/dev/null \
        || ls "$BASE_OUTPUT/turn_level/best.pt" 2>/dev/null \
        || echo "")

echo ""
echo "========================================================"
echo "  Resolved checkpoints for eval:"
echo "    guardlens:    ${GL_ATTR:-(none)}"
echo "    nocf:         ${NOCF_ATTR:-(none)}"
echo "    deberta_flat: ${EXT_DEBERTA:-(none)}"
echo "    turn_level:   ${EXT_TURN:-(none)}"
echo "========================================================"

if [ -z "$GL_ATTR" ]; then
    echo ""
    echo "  FATAL: GuardLens checkpoint missing. Skipping evaluations."
    echo "  Done: $(date)"
    exit 1
fi


# ---- 6. NoCF Specificity Decomposition ----

echo ""
echo "========================================================"
echo "  6. NoCF Specificity Decomposition"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

if [ -n "$NOCF_ATTR" ]; then
    python -m guardlens.evaluation.eval_nocf_decomposition \
        --test-path "$TEST_PATH" \
        --gl-checkpoint "$GL_ATTR" \
        --nocf-checkpoint "$NOCF_ATTR" \
        --output "$REVIEW_OUT/nocf_decomposition.json" \
        --top-k 0.05 0.10 0.15 0.20 \
        --batch-size 4 \
        --device cuda \
    && echo "  6: SUCCESS ($(date +%H:%M:%S))" \
    || echo "  6: FAILED ($(date +%H:%M:%S))"
else
    echo "  6: SKIPPED (no NoCF checkpoint)"
fi


# ---- 7. Top-k Turn Hit Rate ----

echo ""
echo "========================================================"
echo "  7. Top-k Turn Hit Rate"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_topk_turn_hitrate \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$REVIEW_OUT/topk_turn_hitrate.json" \
    --methods guardlens surface_risk attention random \
    --k-values 1 2 3 5 \
    --batch-size 4 \
    --device cuda \
&& echo "  7: SUCCESS ($(date +%H:%M:%S))" \
|| echo "  7: FAILED ($(date +%H:%M:%S))"


# ---- 8. External Evaluator Comparison ----

echo ""
echo "========================================================"
echo "  8. External Evaluator Comparison"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

EXT_CKPTS=""
[ -n "$EXT_DEBERTA" ] && EXT_CKPTS="$EXT_DEBERTA"
[ -n "$EXT_TURN" ] && EXT_CKPTS="$EXT_CKPTS $EXT_TURN"

if [ -n "$EXT_CKPTS" ]; then
    python -m guardlens.evaluation.eval_external \
        --test-path "$TEST_PATH" \
        --attribution-checkpoint "$GL_ATTR" \
        --external-checkpoints $EXT_CKPTS \
        --output "$REVIEW_OUT/external_eval.json" \
        --top-k 0.05 0.10 0.15 0.20 \
        --batch-size 4 \
        --device cuda \
    && echo "  8: SUCCESS ($(date +%H:%M:%S))" \
    || echo "  8: FAILED ($(date +%H:%M:%S))"
else
    echo "  8: SKIPPED (no external evaluator checkpoints)"
fi


# ============================================================
# Final Summary
# ============================================================

echo ""
echo "========================================================"
echo "  Reviewer Response Results"
echo "  $(date)"
echo "========================================================"

echo ""
echo "  Eval outputs:"
for f in nocf_decomposition.json topk_turn_hitrate.json external_eval.json; do
    if [ -f "$REVIEW_OUT/$f" ]; then
        echo "    OK: $f"
    else
        echo "    MISSING: $f"
    fi
done

if [ -f "$REVIEW_OUT/nocf_decomposition.json" ]; then
    echo ""
    echo "  NoCF Decomposition @ 15%:"
    python3 -c "
import json
r = json.load(open('$REVIEW_OUT/nocf_decomposition.json'))
gl = r.get('guardlens', {}).get('15%', {})
nc = r.get('nocf', {}).get('15%', {})
print(f'    GuardLens  DD_all={gl.get(\"dd_all\",0):.4f}  DD_nonSR={gl.get(\"dd_non_sr\",0):.4f}  DD_SR={gl.get(\"dd_sr\",0):.4f}  SR_frac={gl.get(\"sr_fraction\",0):.3f}')
print(f'    NoCF       DD_all={nc.get(\"dd_all\",0):.4f}  DD_nonSR={nc.get(\"dd_non_sr\",0):.4f}  DD_SR={nc.get(\"dd_sr\",0):.4f}  SR_frac={nc.get(\"sr_fraction\",0):.3f}')
d_non = nc.get('dd_non_sr',0) - gl.get('dd_non_sr',0)
d_sr  = nc.get('dd_sr',0) - gl.get('dd_sr',0)
if d_non <= 0 and d_sr > 0:
    print('    VERDICT: CF loss validated (NoCF advantage is surface-risk-driven)')
elif d_non > 0:
    print('    VERDICT: CF loss may hurt non-surface-risk attribution')
" 2>/dev/null || echo "    parse error"
fi

if [ -f "$REVIEW_OUT/topk_turn_hitrate.json" ]; then
    echo ""
    echo "  Top-k Turn Hit Rate (margin above random):"
    python3 -c "
import json
r = json.load(open('$REVIEW_OUT/topk_turn_hitrate.json'))
results = r.get('results', {})
for method in ['guardlens', 'surface_risk', 'attention', 'random']:
    m = results.get(method, {})
    parts = []
    for k in [1, 2, 3, 5]:
        d = m.get(f'top_{k}', {})
        margin = d.get('margin', 0)
        hit = d.get('hit_rate', 0)
        parts.append(f'k={k}: {hit:.3f} ({margin:+.3f})')
    print(f'    {method:<15} {\"  \".join(parts)}')
" 2>/dev/null || echo "    parse error"
fi

echo ""
echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"
