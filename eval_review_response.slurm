#!/bin/bash
#SBATCH --job-name=gl-review-resp
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_review_response_%j.out
#SBATCH --error=logs/eval_review_response_%j.err

# ============================================================
# eval_review_response.slurm — Reviewer response experiments
#
# Runs the three experiments needed for EMNLP rebuttal:
#
#   1. NoCF specificity decomposition (Item 2)
#      Decomposes NoCF's DD advantage into surface-risk-driven
#      vs non-surface-risk components. ~1 GPU-hour.
#
#   2. Top-k turn hit rate with random floor (Item 4)
#      Computes hit rate at top-1/2/3/5 with per-conversation
#      random floor for all attribution methods. ~0.5 GPU-hours.
#
#   3. External evaluator comparison (Item 8)
#      Self-eval vs ConversationDeBERTa/TurnLevel external eval
#      to quantify circularity gap. ~1 GPU-hour.
#
# Prerequisites:
#   - Trained checkpoints for guardlens, guardlens_no_cf,
#     conversation_deberta, turn_level
#   - eval_nocf_decomposition.py and eval_topk_turn_hitrate.py
#     copied into guardlens/evaluation/
#   - eval_full.slurm results (for external evaluator checkpoints)
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints"
TEST_DATA="$HOME/staging/dataset_gen_output/splits/test.jsonl"
OUT_DIR="$RESULTS_DIR/results/review_response"

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
echo "  Reviewer Response Experiments"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Preflight
if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Missing $TEST_DATA"; exit 1
fi
echo "  $(wc -l < "$TEST_DATA") records in $(basename "$TEST_DATA")"

# Checkpoint inventory
echo ""
echo "  Checkpoint check:"
GL_ATTR="$CKPT_DIR/guardlens/best_attribution.pt"
[ ! -f "$GL_ATTR" ] && GL_ATTR="$CKPT_DIR/guardlens/best.pt"

NOCF_ATTR="$CKPT_DIR/guardlens_no_cf/best_attribution.pt"
[ ! -f "$NOCF_ATTR" ] && NOCF_ATTR="$CKPT_DIR/guardlens_no_cf/best.pt"

EXT_DEBERTA="$CKPT_DIR/conversation_deberta/best_detection.pt"
[ ! -f "$EXT_DEBERTA" ] && EXT_DEBERTA="$CKPT_DIR/conversation_deberta/best.pt"

EXT_TURN="$CKPT_DIR/turn_level/best_detection.pt"
[ ! -f "$EXT_TURN" ] && EXT_TURN="$CKPT_DIR/turn_level/best.pt"

for label_ckpt in \
    "guardlens_attr:$GL_ATTR" \
    "nocf_attr:$NOCF_ATTR" \
    "ext_deberta:$EXT_DEBERTA" \
    "ext_turn:$EXT_TURN"; do
    label="${label_ckpt%%:*}"
    ckpt="${label_ckpt#*:}"
    if [ -f "$ckpt" ]; then
        echo "    OK: $label → $(basename "$ckpt")"
    else
        echo "    MISSING: $label → $ckpt"
    fi
done


# ============================================================
# 1. NoCF Specificity Decomposition
#
# Reviewer aeLx: NoCF outperforms full model on DD. Does the
# advantage come from over-attributing surface-risk tokens?
# ============================================================

echo ""
echo "========================================================"
echo "  1. NoCF Specificity Decomposition"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

if [ -f "$GL_ATTR" ] && [ -f "$NOCF_ATTR" ]; then
    python -m guardlens.evaluation.eval_nocf_decomposition \
        --test-path "$TEST_DATA" \
        --gl-checkpoint "$GL_ATTR" \
        --nocf-checkpoint "$NOCF_ATTR" \
        --output "$OUT_DIR/nocf_decomposition.json" \
        --top-k 0.05 0.10 0.15 0.20 \
        --batch-size 4 \
        --device cuda \
    2>"$OUT_DIR/nocf_decomposition.stderr" \
    && echo "1: SUCCESS" || echo "1: FAILED"
else
    echo "1: SKIPPED (missing guardlens or nocf checkpoint)"
fi

echo "1 end: $(date +%H:%M:%S)"


# ============================================================
# 2. Top-k Turn Hit Rate with Random Floor
#
# Reviewer aeLx: top-5 is forgiving with ~14 user turns and
# multiple causal turns. What is the margin above chance?
# ============================================================

echo ""
echo "========================================================"
echo "  2. Top-k Turn Hit Rate"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

if [ -f "$GL_ATTR" ]; then
    python -m guardlens.evaluation.eval_topk_turn_hitrate \
        --test-path "$TEST_DATA" \
        --checkpoint "$GL_ATTR" \
        --output "$OUT_DIR/topk_turn_hitrate.json" \
        --methods guardlens surface_risk attention random \
        --k-values 1 2 3 5 \
        --batch-size 4 \
        --device cuda \
    2>"$OUT_DIR/topk_turn_hitrate.stderr" \
    && echo "2: SUCCESS" || echo "2: FAILED"
else
    echo "2: SKIPPED (missing guardlens checkpoint)"
fi

echo "2 end: $(date +%H:%M:%S)"


# ============================================================
# 3. External Evaluator Comparison
#
# Reviewer aeLx: deletion eval measures internal sensitivity.
# Does the attribution hold up under an external evaluator?
# ============================================================

echo ""
echo "========================================================"
echo "  3. External Evaluator Comparison"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

EXT_CKPTS=""
[ -f "$EXT_DEBERTA" ] && EXT_CKPTS="$EXT_DEBERTA"
[ -f "$EXT_TURN" ] && EXT_CKPTS="$EXT_CKPTS $EXT_TURN"

if [ -f "$GL_ATTR" ] && [ -n "$EXT_CKPTS" ]; then
    python -m guardlens.evaluation.eval_external \
        --test-path "$TEST_DATA" \
        --attribution-checkpoint "$GL_ATTR" \
        --external-checkpoints $EXT_CKPTS \
        --output "$OUT_DIR/external_eval.json" \
        --top-k 0.05 0.10 0.15 0.20 \
        --batch-size 4 \
        --device cuda \
    2>"$OUT_DIR/external_eval.stderr" \
    && echo "3: SUCCESS" || echo "3: FAILED"
else
    echo "3: SKIPPED (missing guardlens or external checkpoints)"
fi

echo "3 end: $(date +%H:%M:%S)"


# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================================"
echo "  Reviewer Response Summary"
echo "  $(date)"
echo "========================================================"

# Output file inventory
echo ""
echo "  Output files:"
for f in nocf_decomposition.json topk_turn_hitrate.json external_eval.json; do
    if [ -f "$OUT_DIR/$f" ]; then
        echo "    OK: $f"
    else
        echo "    MISSING: $f"
    fi
done

# --- 1. NoCF decomposition summary ---
if [ -f "$OUT_DIR/nocf_decomposition.json" ]; then
    echo ""
    echo "  NoCF Decomposition @ 15%:"
    python3 -c "
import json, sys
r = json.load(open('$OUT_DIR/nocf_decomposition.json'))
gl = r.get('guardlens', {}).get('15%', {})
nc = r.get('nocf', {}).get('15%', {})
print(f'    GuardLens  DD_all={gl.get(\"dd_all\",0):.4f}  DD_nonSR={gl.get(\"dd_non_sr\",0):.4f}  DD_SR={gl.get(\"dd_sr\",0):.4f}  SR_frac={gl.get(\"sr_fraction\",0):.3f}')
print(f'    NoCF       DD_all={nc.get(\"dd_all\",0):.4f}  DD_nonSR={nc.get(\"dd_non_sr\",0):.4f}  DD_SR={nc.get(\"dd_sr\",0):.4f}  SR_frac={nc.get(\"sr_fraction\",0):.3f}')
d_all = nc.get('dd_all',0) - gl.get('dd_all',0)
d_non = nc.get('dd_non_sr',0) - gl.get('dd_non_sr',0)
d_sr  = nc.get('dd_sr',0) - gl.get('dd_sr',0)
print(f'    Delta      DD_all={d_all:+.4f}  DD_nonSR={d_non:+.4f}  DD_SR={d_sr:+.4f}')
if d_non <= 0 and d_sr > 0:
    print(f'    VERDICT: CF loss validated (NoCF advantage is surface-risk-driven)')
elif d_non > 0:
    print(f'    VERDICT: CF loss may hurt non-surface-risk attribution')
" 2>/dev/null || echo "    parse error"
fi

# --- 2. Top-k hit rate summary ---
if [ -f "$OUT_DIR/topk_turn_hitrate.json" ]; then
    echo ""
    echo "  Top-k Turn Hit Rate (margin above random):"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/topk_turn_hitrate.json'))
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

# --- 3. External eval gap summary ---
if [ -f "$OUT_DIR/external_eval.json" ]; then
    echo ""
    echo "  External Evaluator Gap @ 15%:"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/external_eval.json'))
se = r.get('self_eval', {}).get('15%', {})
print(f'    Self-eval:  DD={se.get(\"deviation_drop\",0):.3f}  Flip={se.get(\"flip_rate\",0):.3f}')
for name, data in r.get('external_eval', {}).items():
    ext = data.get('15%', {})
    dd_gap = se.get('deviation_drop',0) - ext.get('deviation_drop',0)
    print(f'    {name:<25} DD={ext.get(\"deviation_drop\",0):.3f}  Flip={ext.get(\"flip_rate\",0):.3f}  gap={dd_gap:+.3f}')
" 2>/dev/null || echo "    parse error"
fi

echo ""
echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"
