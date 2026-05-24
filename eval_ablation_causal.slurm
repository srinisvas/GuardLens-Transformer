#!/bin/bash
#SBATCH --job-name=gl-eval-ablation
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_ablation_%j.out
#SBATCH --error=logs/eval_ablation_%j.err

# ============================================================
# eval_ablation_causal.slurm — Full ablation evaluation
#
# Computes the missing ablation metrics for the paper table:
#   Det F1 | Token F1 | DD@15% | Flip@15% | Boundary FPR | Utility
#
# Runs per ablation variant (guardlens, no_fusion, no_cf):
#   1. Causal attribution eval (DD@k, Flip@k, Token F1)
#   2. Boundary stress test (FPR on boundary cases)
#   3. Utility = DD@15% - λ*BoundaryFPR (computed in summary)
#
# Det F1 is pulled from eval_full classification results.
# Re-runs full GuardLens for a consistent comparison row.
#
# Prerequisite: trained checkpoints + eval_full classification
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints"
TEST_DATA="$HOME/staging/dataset_gen_output/splits/test.jsonl"
BOUNDARY_DATA="$HOME/staging/dataset_gen_output/benign_boundary.jsonl"
OUT_DIR="$RESULTS_DIR/results/ablation_causal"

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
echo "  Ablation Causal Attribution + Boundary FPR"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Preflight: check test data
for f in "$TEST_DATA"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"; exit 1
    fi
    echo "  $(wc -l < "$f") records in $(basename $f)"
done

# Models with attribution heads that need causal eval.
# guardlens is included for a consistent comparison row.
# turn_level and conversation_deberta have no attribution head,
# so DD/Flip/boundary FPR are not meaningful for them.
ABLATION_MODELS=("guardlens" "guardlens_no_fusion" "guardlens_no_cf")

# Verify checkpoints exist
echo ""
echo "  Checkpoint check:"
ALL_OK=true
for MODEL in "${ABLATION_MODELS[@]}"; do
    # Prefer best_attribution.pt for causal eval, fall back to best.pt
    ATTR_CKPT="$CKPT_DIR/$MODEL/best_attribution.pt"
    [ ! -f "$ATTR_CKPT" ] && ATTR_CKPT="$CKPT_DIR/$MODEL/best.pt"

    DET_CKPT="$CKPT_DIR/$MODEL/best_detection.pt"
    [ ! -f "$DET_CKPT" ] && DET_CKPT="$CKPT_DIR/$MODEL/best.pt"

    if [ -f "$ATTR_CKPT" ]; then
        echo "    OK: $MODEL attribution → $(basename $ATTR_CKPT)"
    else
        echo "    MISSING: $MODEL (no checkpoint found)"
        ALL_OK=false
    fi

    if [ -f "$DET_CKPT" ]; then
        echo "    OK: $MODEL detection   → $(basename $DET_CKPT)"
    else
        echo "    MISSING: $MODEL detection (no checkpoint found)"
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some checkpoints missing. Will skip those models."
fi


# ============================================================
# 1. Causal Attribution Eval per Ablation Variant
# ============================================================

echo ""
echo "========================================================"
echo "  1. Causal Attribution (DD@k, Flip@k, Nec, Suf)"
echo "========================================================"

for MODEL in "${ABLATION_MODELS[@]}"; do
    ATTR_CKPT="$CKPT_DIR/$MODEL/best_attribution.pt"
    [ ! -f "$ATTR_CKPT" ] && ATTR_CKPT="$CKPT_DIR/$MODEL/best.pt"

    if [ ! -f "$ATTR_CKPT" ]; then
        echo "  $MODEL: SKIPPED (no checkpoint)"
        continue
    fi

    echo ""
    echo "  === $MODEL causal attribution ==="
    echo "  Checkpoint: $(basename $ATTR_CKPT)"
    echo "  Start: $(date +%H:%M:%S)"

    python -m guardlens.evaluation.eval_causal \
        --test-path "$TEST_DATA" \
        --checkpoint "$ATTR_CKPT" \
        --output "$OUT_DIR/causal_${MODEL}.json" \
        --methods guardlens surface_risk random \
        --top-k 0.05 0.10 0.15 0.20 \
        --batch-size 4 \
        --device cuda \
    2>"$OUT_DIR/causal_${MODEL}.stderr" \
    && echo "  $MODEL causal: SUCCESS" || echo "  $MODEL causal: FAILED"

    echo "  $MODEL end: $(date +%H:%M:%S)"
done


# ============================================================
# 2. Boundary Stress Test per Ablation Variant
# ============================================================

echo ""
echo "========================================================"
echo "  2. Boundary Stress Test (FPR)"
echo "========================================================"

if [ ! -f "$BOUNDARY_DATA" ]; then
    echo "  SKIPPED: No boundary data at $BOUNDARY_DATA"
else
    echo "  Boundary records: $(wc -l < "$BOUNDARY_DATA")"

    for MODEL in "${ABLATION_MODELS[@]}"; do
        DET_CKPT="$CKPT_DIR/$MODEL/best_detection.pt"
        [ ! -f "$DET_CKPT" ] && DET_CKPT="$CKPT_DIR/$MODEL/best.pt"

        if [ ! -f "$DET_CKPT" ]; then
            echo "  $MODEL: SKIPPED (no checkpoint)"
            continue
        fi

        echo ""
        echo "  === $MODEL boundary stress ==="
        echo "  Checkpoint: $(basename $DET_CKPT)"
        echo "  Start: $(date +%H:%M:%S)"

        python -m guardlens.evaluation.eval_boundary_stress \
            --boundary-files "$BOUNDARY_DATA" \
            --checkpoint "$DET_CKPT" \
            --output "$OUT_DIR/boundary_${MODEL}.json" \
            --device cuda \
        2>"$OUT_DIR/boundary_${MODEL}.stderr" \
        && echo "  $MODEL boundary: SUCCESS" || echo "  $MODEL boundary: FAILED"

        echo "  $MODEL end: $(date +%H:%M:%S)"
    done
fi


# ============================================================
# Summary: Full Ablation Table
#
# Columns: Det F1 | Token F1 | DD@15% | Flip@15 | BndFPR | Utility
# Utility = DD@15% - λ*BoundaryFPR  (λ=1.0)
#
# Det F1 is pulled from eval_full classification results.
# Everything else comes from this job's outputs.
# ============================================================

CLASSIFICATION_DIR="$RESULTS_DIR/results"  # eval_full outputs

echo ""
echo "========================================================"
echo "  Ablation Table (paper-ready)"
echo "========================================================"

python3 << 'SUMMARY_EOF'
import json, os, sys

results_dir = os.environ.get("RESULTS_DIR", "")
out_dir = os.path.join(results_dir, "results", "ablation_causal")
cls_dir = os.path.join(results_dir, "results")

models = ["guardlens", "guardlens_no_fusion", "guardlens_no_cf"]
display = {
    "guardlens": "Full GuardLens",
    "guardlens_no_fusion": "No Fusion",
    "guardlens_no_cf": "No Counterfactual",
}

LAMBDA = 1.0

# Header
print()
print(f"  {'Model':<20} | {'Det F1':>8} {'Tok F1':>8} {'DD@15%':>8} {'Flip@15':>8} {'BndFPR':>8} {'Utility':>8}")
print(f"  {'-'*20}-+-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*8}")

rows = {}
for model in models:
    name = display[model]

    # --- Detection F1 from eval_full classification ---
    det_f1 = "N/A"
    cls_path = os.path.join(cls_dir, f"classification_{model}.json")
    try:
        cls = json.load(open(cls_path))
        det_f1_val = cls.get("f1", 0)
        det_f1 = f"{det_f1_val:.3f}"
    except Exception:
        pass

    # --- Causal metrics from this job ---
    dd_val = flip_val = tf1_val = None
    dd = flip = tf1 = "N/A"
    causal_path = os.path.join(out_dir, f"causal_{model}.json")
    try:
        r = json.load(open(causal_path))
        ft = r.get("full_test", r)
        gl = ft.get("guardlens", {})
        dd_val = gl.get("deviation_drops", {}).get("15%", None)
        flip_val = gl.get("flip_rates", {}).get("flip@15%", None)
        tf1_val = gl.get("token_f1", None)
        if dd_val is not None: dd = f"{dd_val:.3f}"
        if flip_val is not None: flip = f"{flip_val:.3f}"
        if tf1_val is not None: tf1 = f"{tf1_val:.3f}"
    except Exception:
        pass

    # --- Boundary FPR from this job ---
    bfpr_val = None
    bfpr = "N/A"
    boundary_path = os.path.join(out_dir, f"boundary_{model}.json")
    try:
        b = json.load(open(boundary_path))
        bfpr_val = b.get("false_positive_rate", None)
        if bfpr_val is not None: bfpr = f"{bfpr_val:.3f}"
    except Exception:
        pass

    # --- Utility = DD@15% - λ * BoundaryFPR ---
    utility = "N/A"
    if dd_val is not None and bfpr_val is not None:
        util_val = dd_val - LAMBDA * bfpr_val
        utility = f"{util_val:.3f}"

    print(f"  {name:<20} | {det_f1:>8} {tf1:>8} {dd:>8} {flip:>8} {bfpr:>8} {utility:>8}")

    rows[model] = {
        "det_f1": det_f1, "token_f1": tf1, "dd_15": dd,
        "flip_15": flip, "boundary_fpr": bfpr, "utility": utility,
    }

# Save consolidated JSON for downstream use
consolidated_path = os.path.join(out_dir, "ablation_table.json")
try:
    with open(consolidated_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Saved: ablation_table.json")
except Exception as e:
    print(f"\n  WARNING: Could not save consolidated JSON: {e}")

SUMMARY_EOF


# Boundary FPR per-source breakdown (for appendix)
echo ""
echo "  Boundary FPR per source:"
for MODEL in "${ABLATION_MODELS[@]}"; do
    BOUNDARY="$OUT_DIR/boundary_${MODEL}.json"
    if [ -f "$BOUNDARY" ]; then
        echo ""
        echo "  === $MODEL ==="
        python3 -c "
import json
b = json.load(open('$BOUNDARY'))
pf = b.get('per_family', b.get('per_source', {}))
for src, data in sorted(pf.items()):
    n = data.get('n', data.get('total', 0))
    fp = data.get('fp', data.get('false_positives', 0))
    fpr = data.get('fpr', data.get('false_positive_rate', 0))
    print(f'    {src:<25} FPR={fpr:.3f} ({fp}/{n})')
" 2>/dev/null || echo "    parse error"
    fi
done

# Output file inventory
echo ""
echo "  Output files:"
for MODEL in "${ABLATION_MODELS[@]}"; do
    for f in "causal_${MODEL}.json" "boundary_${MODEL}.json"; do
        if [ -f "$OUT_DIR/$f" ]; then
            echo "    OK: $f"
        else
            echo "    MISSING: $f"
        fi
    done
done
[ -f "$OUT_DIR/ablation_table.json" ] && echo "    OK: ablation_table.json" || echo "    MISSING: ablation_table.json"

echo ""
echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"
