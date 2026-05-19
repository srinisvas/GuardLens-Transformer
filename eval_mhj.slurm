#!/bin/bash
#SBATCH --job-name=gl-mhj
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_mhj_%j.out
#SBATCH --error=logs/eval_mhj_%j.err

# ============================================================
# eval_mhj.slurm — Process MHJ data + evaluate
#
# Step 1: Convert MHJ to v11 schema (CPU)
# Step 2: Classification eval (GPU)
# Step 3: Causal attribution eval (GPU)
#
# Prereq: MHJ source file must exist. Set MHJ_SOURCE below.
# ============================================================

set -uo pipefail

# === EDIT THIS: path to raw MHJ data ===
MHJ_SOURCE="${MHJ_SOURCE:-$HOME/staging/dataset_gen_output/mhj_raw.jsonl}"
MHJ_OUTPUT="$HOME/staging/dataset_gen_output/mhj_external_test.jsonl"

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints"
OUT_DIR="$RESULTS_DIR/results"
DATAGEN_DIR="$HOME/projects/GuardLens-DataGen-V2"

CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT_DIR"

echo "========================================================"
echo "  MHJ External Evaluation"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "========================================================"

# ---- Step 1: Process MHJ ----
if [ -f "$MHJ_OUTPUT" ]; then
    echo "  MHJ already processed: $(wc -l < "$MHJ_OUTPUT") records"
elif [ -f "$MHJ_SOURCE" ]; then
    echo "  Processing MHJ from $MHJ_SOURCE..."
    cd "$DATAGEN_DIR"
    python3 mhj_loader.py \
        --input "$MHJ_SOURCE" \
        --output "$MHJ_OUTPUT" \
        --min-turns 2 \
    && echo "  Step 1: SUCCESS ($(wc -l < "$MHJ_OUTPUT") records)" \
    || { echo "  Step 1: FAILED"; exit 1; }
    cd "$SLURM_SUBMIT_DIR"
else
    echo "ERROR: No MHJ data found."
    echo "  Expected: $MHJ_SOURCE or $MHJ_OUTPUT"
    echo "  Download MHJ and set MHJ_SOURCE, then re-run."
    exit 1
fi

MHJ_COUNT=$(wc -l < "$MHJ_OUTPUT")
echo "  MHJ records: $MHJ_COUNT"

# ---- Step 2: Classification ----
echo ""
echo "=== MHJ Classification ==="
DET_CKPT="$CKPT_DIR/guardlens/best_detection.pt"
[ ! -f "$DET_CKPT" ] && DET_CKPT="$CKPT_DIR/guardlens/best.pt"

python -m guardlens.evaluate \
    --test-path "$MHJ_OUTPUT" \
    --checkpoint "$DET_CKPT" \
    --output "$OUT_DIR/mhj_classification.json" \
    --batch-size 4 \
    --device cuda \
2>"$OUT_DIR/mhj_classification.stderr"

if [ -f "$OUT_DIR/mhj_classification.json" ]; then
    python3 -c "
import json
r = json.load(open('$OUT_DIR/mhj_classification.json'))
print(f'MHJ Classification: F1={r.get(\"f1\",0):.4f} Acc={r.get(\"accuracy\",0):.4f} R={r.get(\"recall\",0):.4f}')
" 2>/dev/null
fi

# ---- Step 3: Causal Attribution ----
echo ""
echo "=== MHJ Causal Attribution ==="
ATTR_CKPT="$CKPT_DIR/guardlens/best_attribution.pt"
[ ! -f "$ATTR_CKPT" ] && ATTR_CKPT="$CKPT_DIR/guardlens/best.pt"

python -m guardlens.evaluation.eval_causal \
    --test-path "$MHJ_OUTPUT" \
    --checkpoint "$ATTR_CKPT" \
    --output "$OUT_DIR/mhj_causal_eval.json" \
    --methods guardlens surface_risk random \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda \
&& echo "Step 3: SUCCESS" || echo "Step 3: FAILED"

echo ""
echo "========================================================"
echo "  MHJ Eval Complete: $(date)"
echo "  Results:"
for f in mhj_classification.json mhj_causal_eval.json; do
    [ -f "$OUT_DIR/$f" ] && echo "    OK: $f" || echo "    MISSING: $f"
done
echo "========================================================"
