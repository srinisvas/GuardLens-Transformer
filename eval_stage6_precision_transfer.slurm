#!/bin/bash
#SBATCH --job-name=guardlens_precision
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/eval_precision_%j.out
#SBATCH --error=logs/eval_precision_%j.err

# Runs:
#   6a. Hard negative attribution precision + minimality curve
#   6b. Cross-model transfer (LlamaGuard, no API cost)
#       Set OPENAI_API_KEY or ANTHROPIC_API_KEY for commercial API instead.

set -euo pipefail
mkdir -p logs

echo "========================================================"
echo "  GuardLens Eval Stage 6: Precision + Transfer"
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
# 6a. Hard negative attribution precision + minimality curve
# -------------------------------------------------------
echo ""
echo "=== 6a. Attribution Precision + Minimality Curve ==="
echo "Start: $(date)"

python -m guardlens.evaluation.eval_attribution_precision \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/attribution_precision.json" \
    --device cuda \
    --batch-size 8 \
    --k-fracs 0.02 0.05 0.08 0.10 0.12 0.15 0.18 0.20 0.25 0.30 \
    --methods guardlens surface_risk integrated_gradients grad_x_input random

echo "6a complete: $(date)"

# -------------------------------------------------------
# 6b. Cross-model transfer (two-phase: GuardLens then LlamaGuard-3)
#
# Phase 1: GuardLens extracts attributions, builds conversation variants,
#          saves to disk, then is unloaded from GPU.
# Phase 2: LlamaGuard-3-8B loaded in 8-bit (~9GB), scores all variants.
#
# Memory budget on A100 40GB:
#   GuardLens (DeBERTa-large) in phase 1: ~3GB
#   LlamaGuard-3-8B 8-bit in phase 2:     ~9GB
#   Both never loaded simultaneously.     safe.
# -------------------------------------------------------
echo ""
echo "=== 6b. Cross-Model Transfer (LlamaGuard-3, two-phase) ==="
echo "Start: $(date)"

# Install bitsandbytes if not present (needed for 8-bit quantisation)
python3 -c "import bitsandbytes" 2>/dev/null || {
    echo "  Installing bitsandbytes..."
    pip install bitsandbytes --quiet --break-system-packages
}

LG_MODEL="meta-llama/Meta-Llama-Guard-3-8B"
QUANT="8bit"   # change to 4bit if A100 40GB is still tight after phase 1

echo "  LlamaGuard model: $LG_MODEL"
echo "  Quantisation:     $QUANT"

# Full test set (50 samples)
python -m guardlens.evaluation.eval_cross_model_transfer \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/cross_model_transfer_full.json" \
    --llamaguard-model "$LG_MODEL" \
    --quantization "$QUANT" \
    --n-samples 50 \
    --k-frac 0.15 \
    --variants-cache "$OUT_DIR/transfer_variants_full.jsonl" \
    --device cuda

echo "  Full test set done: $(date)"

# Free GPU before next run
python3 -c "import torch, gc; gc.collect(); torch.cuda.empty_cache()"

# Implicit subset (30 samples -- the key result for paper)
python -m guardlens.evaluation.eval_cross_model_transfer \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/cross_model_transfer_implicit.json" \
    --llamaguard-model "$LG_MODEL" \
    --quantization "$QUANT" \
    --subset implicit \
    --n-samples 30 \
    --k-frac 0.15 \
    --variants-cache "$OUT_DIR/transfer_variants_implicit.jsonl" \
    --device cuda

echo "6b complete: $(date)"

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "=== Stage 6 Summary ==="
python3 - <<'PYEOF'
import json, os

RES = os.path.expanduser("~/work/results/dataset_gen/results")

# Attribution precision
prec = None
try:
    with open(os.path.join(RES, "attribution_precision.json")) as f:
        prec = json.load(f)
except:
    pass

if prec:
    hn_results = prec.get("hard_negative_attribution_precision", {})
    print("\nHard Negative Attribution Precision:")
    print(f"  {'Subset':<22} {'FPAR':>8} {'MeanAttr':>10}")
    for name in ["adversarial", "hard_negative", "borderline", "genuine_benign"]:
        d = hn_results.get(name, {})
        if d:
            print(f"  {name:<22} {d.get('fpar', 0):>8.4f} {d.get('mean_attribution', 0):>10.4f}")

    min_curves = prec.get("minimality_curve", {})
    sharp = min_curves.get("sharpness", {})
    print("\nMinimality Sharpness:")
    for method, s in sharp.items():
        print(f"  {method:<25}: inflection_k={s.get('inflection_k',0):.3f}  "
              f"slope={s.get('slope_at_inflection',0):.3f}  "
              f"AUC={s.get('auc',0):.4f}")

# Cross-model transfer
for fname, label in [
    ("cross_model_transfer_full.json", "Full test"),
    ("cross_model_transfer_implicit.json", "Implicit"),
]:
    try:
        with open(os.path.join(RES, fname)) as f:
            xfer = json.load(f)
        tfr = xfer.get("transfer_flip_rates", {})
        print(f"\nCross-Model Transfer ({label}):")
        for method in ("guardlens", "surface_risk", "random"):
            d = tfr.get(method, {})
            print(f"  {method:<25}: TransferFlip={d.get('transfer_flip_rate',0):.3f}  "
                  f"({d.get('n_flipped',0)}/{d.get('n_tested',0)})")
    except:
        print(f"\n  {label}: results not found")
PYEOF

echo ""
echo "========================================================"
echo "  Stage 6 complete: $(date)"
echo "========================================================"
