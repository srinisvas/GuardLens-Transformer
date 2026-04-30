#!/bin/bash
#SBATCH --job-name=guardlens_cross_dataset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_cross_%j.out
#SBATCH --error=logs/eval_cross_%j.err

set -euo pipefail
mkdir -p logs

echo "========================================================"
echo "  GuardLens Eval Stage 3: Cross-Dataset Generalisation"
echo "  Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Start: $(date)"
echo "========================================================"

source ~/.bashrc
conda activate guardlens 2>/dev/null || source ~/work/venv/guardlens/bin/activate

RESULTS_DIR=~/work/results/dataset_gen
CKPT_DIR=$RESULTS_DIR/checkpoints
DATA=$RESULTS_DIR/semantic_multiturn_v10_augmented.jsonl
EVAL_DATA=$RESULTS_DIR/eval_data
OUT_DIR=$RESULTS_DIR/results
mkdir -p $OUT_DIR

SEEDS=$EVAL_DATA/cross_dataset_combined.jsonl

if [ ! -f "$SEEDS" ]; then
    echo "ERROR: Seed file not found: $SEEDS"
    echo "Run eval_stage1_setup.slurm first."
    exit 1
fi

cd ~/work/GuardLens-Transformer

# -------------------------------------------------------
# 3a. GuardLens (full model)
# -------------------------------------------------------
echo ""
echo "=== 3a. GuardLens cross-dataset eval ==="

python -m guardlens.evaluation.eval_cross_dataset \
    --seeds "$SEEDS" \
    --benign-pool "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/cross_dataset_guardlens.json" \
    --n-seeds 300 \
    --n-benign 300 \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda

echo "3a complete: $(date)"

# -------------------------------------------------------
# 3b. GuardLens-NoCF (ablation: no counterfactual training)
# -------------------------------------------------------
echo ""
echo "=== 3b. GuardLens-NoCF cross-dataset eval ==="

python -m guardlens.evaluation.eval_cross_dataset \
    --seeds "$SEEDS" \
    --benign-pool "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens_no_cf/best.pt" \
    --output "$OUT_DIR/cross_dataset_guardlens_no_cf.json" \
    --n-seeds 300 \
    --n-benign 300 \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda

echo "3b complete: $(date)"

# -------------------------------------------------------
# 3c. ConversationDeBERTa (baseline)
# -------------------------------------------------------
echo ""
echo "=== 3c. ConversationDeBERTa cross-dataset eval ==="

python -m guardlens.evaluation.eval_cross_dataset \
    --seeds "$SEEDS" \
    --benign-pool "$DATA" \
    --checkpoint "$CKPT_DIR/conversation_deberta/best.pt" \
    --output "$OUT_DIR/cross_dataset_conversation_deberta.json" \
    --n-seeds 300 \
    --n-benign 300 \
    --top-k 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda

echo "3c complete: $(date)"

# -------------------------------------------------------
# 3d. Per-source breakdown (all models combined into one table)
# -------------------------------------------------------
echo ""
echo "=== 3d. Collating cross-dataset results ==="

python3 - <<'PYEOF'
import json, os

OUT_DIR = os.path.expanduser("~/work/results/dataset_gen/results")
models = ["guardlens", "guardlens_no_cf", "conversation_deberta"]

print(f"\n{'='*70}")
print(f"  Cross-Dataset F1 Summary")
print(f"{'='*70}")
print(f"  {'Model':<30} {'F1':>8} {'Acc':>8} {'P':>8} {'R':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for model in models:
    path = os.path.join(OUT_DIR, f"cross_dataset_{model}.json")
    if not os.path.exists(path):
        print(f"  {model:<30} MISSING")
        continue
    with open(path) as f:
        data = json.load(f)
    cls = data["classification"]
    print(f"  {model:<30} {cls['f1']:>8.3f} {cls['accuracy']:>8.3f} "
          f"{cls['precision']:>8.3f} {cls['recall']:>8.3f}")

print(f"\n  Causal Metrics at 15%:")
print(f"  {'Model':<30} {'DevDrop':>10} {'Flip':>10} {'Suf':>10}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

for model in models:
    path = os.path.join(OUT_DIR, f"cross_dataset_{model}.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    causal = data.get("causal", {}).get("15%", {})
    dd = causal.get("deviation_drop", 0)
    flip = causal.get("flip_rate", 0)
    suf = causal.get("sufficiency", 0)
    print(f"  {model:<30} {dd:>10.3f} {flip:>10.3f} {suf:>10.3f}")
PYEOF

echo ""
echo "========================================================"
echo "  Stage 3 complete: $(date)"
echo "========================================================"
