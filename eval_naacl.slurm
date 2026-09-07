#!/bin/bash
#SBATCH --job-name=guardlens-naacl-eval
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_naacl_%j.out
#SBATCH --error=logs/eval_naacl_%j.err

# Targeted NAACL evaluation suite. Reuses the existing reviewer baselines but
# points them only at repaired splits/checkpoints.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
SPLIT_DIR="${SPLIT_DIR:-$HOME/staging/dataset_gen_output/naacl_splits}"
CKPT_DIR="${CKPT_DIR:-$HOME/work/results/guardlens_naacl/checkpoints}"
OUT_DIR="${OUT_DIR:-$HOME/work/results/guardlens_naacl/results}"

TRAIN_PATH="$SPLIT_DIR/train.jsonl"
DEV_PATH="$SPLIT_DIR/dev.jsonl"
TEST_PATH="$SPLIT_DIR/test.jsonl"
GL_ATTR="$CKPT_DIR/guardlens/best_attribution.pt"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export TMPDIR="${TMPDIR:-$HOME/work/.tmp}"
export HF_HOME="${HF_HOME:-$HOME/work/hf_models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$TMPDIR" "$OUT_DIR" logs
cd "$PROJECT_DIR"

for path in "$TRAIN_PATH" "$DEV_PATH" "$TEST_PATH" "$GL_ATTR"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required input missing: $path"
        exit 2
    fi
done

# 1. Shortcut check. This is CPU-light and should be interpreted first.
python -m guardlens.evaluation.eval_length_probe \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --test "$TEST_PATH" \
    --output "$OUT_DIR/length_probe.json"

# 2. Top-k evidence-bearing turn localization against repaired evidence targets.
python -m guardlens.evaluation.eval_topk_turn_hitrate \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/topk_turn_hitrate.json" \
    --methods guardlens surface_risk attention random \
    --k-values 1 2 3 5 \
    --batch-size 4 \
    --device cuda

# 3. Reviewer-requested leave-one-turn-out baseline using the same classifier.
python -m guardlens.evaluation.eval_loto_baseline \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/loto_baseline.json" \
    --k-values 1 2 3 5 \
    --top-k-fracs 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda

# 4. Attribution intervention metrics and tier-stratified analysis.
python -m guardlens.evaluation.eval_causal \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/evidence_eval_with_tiers.json" \
    --methods guardlens surface_risk attention random \
    --top-k 0.05 0.10 0.15 0.20 \
    --tier-eval \
    --batch-size 4 \
    --device cuda

# 5. Utility tradeoff without reusing the old boundary pool. If a repaired
# boundary set is later supplied, eval_utility_grid can be rerun separately.
python -m guardlens.evaluation.eval_utility_grid \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/utility_grid.json" \
    --methods guardlens surface_risk \
    --k-values 0.05 0.10 0.15 0.20 \
    --lambda-values 0.0 0.5 1.0 2.0 5.0 \
    --batch-size 4 \
    --device cuda

echo "========================================================"
echo "NAACL evaluation complete"
echo "Results: $OUT_DIR"
echo "========================================================"

python3 - <<PY
import json, os
out = "$OUT_DIR"
for name in ["length_probe", "topk_turn_hitrate", "loto_baseline", "evidence_eval_with_tiers", "utility_grid"]:
    path = os.path.join(out, name + ".json")
    print(f"{name}: {'OK' if os.path.exists(path) else 'MISSING'}")
if os.path.exists(os.path.join(out, "length_probe.json")):
    r = json.load(open(os.path.join(out, "length_probe.json")))
    t = r["test"]
    print(f"Length probe test AUC={t['roc_auc']:.3f}, balanced_acc={t['balanced_accuracy']:.3f}")
PY
