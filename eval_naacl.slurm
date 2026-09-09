#!/bin/bash
#SBATCH --job-name=guardlens-naacl-eval
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_naacl_%j.out
#SBATCH --error=logs/eval_naacl_%j.err

# Targeted NAACL evaluation suite on repaired splits/checkpoints only.
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
SPLIT_DIR="${SPLIT_DIR:-$HOME/staging/dataset_gen_output/naacl_splits}"
CKPT_DIR="${CKPT_DIR:-$HOME/work/results/guardlens_naacl/checkpoints}"
OUT_DIR="${OUT_DIR:-$HOME/work/results/guardlens_naacl/results}"
LEGACY_BENIGN_STRESS="${LEGACY_BENIGN_STRESS:-$HOME/staging/dataset_gen_output/naacl_legacy_benign_stress.jsonl}"
FRONTIER_BENIGN_STRESS="${FRONTIER_BENIGN_STRESS:-$HOME/staging/dataset_gen_output/naacl_frontier_benign_stress.jsonl}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

TRAIN_PATH="$SPLIT_DIR/train.jsonl"
DEV_PATH="$SPLIT_DIR/dev.jsonl"
TEST_PATH="$SPLIT_DIR/test.jsonl"

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

if [[ "$EVAL_BATCH_SIZE" -le 0 ]]; then
    echo "ERROR: EVAL_BATCH_SIZE must be positive"
    exit 2
fi

pick_ckpt() {
    local dir="$1"
    for name in best_attribution.pt best.pt best_detection.pt; do
        if [[ -f "$dir/$name" ]]; then
            echo "$dir/$name"
            return 0
        fi
    done
    return 1
}

for path in "$TRAIN_PATH" "$DEV_PATH" "$TEST_PATH"; do
    [[ -f "$path" ]] || { echo "ERROR: required split missing: $path"; exit 2; }
done

GL_ATTR=$(pick_ckpt "$CKPT_DIR/guardlens") || {
    echo "ERROR: no GuardLens checkpoint found"; exit 2;
}
NOCF_ATTR=$(pick_ckpt "$CKPT_DIR/guardlens_no_cf" || true)

echo "Primary checkpoint: $GL_ATTR"
[[ -n "$NOCF_ATTR" ]] && echo "NoCF checkpoint: $NOCF_ATTR"
echo "Evaluation batch size: $EVAL_BATCH_SIZE"

# Final held-out shortcut report. Unlike train_naacl preflight, this accesses test
# only after preprocessing/training choices are frozen.
python -m guardlens.evaluation.eval_length_probe \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --test "$TEST_PATH" \
    --output "$OUT_DIR/length_probe.json"

python -m guardlens.evaluation.eval_topk_turn_hitrate \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/topk_turn_hitrate.json" \
    --methods guardlens surface_risk attention random \
    --k-values 1 2 3 5 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device cuda

python -m guardlens.evaluation.eval_loto_baseline \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/loto_baseline.json" \
    --k-values 1 2 3 5 \
    --top-k-fracs 0.05 0.10 0.15 0.20 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device cuda

python -m guardlens.evaluation.eval_causal \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/evidence_eval_with_tiers.json" \
    --methods guardlens surface_risk attention random \
    --top-k 0.05 0.10 0.15 0.20 \
    --tier-eval \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device cuda

python -m guardlens.evaluation.eval_utility_grid \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/utility_grid.json" \
    --methods guardlens surface_risk \
    --k-values 0.05 0.10 0.15 0.20 \
    --lambda-values 0.0 0.5 1.0 2.0 5.0 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device cuda

# Existing NoCF ablation, rerun on exactly the repaired test distribution.
if [[ -n "$NOCF_ATTR" ]]; then
    python -m guardlens.evaluation.eval_causal \
        --test-path "$TEST_PATH" \
        --checkpoint "$NOCF_ATTR" \
        --output "$OUT_DIR/evidence_eval_nocf.json" \
        --methods guardlens \
        --top-k 0.05 0.10 0.15 0.20 \
        --batch-size "$EVAL_BATCH_SIZE" \
        --device cuda

    python -m guardlens.evaluation.eval_utility_grid \
        --test-path "$TEST_PATH" \
        --checkpoint "$NOCF_ATTR" \
        --output "$OUT_DIR/utility_grid_nocf.json" \
        --methods guardlens \
        --k-values 0.05 0.10 0.15 0.20 \
        --lambda-values 0.0 0.5 1.0 2.0 5.0 \
        --batch-size "$EVAL_BATCH_SIZE" \
        --device cuda
else
    echo "WARN: NoCF checkpoint unavailable; NoCF evaluation skipped"
fi

# Stress sets are never training or threshold-tuning sources. Evaluate them
# separately so legacy untrimmed benign behavior and the new frontier-authored
# hard-benign behavior remain distinguishable in the paper/results.
if [[ ! -f "$LEGACY_BENIGN_STRESS" ]]; then
    OLD_STRESS="$HOME/staging/dataset_gen_output/naacl_benign_untrimmed_stress.jsonl"
    if [[ -f "$OLD_STRESS" ]]; then
        echo "WARN: canonical legacy stress path missing; using legacy filename: $OLD_STRESS"
        LEGACY_BENIGN_STRESS="$OLD_STRESS"
    fi
fi

if [[ -f "$LEGACY_BENIGN_STRESS" ]]; then
    python -m guardlens.evaluation.eval_boundary_stress \
        --boundary-files "$LEGACY_BENIGN_STRESS" \
        --checkpoint "$GL_ATTR" \
        --output "$OUT_DIR/legacy_benign_stress.json" \
        --batch-size "$EVAL_BATCH_SIZE" \
        --device cuda
else
    echo "WARN: legacy benign stress file missing: $LEGACY_BENIGN_STRESS"
fi

if [[ -f "$FRONTIER_BENIGN_STRESS" ]]; then
    python -m guardlens.evaluation.eval_boundary_stress \
        --boundary-files "$FRONTIER_BENIGN_STRESS" \
        --checkpoint "$GL_ATTR" \
        --output "$OUT_DIR/frontier_benign_stress.json" \
        --batch-size "$EVAL_BATCH_SIZE" \
        --device cuda
else
    echo "WARN: frontier benign stress file missing: $FRONTIER_BENIGN_STRESS"
fi

echo "========================================================"
echo "NAACL evaluation complete"
echo "Results: $OUT_DIR"
echo "========================================================"

python3 - <<PY
import json, os
out = "$OUT_DIR"
names = [
    "length_probe", "topk_turn_hitrate", "loto_baseline",
    "evidence_eval_with_tiers", "utility_grid", "evidence_eval_nocf",
    "utility_grid_nocf", "legacy_benign_stress", "frontier_benign_stress",
]
for name in names:
    path = os.path.join(out, name + ".json")
    print(f"{name}: {'OK' if os.path.exists(path) else 'MISSING/OPTIONAL'}")
path = os.path.join(out, "length_probe.json")
if os.path.exists(path):
    r = json.load(open(path, encoding="utf-8"))
    t = r["test"]
    print(f"Length probe test AUC={t['roc_auc']:.3f}, balanced_acc={t['balanced_accuracy']:.3f}")
for name in ("legacy_benign_stress", "frontier_benign_stress"):
    path = os.path.join(out, name + ".json")
    if os.path.exists(path):
        r = json.load(open(path, encoding="utf-8"))
        print(f"{name} FPR={r['false_positive_rate']:.3f} n={r['total']}")
PY
