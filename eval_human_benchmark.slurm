#!/bin/bash
#SBATCH --job-name=gl-eval-human
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_human_bench_%j.out
#SBATCH --error=logs/eval_human_bench_%j.err

# ============================================================
# eval_human_benchmark.slurm — Multi-turn causal hit evaluation
#
# Runs GuardLens on 100 human-annotated benchmark conversations
# and computes multi-turn causal hit metrics against human
# multi-pivot annotations.
#
# Uses: best_attribution.pt → token attribution → turn aggregation
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints"
FINAL_DATA="$HOME/staging/dataset_gen_output/final_dataset.jsonl"
ANNOTATIONS="$RESULTS_DIR/eval_data/human_benchmark_annotated.jsonl"
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
echo "  GuardLens Human Benchmark Evaluation"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Preflight
echo ""
echo "  Inputs:"
for f in "$FINAL_DATA" "$ANNOTATIONS"; do
    if [ ! -f "$f" ]; then
        echo "    MISSING: $f"; exit 1
    fi
    echo "    OK: $(basename $f) ($(wc -l < "$f") records)"
done

ATTR_CKPT="$CKPT_DIR/guardlens/best_attribution.pt"
[ ! -f "$ATTR_CKPT" ] && ATTR_CKPT="$CKPT_DIR/guardlens/best.pt"
if [ ! -f "$ATTR_CKPT" ]; then
    echo "    MISSING: checkpoint"; exit 1
fi
echo "    OK: $(basename $ATTR_CKPT)"

# ============================================================
# Run evaluation
# ============================================================

echo ""
python eval_human_benchmark.py \
    --checkpoint "$ATTR_CKPT" \
    --dataset "$FINAL_DATA" \
    --human-annotations "$ANNOTATIONS" \
    --output "$OUT_DIR/human_benchmark_eval.json" \
    --device cuda

EXIT_CODE=$?

# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================================"
echo "  Human Benchmark Evaluation Complete: $(date)"
echo "========================================================"

if [ $EXIT_CODE -eq 0 ] && [ -f "$OUT_DIR/human_benchmark_eval.json" ]; then
    echo "  OK: human_benchmark_eval.json"
    echo ""
    python3 -c "
import json
with open('$OUT_DIR/human_benchmark_eval.json') as f:
    r = json.load(f)
gl = r.get('guardlens_multiturn', {})
sr = r.get('surface_risk_multiturn', {})
cls = r.get('classification', {})
hs = r.get('human_annotation_stats', {})

print(f'  Annotations: {hs.get(\"n_total\",0)} ({hs.get(\"n_adversarial\",0)} adv, {hs.get(\"n_benign\",0)} ben)')
print(f'  Multi-pivot: {hs.get(\"multi_pivot_rate\",0):.1%}, mean causal turns: {hs.get(\"mean_causal_turns\",0):.1f}')
print(f'  Classification: F1={cls.get(\"f1\",0):.3f} Acc={cls.get(\"accuracy\",0):.3f} Kappa={cls.get(\"kappa\",0):.3f}')
print()
print(f'  {\"Metric\":<28} {\"GuardLens\":>10} {\"SurfRisk\":>10} {\"Delta\":>10}')
print(f'  {\"-\"*28} {\"-\"*10} {\"-\"*10} {\"-\"*10}')
for k in ['top1_hit','top3_hit','top5_hit','top3_coverage','top5_coverage','causal_mass_ratio','human_coverage']:
    g = gl.get(k,{}).get('mean',0)
    s = sr.get(k,{}).get('mean',0)
    d = g - s
    print(f'  {k:<28} {g:>10.3f} {s:>10.3f} {d:>+10.3f}')
dg = gl.get('first_causal_distance',{}).get('mean',0)
ds = sr.get('first_causal_distance',{}).get('mean',0)
dd = dg - ds
print(f'  {\"first_causal_distance\":<28} {dg:>10.1f} {ds:>10.1f} {dd:>+10.1f}')
" 2>/dev/null
else
    echo "  FAILED (exit code: ${EXIT_CODE})"
fi

echo ""
echo "========================================================"
