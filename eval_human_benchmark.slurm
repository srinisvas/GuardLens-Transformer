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

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints"
FINAL_DATA="$HOME/staging/dataset_gen_output/final_dataset.jsonl"
EVAL_DATA="$RESULTS_DIR/eval_data"
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
echo "  GuardLens Human Benchmark (Multi-Annotator)"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ATTR_CKPT="$CKPT_DIR/guardlens/best_attribution.pt"
[ ! -f "$ATTR_CKPT" ] && ATTR_CKPT="$CKPT_DIR/guardlens/best.pt"

ANN_A="$EVAL_DATA/human_benchmark_annotated.jsonl"
ANN_B="$EVAL_DATA/adversarial_annotation_completed-user1.jsonl"

for f in "$FINAL_DATA" "$ATTR_CKPT" "$ANN_A" "$ANN_B"; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"; exit 1
    fi
    echo "  OK: $(basename $f)"
done

echo ""
python eval_human_benchmark.py \
    --checkpoint "$ATTR_CKPT" \
    --dataset "$FINAL_DATA" \
    --human-annotations "$ANN_A" "$ANN_B" \
    --annotator-names "Ann_A" "Ann_B" \
    --output "$OUT_DIR/human_benchmark_eval.json" \
    --device cuda

EXIT_CODE=$?

echo ""
echo "========================================================"
if [ $EXIT_CODE -eq 0 ] && [ -f "$OUT_DIR/human_benchmark_eval.json" ]; then
    echo "  SUCCESS"
    echo ""
    python3 -c "
import json
with open('$OUT_DIR/human_benchmark_eval.json') as f:
    r = json.load(f)

for ann in ['Ann_A', 'Ann_B']:
    if ann not in r: continue
    gl = r[ann].get('guardlens_multiturn', {})
    sr = r[ann].get('surface_risk_multiturn', {})
    cls = r[ann].get('classification', {})
    hs = r[ann].get('human_stats', {})
    print(f'  === {ann}: {hs.get(\"n_adversarial\",0)} adv, multi-pivot={hs.get(\"multi_pivot_rate\",0):.0%}, mean_turns={hs.get(\"mean_causal_turns\",0):.1f} ===')
    print(f'  Classification: F1={cls.get(\"f1\",0):.3f} Kappa={cls.get(\"kappa\",0):.3f}')
    print(f'  {\"Metric\":<28} {\"GuardLens\":>10} {\"SurfRisk\":>10} {\"Delta\":>10}')
    print(f'  {\"-\"*28} {\"-\"*10} {\"-\"*10} {\"-\"*10}')
    for k in ['top1_hit','top3_hit','top5_hit','top3_coverage','top5_coverage','causal_mass_ratio','human_coverage']:
        g = gl.get(k,{}).get('mean',0); s = sr.get(k,{}).get('mean',0)
        print(f'  {k:<28} {g:>10.3f} {s:>10.3f} {g-s:>+10.3f}')
    print()

ia = r.get('inter_annotator', {})
for k, v in ia.items():
    print(f'  Inter-annotator ({k}): kappa={v.get(\"kappa\",0):.3f} pivot_jaccard={v.get(\"pivot_jaccard_mean\",0):.3f}')
" 2>/dev/null
else
    echo "  FAILED (exit ${EXIT_CODE})"
fi
echo "  Done: $(date)"
echo "========================================================"
