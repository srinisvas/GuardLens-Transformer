#!/bin/bash
#SBATCH --job-name=gl-p3eval
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_phase3_%j.out
#SBATCH --error=logs/eval_phase3_%j.err

# ============================================================
# Phase 3 checkpoint evaluation
#
# Compares phase 3 (CF-trained) vs phase 2 (no CF) to validate
# that CF fine-tuning improves attribution.
#
# Runs:
#   1. Classification eval (best_phase3.pt)
#   2. Core causal eval with tier/transfer breakdowns
#   3. Boundary stress test
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints/guardlens"
TEST_DATA="$HOME/staging/dataset_gen_output/splits/test.jsonl"
BOUNDARY_DATA="$HOME/staging/dataset_gen_output/benign_boundary.jsonl"
OUT_DIR="$RESULTS_DIR/results"

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
echo "  Phase 3 Checkpoint Evaluation"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

P3_CKPT="$CKPT_DIR/best_phase3.pt"
if [ ! -f "$P3_CKPT" ]; then
    echo "ERROR: best_phase3.pt not found. Run train_phase3.slurm first."
    exit 1
fi

echo "Checkpoint: $P3_CKPT"
echo "Test data: $(wc -l < "$TEST_DATA") records"

# ---- 1. Classification ----
echo ""
echo "=== 1. Classification ==="
python -m guardlens.evaluate \
    --test-path "$TEST_DATA" \
    --checkpoint "$P3_CKPT" \
    --output "$OUT_DIR/classification_guardlens_phase3.json" \
    --batch-size 4 \
    --device cuda \
2>"$OUT_DIR/classification_guardlens_phase3.stderr" \
&& echo "1: SUCCESS" || echo "1: FAILED"

# ---- 2. Core causal eval ----
echo ""
echo "=== 2. Causal Attribution ==="
python -m guardlens.evaluation.eval_causal \
    --test-path "$TEST_DATA" \
    --checkpoint "$P3_CKPT" \
    --output "$OUT_DIR/causal_eval_phase3.json" \
    --methods guardlens surface_risk integrated_gradients random \
    --top-k 0.05 0.10 0.15 0.20 \
    --tier-eval \
    --transfer-eval \
    --batch-size 4 \
    --device cuda \
&& echo "2: SUCCESS" || echo "2: FAILED"

# ---- 3. Boundary stress ----
echo ""
echo "=== 3. Boundary Stress ==="
if [ -f "$BOUNDARY_DATA" ]; then
    python -m guardlens.evaluation.eval_boundary_stress \
        --boundary-files "$BOUNDARY_DATA" \
        --checkpoint "$P3_CKPT" \
        --output "$OUT_DIR/boundary_stress_phase3.json" \
        --device cuda \
    && echo "3: SUCCESS" || echo "3: FAILED"
else
    echo "3: SKIPPED"
fi

# ---- Summary: Phase 2 vs Phase 3 comparison ----
echo ""
echo "========================================================"
echo "  Phase 2 vs Phase 3 Comparison"
echo "========================================================"

python3 -c "
import json, os

OUT = '$OUT_DIR'

def load(f):
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        return json.load(open(p))
    return None

# Classification comparison
print('  Classification:')
print(f'  {\"Checkpoint\":<20} {\"F1\":>8} {\"Acc\":>8} {\"AttrF1\":>8} {\"PivMal\":>8} {\"Thr\":>6}')
print(f'  {\"-\"*20} {\"-\"*8} {\"-\"*8} {\"-\"*8} {\"-\"*8} {\"-\"*6}')

for label, fname in [('Phase 2', 'classification_guardlens.json'),
                     ('Phase 3', 'classification_guardlens_phase3.json')]:
    r = load(fname)
    if r:
        print(f'  {label:<20} {r.get(\"f1\",0):>8.4f} {r.get(\"accuracy\",0):>8.4f} '
              f'{r.get(\"attr_f1\",0):>8.4f} {r.get(\"pivot_accuracy_malicious\",0):>8.4f} '
              f'{r.get(\"threshold\",0.5):>6.2f}')
    else:
        print(f'  {label:<20} no results')

# Causal attribution comparison
print()
print('  Causal Attribution (GuardLens):')
print(f'  {\"Checkpoint\":<20} {\"DD@15%\":>8} {\"Flip@15%\":>10} {\"TokF1\":>8}')
print(f'  {\"-\"*20} {\"-\"*8} {\"-\"*10} {\"-\"*8}')

for label, fname in [('Phase 2', 'causal_eval_results.json'),
                     ('Phase 3', 'causal_eval_phase3.json')]:
    r = load(fname)
    if r:
        gl = r.get('full_test', {}).get('guardlens', {})
        dd = gl.get('deviation_drops', {}).get('15%', 0)
        flip = gl.get('flip_rates', {}).get('flip@15%', 0)
        tf1 = gl.get('token_f1', 0)
        print(f'  {label:<20} {dd:>8.3f} {flip:>10.3f} {tf1:>8.3f}')
    else:
        print(f'  {label:<20} no results')

# Tier comparison
print()
print('  Per-tier DD@15% (GuardLens):')
print(f'  {\"Tier\":<20} {\"Phase 2\":>10} {\"Phase 3\":>10} {\"Delta\":>10}')
print(f'  {\"-\"*20} {\"-\"*10} {\"-\"*10} {\"-\"*10}')

p2 = load('causal_eval_results.json')
p3 = load('causal_eval_phase3.json')
if p2 and p3:
    p2_tiers = p2.get('tier_eval', {})
    p3_tiers = p3.get('tier_eval', {})
    for tier in ['cf_weak', 'cf_strong', 'llm_confirmed', 'construction']:
        dd2 = p2_tiers.get(tier, {}).get('guardlens', {}).get('deviation_drops', {}).get('15%', None)
        dd3 = p3_tiers.get(tier, {}).get('guardlens', {}).get('deviation_drops', {}).get('15%', None)
        if dd2 is not None and dd3 is not None:
            delta = dd3 - dd2
            print(f'  {tier:<20} {dd2:>10.3f} {dd3:>10.3f} {delta:>+10.3f}')
        elif dd2 is not None:
            print(f'  {tier:<20} {dd2:>10.3f} {\"—\":>10}')

# Boundary comparison
print()
print('  Boundary Stress:')
for label, fname in [('Phase 2', 'boundary_stress.json'),
                     ('Phase 3', 'boundary_stress_phase3.json')]:
    r = load(fname)
    if r:
        print(f'  {label:<20} FPR={r.get(\"false_positive_rate\",0):.4f} '
              f'MeanP={r.get(\"mean_p_adv\",0):.4f} '
              f'FP={r.get(\"false_positives\",0)}/{r.get(\"total\",0)}')
    else:
        print(f'  {label:<20} no results')
" 2>/dev/null

echo ""
echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"
