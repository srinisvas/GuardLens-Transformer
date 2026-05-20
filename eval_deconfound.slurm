#!/bin/bash
#SBATCH --job-name=gl-deconfound
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_deconfound_%j.out
#SBATCH --error=logs/eval_deconfound_%j.err

# ============================================================
# Deconfounded evaluation + attribution utility metrics
#
# Runs:
#   1. Attribution utility (DD - λ*FPR tradeoff)
#   2. Causal turn mass + pivot window accuracy
#   3. Deconfounded test variants:
#      A. SR-neutralized adversarial
#      B. Noise-equalized adversarial
#      C. SR-injected benign
#      D. Combined (A+B)
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT="$RESULTS_DIR/checkpoints/guardlens/best_attribution.pt"
TEST_DATA="$HOME/staging/dataset_gen_output/splits/test.jsonl"
OUT_DIR="$RESULTS_DIR/results"

CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT_DIR/deconfounded"

echo "========================================================"
echo "  Deconfounded Evaluation + Attribution Utility"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Missing checkpoint: $CKPT"
    exit 1
fi

# Fix #10: Check prerequisite result files
for f in "$OUT_DIR/causal_eval_results.json" "$OUT_DIR/boundary_stress.json" "$OUT_DIR/surface_risk_fpr.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing prerequisite: $f"
        echo "  Run eval_full.slurm first"
        exit 1
    fi
done
echo "  Prerequisites OK"

# ---- 1. Attribution Utility + Causal Turn Mass + Pivot Window ----
echo ""
echo "========================================================"
echo "  1. Attribution Utility + Turn Mass + Pivot Window"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_attribution_utility \
    --test-path "$TEST_DATA" \
    --causal-results "$OUT_DIR/causal_eval_results.json" \
    --boundary-results "$OUT_DIR/boundary_stress.json" \
    --surface-fpr-results "$OUT_DIR/surface_risk_fpr.json" \
    --checkpoint "$CKPT" \
    --output "$OUT_DIR/attribution_utility.json" \
    --device cuda \
&& echo "1: SUCCESS" || echo "1: FAILED"

echo "1 end: $(date +%H:%M:%S)"

# ---- 2. Deconfounded Variants ----
echo ""
echo "========================================================"
echo "  2. Deconfounded Test Variants"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_deconfounded \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT" \
    --output-dir "$OUT_DIR/deconfounded" \
    --methods guardlens surface_risk random \
    --top-k 0.05 0.10 0.15 0.20 \
    --device cuda \
&& echo "2: SUCCESS" || echo "2: FAILED"

echo "2 end: $(date +%H:%M:%S)"

# ---- Summary ----
echo ""
echo "========================================================"
echo "  Summary"
echo "========================================================"

# Print utility table
if [ -f "$OUT_DIR/attribution_utility.json" ]; then
    echo ""
    echo "  Attribution Utility (λ=1.0):"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/attribution_utility.json'))
ut = r.get('utility_all_benign', {}).get('lambda_1.0', {})
print(f'  {\"Method\":<25} {\"DD@15%\":>8} {\"FPR\":>8} {\"Utility\":>10}')
print(f'  {\"-\"*25} {\"-\"*8} {\"-\"*8} {\"-\"*10}')
for m, v in sorted(ut.items(), key=lambda x: -x[1]['utility']):
    print(f'  {m:<25} {v[\"dd\"]:>8.3f} {v[\"fpr\"]:>8.3f} {v[\"utility\"]:>10.3f}')
" 2>/dev/null

    echo ""
    echo "  Causal Turn Mass:"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/attribution_utility.json'))
ctm = r.get('causal_turn_mass', {})
cm = ctm.get('causal_turn_mass', {})
print(f'  Mean: {cm.get(\"mean\", 0):.3f}, Median: {cm.get(\"median\", 0):.3f}')
prm = ctm.get('per_role_mass', {})
for role, stats in sorted(prm.items(), key=lambda x: -x[1]['mean']):
    print(f'    {role:<20} mean={stats[\"mean\"]:.4f} n={stats[\"n_turns\"]}')
" 2>/dev/null

    echo ""
    echo "  Pivot Window Accuracy:"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/attribution_utility.json'))
pw = r.get('pivot_window', {})
for w, d in pw.get('window_accuracy', {}).items():
    print(f'    {w}: {d[\"accuracy\"]:.3f} ({d[\"correct\"]}/{pw[\"total\"]})')
ds = pw.get('distance_stats', {})
if ds:
    print(f'  Distance: mean={ds[\"mean\"]:.1f} median={ds[\"median\"]:.1f} p90={ds[\"p90\"]:.1f}')
" 2>/dev/null
fi

# Print deconfounded comparison
if [ -f "$OUT_DIR/deconfounded/deconfounded_results.json" ]; then
    echo ""
    echo "  Deconfounded Comparison (DD@15%):"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/deconfounded/deconfounded_results.json'))
print(f'  {\"Variant\":<25} {\"GuardLens\":>12} {\"SurfRisk\":>12} {\"Random\":>12}')
print(f'  {\"-\"*25} {\"-\"*12} {\"-\"*12} {\"-\"*12}')
for variant in ['original', 'sr_neutralized', 'sr_neutralized_changed', 'noise_equalized', 'combined']:
    vr = r.get(variant, {})
    gl = vr.get('guardlens', {}).get('deviation_drops', {}).get('15%', 0)
    sr = vr.get('surface_risk', {}).get('deviation_drops', {}).get('15%', 0)
    rn = vr.get('random', {}).get('deviation_drops', {}).get('15%', 0)
    print(f'  {variant:<25} {gl:>12.3f} {sr:>12.3f} {rn:>12.3f}')

inj = r.get('sr_injected_benign', {})
if inj:
    print(f'')
    print(f'  SR-Injected Benign FPR:')
    print(f'    GuardLens:    {inj.get(\"guardlens_fpr\", 0):.3f} ({inj.get(\"guardlens_fp\", 0)}/{inj.get(\"total\", 0)})')
    print(f'    Surface Risk: {inj.get(\"surface_risk_fpr\", 0):.3f} ({inj.get(\"surface_risk_fp\", 0)}/{inj.get(\"total\", 0)})')
" 2>/dev/null
fi

echo ""
echo "========================================================"
echo "  Done: $(date +%H:%M:%S)"
echo "========================================================"
