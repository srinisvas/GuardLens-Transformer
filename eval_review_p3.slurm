#!/bin/bash
#SBATCH --job-name=gl-review-p3
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=logs/review_p3_%j.out
#SBATCH --error=logs/review_p3_%j.err

# ============================================================
# Phase 3: Reviewer i7qz + MyZK response experiments
#
#   1. LOTO occlusion baseline (~1.5h)
#   2. Utility grid over k and lambda (~30min)
#   3. Per-supervision-tier causal eval (~30min)
# ============================================================

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"

TEST_PATH="$HOME/staging/dataset_gen_output/splits/test.jsonl"
CKPT_DIR="$HOME/work/results/guardlens_v11/checkpoints"
OUT_DIR="$HOME/work/results/guardlens_v11/results/review_response"
BOUNDARY_DIR="$HOME/staging/dataset_gen_output"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" "$OUT_DIR" logs

cd "$PROJECT_DIR"

echo "========================================================"
echo "  Reviewer Response Phase 3 (i7qz + MyZK)"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  Python: $(which python) ($(python --version 2>&1))"

GL_ATTR=$(ls "$CKPT_DIR/guardlens/best_attribution.pt" 2>/dev/null || echo "")
if [ -z "$GL_ATTR" ]; then
    echo "FATAL: No GuardLens checkpoint."; exit 1
fi

# Find boundary files (use array to handle empty case safely)
BOUNDARY_ARGS=()
for f in "$BOUNDARY_DIR"/benign_boundary.jsonl \
         "$BOUNDARY_DIR"/old_benign_boundary_or_unused.jsonl; do
    [ -f "$f" ] && BOUNDARY_ARGS+=("$f")
done
echo "  Boundary files: ${#BOUNDARY_ARGS[@]} found"


# ---- 1. LOTO Occlusion Baseline ----

echo ""
echo "========================================================"
echo "  1. LOTO Occlusion Baseline"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_loto_baseline \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/loto_baseline.json" \
    --k-values 1 2 3 5 \
    --top-k-fracs 0.05 0.10 0.15 0.20 \
    --batch-size 4 \
    --device cuda \
&& echo "  1: SUCCESS ($(date +%H:%M:%S))" \
|| echo "  1: FAILED ($(date +%H:%M:%S))"


# ---- 2. Utility Grid ----

echo ""
echo "========================================================"
echo "  2. Attribution Utility Grid (k x lambda)"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

if [ "${#BOUNDARY_ARGS[@]}" -gt 0 ]; then
    python -m guardlens.evaluation.eval_utility_grid \
        --test-path "$TEST_PATH" \
        --checkpoint "$GL_ATTR" \
        --boundary-files "${BOUNDARY_ARGS[@]}" \
        --output "$OUT_DIR/utility_grid.json" \
        --methods guardlens surface_risk \
        --k-values 0.05 0.10 0.15 0.20 \
        --lambda-values 0.0 0.5 1.0 2.0 5.0 \
        --batch-size 4 \
        --device cuda
else
    python -m guardlens.evaluation.eval_utility_grid \
        --test-path "$TEST_PATH" \
        --checkpoint "$GL_ATTR" \
        --output "$OUT_DIR/utility_grid.json" \
        --methods guardlens surface_risk \
        --k-values 0.05 0.10 0.15 0.20 \
        --lambda-values 0.0 0.5 1.0 2.0 5.0 \
        --batch-size 4 \
        --device cuda
fi
echo "  2a end: $(date +%H:%M:%S)"

# NoCF utility (separate checkpoint, guardlens attribution only)
NOCF_ATTR=$(ls "$CKPT_DIR/guardlens_no_cf/best_attribution.pt" 2>/dev/null || echo "")
if [ -n "$NOCF_ATTR" ]; then
    echo ""
    echo "  2b. NoCF Utility..."
    if [ "${#BOUNDARY_ARGS[@]}" -gt 0 ]; then
        python -m guardlens.evaluation.eval_utility_grid \
            --test-path "$TEST_PATH" \
            --checkpoint "$NOCF_ATTR" \
            --boundary-files "${BOUNDARY_ARGS[@]}" \
            --output "$OUT_DIR/utility_grid_nocf.json" \
            --methods guardlens \
            --k-values 0.05 0.10 0.15 0.20 \
            --lambda-values 0.0 0.5 1.0 2.0 5.0 \
            --batch-size 4 \
            --device cuda
    else
        python -m guardlens.evaluation.eval_utility_grid \
            --test-path "$TEST_PATH" \
            --checkpoint "$NOCF_ATTR" \
            --output "$OUT_DIR/utility_grid_nocf.json" \
            --methods guardlens \
            --k-values 0.05 0.10 0.15 0.20 \
            --lambda-values 0.0 0.5 1.0 2.0 5.0 \
            --batch-size 4 \
            --device cuda
    fi
    echo "  2b end: $(date +%H:%M:%S)"
fi


# ---- 3. Per-Tier Causal Eval ----

echo ""
echo "========================================================"
echo "  3. Per-Supervision-Tier Causal Eval"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_causal \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --output "$OUT_DIR/causal_eval_with_tiers.json" \
    --methods guardlens surface_risk attention random \
    --top-k 0.05 0.10 0.15 0.20 \
    --tier-eval \
    --batch-size 4 \
    --device cuda \
&& echo "  3: SUCCESS ($(date +%H:%M:%S))" \
|| echo "  3: FAILED ($(date +%H:%M:%S))"


# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================================"
echo "  Phase 3 Summary — $(date)"
echo "========================================================"

# LOTO vs GuardLens comparison
if [ -f "$OUT_DIR/loto_baseline.json" ] && [ -f "$OUT_DIR/topk_turn_hitrate.json" ]; then
    echo ""
    echo "  LOTO vs GuardLens Turn Hit Rate:"
    python3 -c "
import json
loto = json.load(open('$OUT_DIR/loto_baseline.json'))
gl = json.load(open('$OUT_DIR/topk_turn_hitrate.json'))
gl_r = gl.get('results', {}).get('guardlens', {})
loto_r = loto.get('turn_hit_rates', {})
print(f'  {\"k\":<6} {\"LOTO\":>10} {\"GuardLens\":>10} {\"Random\":>10} {\"GL-LOTO\":>10}')
print(f'  {\"-\"*6} {\"-\"*10} {\"-\"*10} {\"-\"*10} {\"-\"*10}')
for k in [1, 2, 3, 5]:
    l = loto_r.get(f'top_{k}', {}).get('hit_rate', 0)
    g = gl_r.get(f'top_{k}', {}).get('hit_rate', 0)
    r = loto_r.get(f'top_{k}', {}).get('random_floor', 0)
    print(f'  {k:<6} {l:>10.3f} {g:>10.3f} {r:>10.3f} {g-l:>+10.3f}')
" 2>/dev/null || echo "    parse error"
fi

# LOTO DD
if [ -f "$OUT_DIR/loto_baseline.json" ]; then
    echo ""
    echo "  LOTO Deviation Drop:"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/loto_baseline.json'))
dd = r.get('deviation_drop', {})
for k in ['5%', '10%', '15%', '20%']:
    d = dd.get(k, {})
    print(f'    DD@{k}: {d.get(\"dd\", 0):.3f} (n={d.get(\"n\", 0)})')
" 2>/dev/null || echo "    parse error"
fi

# Utility grid winner
if [ -f "$OUT_DIR/utility_grid.json" ]; then
    echo ""
    echo "  Utility Grid Winners (GL vs SR):"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/utility_grid.json'))
grid = r.get('grid', {})
for k_str in ['5%', '10%', '15%', '20%']:
    wins = []
    for lam_str, methods in grid.get(k_str, {}).items():
        gl = methods.get('guardlens', {}).get('utility', 0)
        sr = methods.get('surface_risk', {}).get('utility', 0)
        winner = 'GL' if gl > sr else 'SR'
        wins.append(f'{lam_str}:{winner}')
    print(f'    k={k_str}: {\"  \".join(wins)}')
" 2>/dev/null || echo "    parse error"
fi

# Per-tier
if [ -f "$OUT_DIR/causal_eval_with_tiers.json" ]; then
    echo ""
    echo "  Per-Tier DD@15% (guardlens method):"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/causal_eval_with_tiers.json'))
tiers = r.get('tier_eval', {})
for tier, data in sorted(tiers.items()):
    gl = data.get('guardlens', {})
    dd = gl.get('deviation_drops', {}).get('15%', 0)
    n = gl.get('n_adversarial', 0)
    print(f'    {tier:<20} DD@15%={dd:.3f} (n={n})')
" 2>/dev/null || echo "    parse error"
fi

echo ""
echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"
