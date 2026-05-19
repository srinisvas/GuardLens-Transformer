#!/bin/bash
#SBATCH --job-name=gl-transfer-stable
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_transfer_stable_%j.out
#SBATCH --error=logs/eval_transfer_stable_%j.err

# ============================================================
# Stabilized ShieldGemma transfer evaluation
#
# Uses final_dataset.jsonl (not test split) to maximize the
# number of adversarial records available for subset filtering.
#
# Runs:
#   A. transfer_success subset (all available, k=0.15)
#   B. lexical_pivot subset (all available, k=0.15)
#   C. contextual_pivot subset (all available, k=0.15)
#   D. Threshold sensitivity on transfer_success (0.3, 0.4, 0.5)
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT="$RESULTS_DIR/checkpoints/guardlens/best.pt"
# Use final_dataset for maximum adversarial coverage
TEST_DATA="$HOME/staging/dataset_gen_output/final_dataset.jsonl"
OUT_DIR="$RESULTS_DIR/results"

CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT_DIR" logs

# ---- Preflight ----
echo "========================================================"
echo "  Stabilized ShieldGemma Transfer Evaluation"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Missing checkpoint: $CKPT"
    exit 1
fi
if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Missing data: $TEST_DATA"
    exit 1
fi

echo "Checkpoint: $CKPT"
echo "Data: $TEST_DATA ($(wc -l < "$TEST_DATA") records)"
echo ""

# Count available subsets
python3 -c "
import json
from collections import Counter
records = []
with open('$TEST_DATA') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
mal = [r for r in records if r.get('label') == 1]
print(f'Total: {len(records)}, Malicious: {len(mal)}')
print(f'Transfer tiers: {Counter(r.get(\"transfer_tier\",\"?\") for r in mal)}')
print(f'Pivot kinds: {Counter(r.get(\"pivot_kind\",\"?\") for r in mal)}')
"

# ---- A. Transfer success subset ----
echo ""
echo "========================================================"
echo "  A. Transfer success subset"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_cross_model_transfer \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT" \
    --output "$OUT_DIR/shieldgemma_transfer_success.json" \
    --external-model google/shieldgemma-9b \
    --n-samples 300 \
    --k-frac 0.15 \
    --subset transfer_success \
    --threshold 0.5 \
    --variants-cache "$OUT_DIR/variants_transfer_success.jsonl" \
    --device cuda \
&& echo "A: SUCCESS" || echo "A: FAILED"
echo "A end: $(date +%H:%M:%S)"

# ---- B. Lexical pivot subset ----
echo ""
echo "========================================================"
echo "  B. Lexical pivot subset"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_cross_model_transfer \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT" \
    --output "$OUT_DIR/shieldgemma_lexical_pivot.json" \
    --external-model google/shieldgemma-9b \
    --n-samples 200 \
    --k-frac 0.15 \
    --subset lexical_pivot \
    --threshold 0.5 \
    --variants-cache "$OUT_DIR/variants_lexical_pivot.jsonl" \
    --device cuda \
&& echo "B: SUCCESS" || echo "B: FAILED"
echo "B end: $(date +%H:%M:%S)"

# ---- C. Contextual pivot subset ----
echo ""
echo "========================================================"
echo "  C. Contextual pivot subset"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_cross_model_transfer \
    --test-path "$TEST_DATA" \
    --checkpoint "$CKPT" \
    --output "$OUT_DIR/shieldgemma_contextual_pivot.json" \
    --external-model google/shieldgemma-9b \
    --n-samples 200 \
    --k-frac 0.15 \
    --subset contextual_pivot \
    --threshold 0.5 \
    --variants-cache "$OUT_DIR/variants_contextual_pivot.jsonl" \
    --device cuda \
&& echo "C: SUCCESS" || echo "C: FAILED"
echo "C end: $(date +%H:%M:%S)"

# ---- D. Threshold sensitivity on transfer_success ----
echo ""
echo "========================================================"
echo "  D. Threshold sensitivity (0.3, 0.4, 0.5)"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

for THRESH in 0.3 0.4 0.5; do
    echo ""
    echo "--- Threshold: $THRESH ---"
    python -m guardlens.evaluation.eval_cross_model_transfer \
        --test-path "$TEST_DATA" \
        --checkpoint "$CKPT" \
        --output "$OUT_DIR/shieldgemma_thresh_${THRESH}.json" \
        --external-model google/shieldgemma-9b \
        --n-samples 300 \
        --k-frac 0.15 \
        --subset transfer_success \
        --threshold "$THRESH" \
        --variants-cache "$OUT_DIR/variants_transfer_success.jsonl" \
        --device cuda \
    && echo "  Threshold $THRESH: SUCCESS" || echo "  Threshold $THRESH: FAILED"
done
echo "D end: $(date +%H:%M:%S)"

# ---- Summary ----
echo ""
echo "========================================================"
echo "  Transfer Eval Summary"
echo "========================================================"

for f in shieldgemma_transfer_success.json shieldgemma_lexical_pivot.json \
         shieldgemma_contextual_pivot.json \
         shieldgemma_thresh_0.3.json shieldgemma_thresh_0.4.json shieldgemma_thresh_0.5.json; do
    RESULT="$OUT_DIR/$f"
    if [ -f "$RESULT" ]; then
        echo ""
        echo "  $f:"
        python3 -c "
import json
r = json.load(open('$RESULT'))
tfr = r.get('transfer_flip_rates', r.get('results', r))
for method in ['guardlens', 'surface_risk', 'random']:
    data = tfr.get(method, {})
    tf = data.get('transfer_flip_rate', '?')
    flipped = data.get('n_flipped', data.get('flipped', '?'))
    tested = data.get('n_tested', data.get('tested', '?'))
    n_orig = data.get('n_compliant_original', '?')
    print(f'    {method:<20} TransferFlip={tf}  flipped={flipped}/{tested}  compliant_orig={n_orig}')
thresh = r.get('threshold', r.get('shieldgemma_threshold', '?'))
print(f'    threshold={thresh}')
" 2>/dev/null || echo "    (parse error)"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "========================================================"
echo "  Done: $(date +%H:%M:%S)"
echo "========================================================"
