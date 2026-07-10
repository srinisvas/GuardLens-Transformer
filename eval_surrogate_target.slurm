#!/bin/bash
#SBATCH --job-name=gl-target-causal-v2
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=logs/target_causal_v2_%j.out
#SBATCH --error=logs/target_causal_v2_%j.err

# ============================================================
# Cached-original anchored target-LLM counterfactual test
#
# Previous run: Llama-3-8B-Instruct refused all 87 original
# replays (0% unsafe rate) due to wrong settings (temp 0.01,
# system prompt) and possible checkpoint drift.
#
# This run:
#   - Uses cached unsafe response as original anchor
#   - Correct generation settings (temp 0.7, no sys prompt,
#     max_new_tokens 300)
#   - Llama-3-8B-Instruct as target (original generation model)
#   - Qwen2.5-7B-Instruct as independent judge
#   - K=3 repeated samples per neutralized condition
#   - Random baseline restricted to pre-outcome turns
# ============================================================

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"

TEST_PATH="$HOME/staging/dataset_gen_output/splits/test.jsonl"
CKPT_DIR="$HOME/work/results/guardlens_v11/checkpoints"
OUT_DIR="$HOME/work/results/guardlens_v11/results/review_response"

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
echo "  Cached-Original Target-LLM Counterfactual Test"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GL_ATTR=$(ls "$CKPT_DIR/guardlens/best_attribution.pt" 2>/dev/null || echo "")
if [ -z "$GL_ATTR" ]; then
    echo "FATAL: No GuardLens checkpoint"; exit 1
fi

TARGET_MODEL="$HOME/work/hf_models/llama3-8b-instruct"
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "  Target (original): $TARGET_MODEL"
echo "  Judge: $JUDGE_MODEL"
echo "  Checkpoint: $GL_ATTR"

python -m guardlens.evaluation.eval_target_llm_causal \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --target-model "$TARGET_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --output "$OUT_DIR/target_llm_causal_v2.json" \
    --temperature 0.7 \
    --system-prompt "" \
    --n-samples 3 \
    --max-conversations 87 \
    --batch-size 4 \
    --device cuda

echo ""
echo "========================================================"
echo "  Results Summary"
echo "========================================================"

if [ -f "$OUT_DIR/target_llm_causal_v2.json" ]; then
    python3 -c "
import json
r = json.load(open('$OUT_DIR/target_llm_causal_v2.json'))
m = r.get('metrics', {})
print(f'  Total processed: {m.get(\"n_total\", 0)}')
print(f'  Live unsafe rate: {m.get(\"live_unsafe_rate\", 0):.3f} ({m.get(\"n_live_unsafe\",0)}/{m.get(\"n_total\",0)})')
print()
for method in ['guardlens', 'surface_risk', 'random']:
    d = m.get(method, {})
    fs = d.get('mean_flip_strength', 0)
    ci = d.get('ci_95', [0, 0])
    print(f'  {method:<20} flip_strength={fs:.3f} 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] n={d.get(\"n\",0)}')
print()
for control in ['surface_risk', 'random']:
    p = m.get(f'paired_vs_{control}', {})
    ci = p.get('ci_95', [0, 0])
    sig = ' *' if ci[0] > 0 else ''
    print(f'  GL - {control:<15} delta={p.get(\"mean_delta\",0):+.3f} 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] n={p.get(\"n\",0)}{sig}')
" || echo "  parse error"
fi

echo ""
echo "  Done: $(date)"
