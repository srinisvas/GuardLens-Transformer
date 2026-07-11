#!/bin/bash
#SBATCH --job-name=gl-target-causal-v3-qwen
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=/gpfs/home/s001/ssubram7/work/results/guardlens_v11/logs/target_causal_v3_qwen_%j.out
#SBATCH --error=/gpfs/home/s001/ssubram7/work/results/guardlens_v11/logs/target_causal_v3_qwen_%j.err

# ============================================================
# Cached-original anchored target-LLM counterfactual test
#
# This run:
#   - Uses Qwen2.5-7B-Instruct as the target for qwen7b-success records
#   - Uses Llama-3-8B-Instruct as an independent judge
#   - Uses cached qwen7b_validation response snippets to reconstruct
#     target-specific assistant history where available
#   - Uses generation settings: temp 0.7, no system prompt, max_new_tokens 300
#   - Uses K=3 repeated samples per neutralized condition
#   - Restricts random baseline to user turns before or at the outcome request
# ============================================================

set -euo pipefail

PROJECT_DIR="/gpfs/home/s001/ssubram7/projects/GuardLens-Transformer"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"

TEST_PATH="$HOME/staging/dataset_gen_output/splits/test.jsonl"
CKPT_DIR="$HOME/work/results/guardlens_v11/checkpoints"
OUT_DIR="$HOME/work/results/guardlens_v11/results/review_response"
LOG_DIR="$HOME/work/results/guardlens_v11/logs"

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_DATASETS_CACHE="$HOME/work/hf_models/datasets"
export HF_HUB_OFFLINE=1

mkdir -p "$TMPDIR" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$OUT_DIR" "$LOG_DIR"

echo "========================================================"
echo "  Cached-Original Target-LLM Counterfactual Test v3"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

cd "$PROJECT_DIR"

GL_ATTR=$(ls "$CKPT_DIR/guardlens/best_attribution.pt" 2>/dev/null || true)
if [ -z "$GL_ATTR" ]; then
    echo "FATAL: No GuardLens checkpoint found at $CKPT_DIR/guardlens/best_attribution.pt"
    exit 1
fi

QWEN_PATH=$(ls -d "$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"* 2>/dev/null | head -1 || true)
if [ -z "$QWEN_PATH" ]; then
    echo "FATAL: Qwen snapshot not found under $TRANSFORMERS_CACHE"
    echo "Available Qwen-related cache paths:"
    find "$HF_HOME" -maxdepth 6 -iname "*qwen*" -print || true
    exit 1
fi

TARGET_MODEL="$QWEN_PATH"
JUDGE_MODEL="$HOME/work/hf_models/llama3-8b-instruct"
OUTPUT_JSON="$OUT_DIR/target_llm_causal_v3_qwen.json"

if [ ! -d "$JUDGE_MODEL" ]; then
    echo "FATAL: Judge model path not found: $JUDGE_MODEL"
    exit 1
fi

echo "  Project dir: $PROJECT_DIR"
echo "  Test path:   $TEST_PATH"
echo "  Checkpoint:  $GL_ATTR"
echo "  Target:      $TARGET_MODEL"
echo "  Judge:       $JUDGE_MODEL"
echo "  Output:      $OUTPUT_JSON"
echo "  HF_HOME:     $HF_HOME"
echo "  Cache:       $TRANSFORMERS_CACHE"
echo "  Offline:     $HF_HUB_OFFLINE"
echo "========================================================"

python -m guardlens.evaluation.eval_target_llm_causal \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --target-model "$TARGET_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --output "$OUTPUT_JSON" \
    --temperature 0.7 \
    --system-prompt "" \
    --n-samples 3 \
    --max-conversations 87 \
    --batch-size 4 \
    --device cuda \
    --cache-dir "$TRANSFORMERS_CACHE"

echo ""
echo "========================================================"
echo "  Results Summary"
echo "========================================================"

if [ ! -f "$OUTPUT_JSON" ]; then
    echo "FATAL: Output file was not created: $OUTPUT_JSON"
    exit 1
fi

ls -lh "$OUTPUT_JSON"

python - <<PY
import json, os

p = os.path.expanduser("$OUTPUT_JSON")
r = json.load(open(p))
m = r.get("metrics", {})

print(f"  Total processed: {m.get('n_total', 0)}")
print(f"  Live unsafe rate: {m.get('live_unsafe_rate', 0):.3f} ({m.get('n_live_unsafe',0)}/{m.get('n_total',0)})")
print()

for method in ["guardlens", "surface_risk", "random"]:
    d = m.get(method, {})
    fs = d.get("mean_flip_strength", 0)
    ci = d.get("ci_95", [0, 0])
    print(f"  {method:<20} flip_strength={fs:.3f} 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] n={d.get('n',0)}")

print()

for control in ["surface_risk", "random"]:
    p = m.get(f"paired_vs_{control}", {})
    ci = p.get("ci_95", [0, 0])
    sig = " *" if ci[0] > 0 else ""
    print(f"  GL - {control:<15} delta={p.get('mean_delta',0):+.3f} 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] n={p.get('n',0)}{sig}")

# Quick debug signal
records = r.get("per_record", [])
if records:
    used = [x.get("n_val_responses_used", 0) for x in records]
    print()
    print(f"  Avg validation responses used: {sum(used)/len(used):.2f}")
    print(f"  Max validation responses used: {max(used)}")
PY

echo ""
echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"