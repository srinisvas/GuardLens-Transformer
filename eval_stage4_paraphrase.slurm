#!/bin/bash
#SBATCH --job-name=guardlens_paraphrase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_paraphrase_%j.out
#SBATCH --error=logs/eval_paraphrase_%j.err

# Uses 2 GPUs: GPU 0 for vLLM paraphrase server, GPU 1 for GuardLens eval.
# vLLM server is started as a background process and killed at the end.

set -euo pipefail
mkdir -p logs

echo "========================================================"
echo "  GuardLens Eval Stage 4: Paraphrase Robustness"
echo "  Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start: $(date)"
echo "========================================================"

source ~/.bashrc
conda activate guardlens 2>/dev/null || source ~/work/venv/guardlens/bin/activate

# Install openai client if not present (needed for vLLM backend)
pip install openai --quiet --break-system-packages 2>/dev/null || true

RESULTS_DIR=~/work/results/dataset_gen
CKPT_DIR=$RESULTS_DIR/checkpoints
DATA=$RESULTS_DIR/semantic_multiturn_v10_augmented.jsonl
OUT_DIR=$RESULTS_DIR/results
mkdir -p $OUT_DIR

cd ~/work/GuardLens-Transformer

# -------------------------------------------------------
# Start vLLM server on GPU 0
# -------------------------------------------------------
VLLM_PORT=8742  # Use non-default port to avoid conflicts
VLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
VLLM_LOG="$OUT_DIR/vllm_server_${SLURM_JOB_ID}.log"

echo ""
echo "Starting vLLM server (GPU 0, port $VLLM_PORT)..."
echo "Model: $VLLM_MODEL"

# Check if vLLM is available
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "vLLM not available. Installing..."
    pip install vllm --quiet --break-system-packages || {
        echo "vLLM install failed. Falling back to HuggingFace backend."
        USE_HF=1
    }
fi

USE_HF=${USE_HF:-0}
VLLM_PID=""

if [ "$USE_HF" = "0" ]; then
    # Start vLLM on GPU 0
    CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --port $VLLM_PORT \
        --tensor-parallel-size 1 \
        --dtype float16 \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.85 \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "  vLLM server PID: $VLLM_PID"

    # Wait for server to be ready (up to 3 minutes)
    echo "  Waiting for vLLM server..."
    MAX_WAIT=180
    WAIT=0
    until curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
        sleep 5
        WAIT=$((WAIT + 5))
        if [ $WAIT -ge $MAX_WAIT ]; then
            echo "  WARNING: vLLM server did not start in ${MAX_WAIT}s. Falling back to HF."
            kill $VLLM_PID 2>/dev/null || true
            VLLM_PID=""
            USE_HF=1
            break
        fi
        echo "  ... waiting ${WAIT}s"
    done

    if [ "$USE_HF" = "0" ]; then
        echo "  vLLM server ready."
        PARAPHRASE_BACKEND="vllm"
        VLLM_URL="http://localhost:$VLLM_PORT/v1"
        EVAL_GPU="CUDA_VISIBLE_DEVICES=1"
    fi
fi

if [ "$USE_HF" = "1" ]; then
    echo "  Using HuggingFace backend (both GPUs available for eval)."
    PARAPHRASE_BACKEND="hf"
    VLLM_URL=""
    EVAL_GPU=""
fi

# Cleanup function
cleanup() {
    if [ -n "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# -------------------------------------------------------
# 4a. Full test set (all adversarial)
# -------------------------------------------------------
echo ""
echo "=== 4a. Paraphrase robustness (full adversarial test set) ==="
echo "Start: $(date)"

eval $EVAL_GPU python -m guardlens.evaluation.eval_paraphrase \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/paraphrase_eval_full.json" \
    --n-samples 150 \
    --paraphrase-model "$PARAPHRASE_BACKEND" \
    --vllm-model "$VLLM_MODEL" \
    ${VLLM_URL:+--vllm-url "$VLLM_URL"} \
    --top-k 0.15 \
    --save-examples 10 \
    --device cuda

echo "4a complete: $(date)"

# -------------------------------------------------------
# 4b. Implicit trigger subset
# -------------------------------------------------------
echo ""
echo "=== 4b. Paraphrase robustness (implicit triggers only) ==="
echo "Start: $(date)"

eval $EVAL_GPU python -m guardlens.evaluation.eval_paraphrase \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/paraphrase_eval_implicit.json" \
    --n-samples 80 \
    --subset implicit \
    --paraphrase-model "$PARAPHRASE_BACKEND" \
    --vllm-model "$VLLM_MODEL" \
    ${VLLM_URL:+--vllm-url "$VLLM_URL"} \
    --top-k 0.15 \
    --save-examples 5 \
    --device cuda

echo "4b complete: $(date)"

# -------------------------------------------------------
# 4c. Explicit trigger subset
# -------------------------------------------------------
echo ""
echo "=== 4c. Paraphrase robustness (explicit triggers only) ==="
echo "Start: $(date)"

eval $EVAL_GPU python -m guardlens.evaluation.eval_paraphrase \
    --data "$DATA" \
    --checkpoint "$CKPT_DIR/guardlens/best.pt" \
    --output "$OUT_DIR/paraphrase_eval_explicit.json" \
    --n-samples 80 \
    --subset explicit \
    --paraphrase-model "$PARAPHRASE_BACKEND" \
    --vllm-model "$VLLM_MODEL" \
    ${VLLM_URL:+--vllm-url "$VLLM_URL"} \
    --top-k 0.15 \
    --save-examples 5 \
    --device cuda

echo "4c complete: $(date)"

# -------------------------------------------------------
# Summarise paraphrase results
# -------------------------------------------------------
echo ""
echo "=== Paraphrase robustness summary ==="
python3 - <<'PYEOF'
import json, os

OUT_DIR = os.path.expanduser("~/work/results/dataset_gen/results")
subsets = [
    ("full", "paraphrase_eval_full.json"),
    ("implicit", "paraphrase_eval_implicit.json"),
    ("explicit", "paraphrase_eval_explicit.json"),
]

print(f"\n{'='*65}")
print(f"  Paraphrase Robustness Summary")
print(f"{'='*65}")
print(f"  {'Subset':<15} {'N':>6} {'TurnRho':>10} {'TokRho':>10} {'TopK%':>10} {'Pivot%':>10}")
print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for subset_name, fname in subsets:
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        print(f"  {subset_name:<15} MISSING")
        continue
    with open(path) as f:
        data = json.load(f)
    agg = data.get("aggregate", {})
    n = data.get("n_samples", 0)
    print(f"  {subset_name:<15} {n:>6} "
          f"{agg.get('turn_rho_mean', 0):>10.3f} "
          f"{agg.get('token_rho_mean', 0):>10.3f} "
          f"{agg.get('topk_stability', 0):>10.3f} "
          f"{agg.get('pivot_stability', 0):>10.3f}")

print(f"\n  Interpretation:")
print(f"    Turn/Token Spearman rho: > 0.60 = attribution is semantic, not lexical")
print(f"    Top-k stability: fraction of top-k tokens shared between original and paraphrase")
print(f"    Pivot stability: same turn identified as adversarial pivot after paraphrase")
PYEOF

echo ""
echo "========================================================"
echo "  Stage 4 complete: $(date)"
echo "========================================================"
