#!/bin/bash
#SBATCH --job-name=target_causal_v3_dbg
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home/s001/ssubram7/work/results/guardlens_v11/logs/target_causal_v3_debug_%j.out
#SBATCH --error=/gpfs/home/s001/ssubram7/work/results/guardlens_v11/logs/target_causal_v3_debug_%j.err

echo "========================================================"
echo "  Target-LLM Causal v3 Debug (val_response_map)"
echo "  Job: $SLURM_JOB_ID  Node: $(hostname -s)"
echo "  $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$HOME/work/conda_envs/guardlens_train"

cd /gpfs/home/s001/ssubram7/projects/GuardLens-Transformer

TEST_PATH="$HOME/staging/dataset_gen_output/splits/test.jsonl"
GL_ATTR="$HOME/work/results/guardlens_v11/checkpoints/guardlens/best_attribution.pt"
OUT_DIR="$HOME/work/results/guardlens_v11/results/review_response"
mkdir -p "$OUT_DIR"

python -m guardlens.evaluation.eval_target_llm_causal \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --target-model "Qwen/Qwen2.5-7B-Instruct" \
    --judge-model "$HOME/work/hf_models/llama3-8b-instruct" \
    --output "$OUT_DIR/target_llm_causal_v3_debug10.json" \
    --temperature 0.7 \
    --system-prompt "" \
    --n-samples 3 \
    --max-conversations 10 \
    --batch-size 4 \
    --device cuda

echo ""
echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"
