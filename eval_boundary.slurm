#!/bin/bash
#SBATCH --job-name=gl-boundary
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --output=logs/eval_boundary_%j.out
#SBATCH --error=logs/eval_boundary_%j.err

set -uo pipefail

CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

echo "========================================================"
echo "  Boundary Stress Test"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "========================================================"

python -m guardlens.evaluation.eval_boundary_stress \
    --boundary-files $HOME/staging/dataset_gen_output/benign_boundary.jsonl \
    --checkpoint $HOME/work/results/guardlens_v11/checkpoints/guardlens/best.pt \
    --output $HOME/work/results/guardlens_v11/results/boundary_stress.json \
    --device cuda

echo "Done: $(date)"

