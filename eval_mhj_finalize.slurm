#!/bin/bash
#SBATCH --job-name=gl-public-finalize
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_public_finalize_%j.out
#SBATCH --error=logs/eval_public_finalize_%j.err

set -euo pipefail
: "${EVAL_DATA:?canonical dataset required}"
: "${EVAL_RUN_ROOT:?output directory required}"
: "${EVAL_SHARDS:?submit with submit_eval_mhj.sh}"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${EVAL_CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
cd "$SLURM_SUBMIT_DIR"
python -u -m eval_platform finalize --data "$EVAL_DATA" --output "$EVAL_RUN_ROOT" --shard-count "$EVAL_SHARDS"
