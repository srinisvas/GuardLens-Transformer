#!/bin/bash
#SBATCH --job-name=gl-dev-finalize
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_internal_dev_finalize_%j.out
#SBATCH --error=logs/eval_internal_dev_finalize_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
: "${EVAL_DATA:?prepared internal dev JSONL required}"
: "${EVAL_OUTPUT:?diagnostic output directory required}"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

cd "$PROJECT_DIR"
[[ -f "$EVAL_DATA" ]] || { echo "ERROR: missing data: $EVAL_DATA"; exit 2; }
[[ -f "$EVAL_OUTPUT/manifest.json" ]] || { echo "ERROR: missing diagnostic manifest"; exit 2; }

python -u -m eval_platform finalize \
  --data "$EVAL_DATA" \
  --output "$EVAL_OUTPUT" \
  --shard-count 1

echo "Internal dev diagnostic finalized: $EVAL_OUTPUT/report.json"
echo "Held-out test accessed: NO"
