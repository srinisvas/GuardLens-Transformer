#!/bin/bash
#SBATCH --job-name=gl-dev-compare
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/compare_internal_dev_%j.out
#SBATCH --error=logs/compare_internal_dev_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
: "${MATRIX_ROOT:?matrix root required}"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

cd "$PROJECT_DIR"
OUTPUT="$MATRIX_ROOT/internal_dev_comparison.json"
[[ ! -e "$OUTPUT" ]] || { echo "ERROR: comparison output exists: $OUTPUT"; exit 2; }

python -m eval_platform compare-runs \
  --runs \
    "mean_frozen_aux=$MATRIX_ROOT/diagnostics/mean_frozen_aux" \
    "attention_frozen_aux=$MATRIX_ROOT/diagnostics/attention_frozen_aux" \
    "attention_top4_aux=$MATRIX_ROOT/diagnostics/attention_top4_aux" \
    "attention_top4_primary=$MATRIX_ROOT/diagnostics/attention_top4_primary" \
  --output "$OUTPUT"

echo "Internal dev matrix comparison: $OUTPUT"
echo "Held-out test accessed: NO"

