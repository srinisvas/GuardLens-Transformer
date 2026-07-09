#!/bin/bash
#SBATCH --job-name=gl-bootstrap-ci
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bootstrap_ci_%j.out
#SBATCH --error=logs/bootstrap_ci_%j.err

# No GPU needed — pure numpy computation.

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11/results"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
mkdir -p "$TMPDIR" logs

cd "$SLURM_SUBMIT_DIR"

echo "========================================================"
echo "  GuardLens Bootstrap CI Computation"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"

CAUSAL="$RESULTS_DIR/causal_eval_results.json"
BOUNDARY="$RESULTS_DIR/boundary_stress.json"
HUMAN="$RESULTS_DIR/human_benchmark_eval.json"
OUTPUT="$RESULTS_DIR/bootstrap_cis.json"

for f in "$CAUSAL" "$BOUNDARY" "$HUMAN"; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"; exit 1
    fi
    echo "  OK: $(basename $f)"
done

echo ""
python compute_bootstrap_cis.py \
    --causal    "$CAUSAL" \
    --boundary  "$BOUNDARY" \
    --human     "$HUMAN" \
    --output    "$OUTPUT" \
    --n-bootstrap 10000 \
    --seed 42 \
    --ci 0.95

EXIT_CODE=$?

echo ""
echo "========================================================"
if [ $EXIT_CODE -eq 0 ] && [ -f "$OUTPUT" ]; then
    echo "  SUCCESS: $OUTPUT"
else
    echo "  FAILED (exit ${EXIT_CODE})"
fi
echo "  Done: $(date)"
echo "========================================================"
