#!/bin/bash
#SBATCH --job-name=guardlens_eval_setup
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_setup_%j.out
#SBATCH --error=logs/eval_setup_%j.err

set -euo pipefail
mkdir -p logs

echo "========================================================"
echo "  GuardLens Eval Stage 1: Data Setup"
echo "  Job: $SLURM_JOB_ID"
echo "  Start: $(date)"
echo "========================================================"

source ~/.bashrc
conda activate guardlens 2>/dev/null || source ~/work/venv/guardlens/bin/activate

# Install datasets library if not present (for ToxicChat)
pip install datasets --quiet --break-system-packages 2>/dev/null || true

RESULTS_DIR=~/work/results/dataset_gen

# Verify training checkpoints exist
echo "Verifying training checkpoints..."
for model in guardlens guardlens_no_fusion guardlens_no_cf turn_level conversation_deberta; do
    ckpt="$RESULTS_DIR/checkpoints/$model/best.pt"
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Missing checkpoint: $ckpt"
        exit 1
    fi
    echo "  OK: $model/best.pt"
done

# Run data download
echo ""
echo "Downloading external evaluation datasets..."
bash ~/work/GuardLens-Transformer/setup_eval_data.sh \
    "$RESULTS_DIR/eval_data"

echo ""
echo "Setup complete: $(date)"
echo "Data directory: $RESULTS_DIR/eval_data"
ls -la "$RESULTS_DIR/eval_data/"
