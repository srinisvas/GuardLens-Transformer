#!/bin/bash
#SBATCH --job-name=gl_naacl_smoke
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=logs/naacl_window_smoke_%j.out
#SBATCH --error=logs/naacl_window_smoke_%j.err

# Mandatory runtime gate before train_naacl.slurm. Exercises one phase-3
# GuardLens optimizer step on the longest malicious and benign training records.
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
SPLIT_DIR="${SPLIT_DIR:-$HOME/staging/dataset_gen_output/naacl_splits}"
TRAIN_PATH="${TRAIN_PATH:-$SPLIT_DIR/train.jsonl}"
MAX_TURNS="${MAX_TURNS:-48}"
MAX_TOKENS="${MAX_TOKENS:-192}"
BATCH_SIZE="${BATCH_SIZE:-2}"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export TMPDIR="${TMPDIR:-$HOME/work/.tmp}"
export HF_HOME="${HF_HOME:-$HOME/work/hf_models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$TMPDIR" logs
cd "$PROJECT_DIR"

[[ -f "$TRAIN_PATH" ]] || { echo "ERROR: missing train split: $TRAIN_PATH"; exit 2; }

echo "========================================================"
echo " GuardLens NAACL 48-turn phase-3 memory smoke"
echo " Train: $TRAIN_PATH"
echo " max_turns=$MAX_TURNS max_tokens=$MAX_TOKENS batch=$BATCH_SIZE"
echo "========================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

python -m guardlens.evaluation.smoke_training_window \
  --train "$TRAIN_PATH" \
  --backbone microsoft/deberta-v3-base \
  --max-turns "$MAX_TURNS" \
  --max-tokens "$MAX_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --seed 42
