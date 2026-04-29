#!/bin/bash
# ============================================================
# setup_guardlens_env.sh
#
# Creates conda environment for GuardLens model training.
# Installs to ~/work/conda_envs/ (GPFS data, plenty of space).
#
# Usage:
#   bash setup_guardlens_env.sh
# ============================================================

set -euo pipefail

ENV_PREFIX="$HOME/work/conda_envs/guardlens_train"

export PIP_CACHE_DIR="$HOME/work/.pip_cache"
export TMPDIR="$HOME/work/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

echo "=== GuardLens training environment ==="
echo "  Prefix: $ENV_PREFIX"

CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ -d "$ENV_PREFIX" ]]; then
    echo "  Environment exists. Activating..."
    conda activate "$ENV_PREFIX"
else
    echo "  Creating environment..."
    conda create --prefix "$ENV_PREFIX" python=3.11 -y
    conda activate "$ENV_PREFIX"
fi

echo ""
echo "Installing packages (pre-built wheels only)..."

pip install --upgrade pip
pip install torch --only-binary=:all:
pip install transformers>=4.40.0 accelerate>=0.28.0 numpy>=1.24.0 requests>=2.31.0

echo ""
echo "--- Verification ---"
python3 -c "import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'  transformers {transformers.__version__}')"

echo ""
echo "=== Ready ==="
echo "  Activate: conda activate $ENV_PREFIX"
echo ""
echo "  Pre-download DeBERTa backbone (run once):"
echo "    python -c \"from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/deberta-v3-base'); AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')\""
