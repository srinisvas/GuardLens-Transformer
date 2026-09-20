#!/bin/bash
# ============================================================
# setup_guardlens_env.sh
#
# Creates or repairs the canonical GuardLens training environment.
#
# Default:
#   bash setup_guardlens_env.sh
#       Reuses the existing prefix (if present) and installs/verifies the full
#       requirements set.
#
# Clean rebuild:
#   bash setup_guardlens_env.sh --recreate
#
# Environment prefix:
#   ~/work/conda_envs/guardlens_train
# ============================================================

set -euo pipefail

ENV_PREFIX="${GUARDLENS_ENV_PREFIX:-$HOME/work/conda_envs/guardlens_train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
RECREATE=0

usage() {
    cat <<'EOF'
Usage: bash setup_guardlens_env.sh [--recreate]

  --recreate   Remove the existing GuardLens conda prefix and build it again.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recreate)
            RECREATE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -f "$REQUIREMENTS" ]] || {
    echo "ERROR: requirements.txt not found at $REQUIREMENTS" >&2
    exit 2
}

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/work/.pip_cache}"
export TMPDIR="${TMPDIR:-$HOME/work/.tmp}"

# GuardLens owns one canonical Hugging Face cache. Do not inherit generic
# HF_HOME/HF_HUB_CACHE values from an interactive shell, because SLURM jobs
# may otherwise look in a different cache than environment setup populated.
export HF_HOME="${GUARDLENS_HF_HOME:-$HOME/work/hf_models}"
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HUB_CACHE"

echo "========================================================"
echo " GuardLens environment setup"
echo " Prefix:       $ENV_PREFIX"
echo " Requirements: $REQUIREMENTS"
echo " Recreate:     $RECREATE"
echo " HF_HOME:      $HF_HOME"
echo " HF_HUB_CACHE: $HF_HUB_CACHE"
echo "========================================================"

CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ "$RECREATE" -eq 1 && -d "$ENV_PREFIX" ]]; then
    if [[ "${CONDA_PREFIX:-}" == "$ENV_PREFIX" ]]; then
        conda deactivate || true
    fi
    echo "Removing existing environment..."
    conda env remove --prefix "$ENV_PREFIX" -y
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "Creating Python 3.11 environment..."
    conda create --prefix "$ENV_PREFIX" python=3.11 -y
fi

conda activate "$ENV_PREFIX"

echo
echo "=== Python / pip ==="
which python
python --version
python -m pip --version

echo
echo "=== Installing binary tooling ==="
python -m pip install --upgrade pip setuptools wheel

# Install torch first. This keeps Torch resolution separate from the rest of the
# environment and avoids an evaluator dependency unexpectedly choosing it.
echo
echo "=== Installing PyTorch ==="
python -m pip install --upgrade --only-binary=:all: "torch>=2.2,<3"

echo
echo "=== Installing GuardLens dependencies ==="
python -m pip install --upgrade -r "$REQUIREMENTS"

echo
echo "=== Dependency consistency ==="
python -m pip check

echo
echo "=== Import verification ==="
python - <<'PY'
import importlib

required = [
    "torch",
    "transformers",
    "accelerate",
    "tokenizers",
    "numpy",
    "scipy",
    "requests",
    "matplotlib",
]

for name in required:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "<unknown>")
    print(f"  {name:14s} {version}")

import torch
print(f"  torch CUDA build:     {torch.version.cuda}")
print(f"  torch CUDA available: {torch.cuda.is_available()}")
if torch.version.cuda is None:
    raise RuntimeError(
        "CPU-only PyTorch build detected; GuardLens canonical training requires "
        "a CUDA-enabled PyTorch wheel"
    )
if torch.cuda.is_available():
    print(f"  GPU:                  {torch.cuda.get_device_name(0)}")
else:
    print("  GPU visibility:       none on this node (acceptable for login-node setup)")
PY

echo
echo "=== ModernBERT tokenizer/config verification ==="
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

name = "answerdotai/ModernBERT-large"
revision = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
tokenizer = AutoTokenizer.from_pretrained(
    name, revision=revision, use_fast=True
)
config = AutoConfig.from_pretrained(name, revision=revision)

if not getattr(tokenizer, "is_fast", False):
    raise RuntimeError("ModernBERT tokenizer is not fast; offset mapping is required")

probe = tokenizer(
    "GuardLens tokenizer offset verification.",
    truncation=False,
    return_offsets_mapping=True,
    add_special_tokens=True,
)
if "offset_mapping" not in probe:
    raise RuntimeError("Tokenizer did not return offset mappings")

limit = getattr(config, "max_position_embeddings", None)
hidden = getattr(config, "hidden_size", None)
if limit != 8192:
    raise RuntimeError(
        f"Unexpected ModernBERT max_position_embeddings={limit}; expected 8192"
    )
if hidden != 1024:
    raise RuntimeError(
        f"Unexpected ModernBERT hidden_size={hidden}; expected 1024"
    )

print(f"  tokenizer: {tokenizer.__class__.__name__}")
print(f"  fast:      {tokenizer.is_fast}")
print(f"  positions: {limit}")
print(f"  hidden:    {hidden}")
print(f"  offsets:   OK ({len(probe['offset_mapping'])} entries)")
PY

echo
echo "=== Caching ModernBERT-large weights ==="
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="answerdotai/ModernBERT-large",
    revision="45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "tokenizer*",
        "special_tokens_map.json",
    ],
)
print("  ModernBERT-large snapshot cached")
PY

echo
echo "=== GuardLens package import verification ==="
cd "$SCRIPT_DIR"
python - <<'PY'
from guardlens import GuardLens, GuardLensConfig
from guardlens.data.causal_targets import build_evidence_turn_targets
from guardlens.training.schedule import get_lambda_schedule

cfg = GuardLensConfig()
assert cfg.backbone_name == "answerdotai/ModernBERT-large"
assert cfg.backbone_revision == "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
assert cfg.backbone_dim == 1024
assert cfg.max_tokens_per_turn == 8192
assert get_lambda_schedule(5, cfg)[1] == 0.25
assert get_lambda_schedule(9, cfg)[1] == 1.0
print("  GuardLens imports: OK")
print("  training schedule: OK")
PY

echo
echo "========================================================"
echo " Environment ready"
echo " Activate with:"
echo "   conda activate $ENV_PREFIX"
echo "========================================================"
