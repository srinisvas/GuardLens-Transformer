#!/bin/bash
#SBATCH --job-name=gl-dev-diag
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_internal_dev_%j.out
#SBATCH --error=logs/eval_internal_dev_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
: "${EVAL_DATA:?prepared internal dev JSONL required}"
: "${EVAL_CHECKPOINT:?trained checkpoint required}"
: "${EVAL_OUTPUT:?fresh diagnostic output directory required}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-configs/eval_protocol_retrospective.json}"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export TMPDIR="${TMPDIR:-$HOME/work/.tmp}"
export HF_HOME="${GUARDLENS_HF_HOME:-$HOME/work/hf_models}"
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" logs

cd "$PROJECT_DIR"
for path in "$EVAL_DATA" "$EVAL_DATA.manifest.json" "$EVAL_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path"; exit 2; }
done
if [[ -e "$EVAL_OUTPUT" ]]; then
  [[ -f "$EVAL_OUTPUT/manifest.json" ]] || {
    echo "ERROR: existing diagnostic output has no manifest: $EVAL_OUTPUT"
    exit 2
  }
  echo "Resuming exact manifest-bound diagnostic: $EVAL_OUTPUT"
fi
[[ -f "$EVAL_PROTOCOL" ]] || { echo "ERROR: missing protocol: $EVAL_PROTOCOL"; exit 2; }

EXPECTED_BRANCH="naacl-causal-localization-redesign"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CODE_SHA=$(git rev-parse HEAD)
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected git branch $EXPECTED_BRANCH, got $CURRENT_BRANCH"
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked working-tree changes detected; commit or stash them before a scientific run"
  exit 2
fi

EVAL_DATA_SHA256=$(sha256sum "$EVAL_DATA" | awk '{print $1}')
echo "Code SHA: $CODE_SHA"
echo "Checkpoint: $EVAL_CHECKPOINT"
echo "Internal dev: $EVAL_DATA"
echo "Output: $EVAL_OUTPUT"
echo "Protocol: $EVAL_PROTOCOL"
echo "External guards: DISABLED"
echo "Held-out test: NOT ACCESSED"

python -u -m eval_platform diagnose-dev \
  --data "$EVAL_DATA" \
  --data-sha256 "$EVAL_DATA_SHA256" \
  --checkpoint "$EVAL_CHECKPOINT" \
  --output "$EVAL_OUTPUT" \
  --protocol "$EVAL_PROTOCOL" \
  --lexicon configs/eval_surface_lexicon.json \
  --device cuda \
  --shard-index 0 \
  --shard-count 1
