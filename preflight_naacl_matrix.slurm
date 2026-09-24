#!/bin/bash
#SBATCH --job-name=gl-matrix-preflight
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preflight_naacl_matrix_%j.out
#SBATCH --error=logs/preflight_naacl_matrix_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
FREEZE_DIR="${FREEZE_DIR:-$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a}"
REPORT_PATH="${REPORT_PATH:-$FREEZE_DIR/data_prep_freeze_report.json}"
BACKBONE="${BACKBONE:-answerdotai/ModernBERT-large}"
BACKBONE_REVISION="${BACKBONE_REVISION:-45bb4654a4d5aaff24dd11d4781fa46d39bf8c13}"
MAX_TURNS="${MAX_TURNS:-64}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
INPUT_VIEW="${INPUT_VIEW:-pre_response}"
LENGTH_AUC_CEILING="${LENGTH_AUC_CEILING:-0.650}"
: "${SHARED_PREFLIGHT_DIR:?fresh shared preflight directory required}"
: "${PREPARED_DEV:?fresh prepared internal dev JSONL path required}"

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
[[ ! -e "$SHARED_PREFLIGHT_DIR" ]] || { echo "ERROR: shared preflight exists: $SHARED_PREFLIGHT_DIR"; exit 2; }
[[ ! -e "$PREPARED_DEV" ]] || { echo "ERROR: prepared dev exists: $PREPARED_DEV"; exit 2; }

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

echo "Running complete CPU contract suite before GPU submission"
python -m unittest discover -v -p 'test*.py'

mkdir -p "$SHARED_PREFLIGHT_DIR"
for variant in primary primary_plus_auxiliary; do
  if [[ "$variant" == "primary" ]]; then
    split_dir="$FREEZE_DIR/splits_primary"
  else
    split_dir="$FREEZE_DIR/splits_primary_plus_train_auxiliary"
  fi
  train_path="$split_dir/train.jsonl"
  dev_path="$split_dir/dev.jsonl"
  output_dir="$SHARED_PREFLIGHT_DIR/$variant"
  mkdir -p "$output_dir"

  python -m guardlens.data.verify_freeze \
    --report "$REPORT_PATH" \
    --train "$train_path" \
    --dev "$dev_path" \
    --variant "$variant" \
    --output "$output_dir/freeze_verification.json"

  python -m guardlens.data.audit_representation \
    --train "$train_path" \
    --dev "$dev_path" \
    --backbone "$BACKBONE" \
    --backbone-revision "$BACKBONE_REVISION" \
    --max-turns "$MAX_TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --input-view "$INPUT_VIEW" \
    --output "$output_dir/representation_coverage.json"

  python -m guardlens.evaluation.eval_length_probe_preflight \
    --train "$train_path" \
    --dev "$dev_path" \
    --input-view "$INPUT_VIEW" \
    --output "$output_dir/length_probe_dev.json"

  python -m guardlens.data.preflight_marker create \
    --variant "$variant" \
    --train "$train_path" \
    --dev "$dev_path" \
    --code-sha "$CODE_SHA" \
    --input-view "$INPUT_VIEW" \
    --freeze-report "$output_dir/freeze_verification.json" \
    --representation-report "$output_dir/representation_coverage.json" \
    --length-report "$output_dir/length_probe_dev.json" \
    --length-auc-ceiling "$LENGTH_AUC_CEILING" \
    --output "$output_dir/marker.json"
done

DEV_PATH="$FREEZE_DIR/splits_primary/dev.jsonl"
SOURCE_REVISION=$(sha256sum "$DEV_PATH" | awk '{print $1}')
mkdir -p "$(dirname "$PREPARED_DEV")"
python -m eval_platform prepare \
  --source internal \
  --source-revision "$SOURCE_REVISION" \
  --split dev \
  --input "$DEV_PATH" \
  --output "$PREPARED_DEV"

echo "Shared preflight complete for both training variants"
echo "Length-only AUC ceiling: $LENGTH_AUC_CEILING"
echo "Held-out test accessed: NO"
