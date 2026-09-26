#!/bin/bash
#SBATCH --job-name=gl-top4-preflight
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preflight_naacl_top4_controls_%j.out
#SBATCH --error=logs/preflight_naacl_top4_controls_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
FREEZE_DIR="${FREEZE_DIR:-$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a}"
REPORT_PATH="${REPORT_PATH:-$FREEZE_DIR/data_prep_freeze_report.json}"
BACKBONE="${BACKBONE:-answerdotai/ModernBERT-large}"
BACKBONE_REVISION="${BACKBONE_REVISION:-45bb4654a4d5aaff24dd11d4781fa46d39bf8c13}"
MAX_TURNS="${MAX_TURNS:-64}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
LENGTH_AUC_CEILING="${LENGTH_AUC_CEILING:-0.650}"
: "${SHARED_PREFLIGHT_DIR:?fresh control preflight directory required}"
: "${PREPARED_DEV:?existing prepared internal dev JSONL required}"

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
[[ ! -e "$SHARED_PREFLIGHT_DIR" ]] || {
  echo "ERROR: control preflight exists: $SHARED_PREFLIGHT_DIR"
  exit 2
}
[[ -f "$PREPARED_DEV" && -f "$PREPARED_DEV.manifest.json" ]] || {
  echo "ERROR: existing prepared internal dev or its manifest is missing"
  exit 2
}

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

echo "Running complete CPU contract suite before the two missing GPU controls"
python -m unittest discover -v -p 'test*.py'

variant="primary_plus_auxiliary"
train_path="$FREEZE_DIR/splits_primary_plus_train_auxiliary/train.jsonl"
dev_path="$FREEZE_DIR/splits_primary_plus_train_auxiliary/dev.jsonl"
input_view="retrospective"
variant_dir="$SHARED_PREFLIGHT_DIR/$variant"
output_dir="$variant_dir/$input_view"
mkdir -p "$output_dir"

python -m guardlens.data.verify_freeze \
  --report "$REPORT_PATH" \
  --train "$train_path" \
  --dev "$dev_path" \
  --variant "$variant" \
  --output "$variant_dir/freeze_verification.json"

python -m guardlens.data.audit_representation \
  --train "$train_path" \
  --dev "$dev_path" \
  --backbone "$BACKBONE" \
  --backbone-revision "$BACKBONE_REVISION" \
  --max-turns "$MAX_TURNS" \
  --max-tokens "$MAX_TOKENS" \
  --input-view "$input_view" \
  --output "$output_dir/representation_coverage.json"

python -m guardlens.evaluation.eval_length_probe_preflight \
  --train "$train_path" \
  --dev "$dev_path" \
  --input-view "$input_view" \
  --output "$output_dir/length_probe_dev.json"

python -m guardlens.data.preflight_marker create \
  --variant "$variant" \
  --train "$train_path" \
  --dev "$dev_path" \
  --code-sha "$CODE_SHA" \
  --input-view "$input_view" \
  --backbone "$BACKBONE" \
  --backbone-revision "$BACKBONE_REVISION" \
  --max-turns "$MAX_TURNS" \
  --max-tokens "$MAX_TOKENS" \
  --freeze-report "$variant_dir/freeze_verification.json" \
  --representation-report "$output_dir/representation_coverage.json" \
  --length-report "$output_dir/length_probe_dev.json" \
  --length-auc-ceiling "$LENGTH_AUC_CEILING" \
  --output "$output_dir/marker.json"

python - "$PREPARED_DEV" <<'PY'
import json
import sys
from eval_platform.adapters import require_internal_dev_strata
from eval_platform.contract import file_hash, read_jsonl, validate_collection

path = sys.argv[1]
manifest = json.load(open(path + ".manifest.json", encoding="utf-8"))
if manifest.get("source") != "internal" or manifest.get("split") != "dev":
    raise SystemExit("ERROR: prepared input is not the internal development split")
if manifest.get("output_sha256") != file_hash(path):
    raise SystemExit("ERROR: prepared internal dev bytes differ from its manifest")
records = validate_collection(list(read_jsonl(path)))
counts = require_internal_dev_strata(records)
print("Reverified existing internal dev: " + json.dumps(counts, sort_keys=True))
PY

echo "Top-4 sibling control preflight complete"
echo "Prepared internal dev reused byte-for-byte: $PREPARED_DEV"
echo "Held-out test accessed: NO"
echo "External evaluation submitted: NO"
