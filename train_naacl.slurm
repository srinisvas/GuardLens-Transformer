#!/bin/bash
#SBATCH --job-name=guardlens-naacl
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=logs/train_naacl_%j.out
#SBATCH --error=logs/train_naacl_%j.err

# Retrain the unchanged GuardLens architecture and existing baselines on the
# validity-repaired NAACL splits. Old v11 checkpoints are never touched.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
SPLIT_DIR="${SPLIT_DIR:-$HOME/staging/dataset_gen_output/naacl_splits}"
BASE_OUTPUT="${BASE_OUTPUT:-$HOME/work/results/guardlens_naacl/checkpoints}"
PRECHECK_DIR="${PRECHECK_DIR:-$HOME/work/results/guardlens_naacl/preflight}"
MAX_LENGTH_AUC="${MAX_LENGTH_AUC:-0.65}"

TRAIN_PATH="$SPLIT_DIR/train.jsonl"
DEV_PATH="$SPLIT_DIR/dev.jsonl"
TEST_PATH="$SPLIT_DIR/test.jsonl"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export TMPDIR="${TMPDIR:-$HOME/work/.tmp}"
export HF_HOME="${HF_HOME:-$HOME/work/hf_models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$TMPDIR" "$BASE_OUTPUT" "$PRECHECK_DIR" logs
cd "$PROJECT_DIR"

for path in "$TRAIN_PATH" "$DEV_PATH" "$TEST_PATH"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing split: $path"
        exit 2
    fi
done

echo "========================================================"
echo " GuardLens NAACL validity-repair retraining"
echo " Train:  $TRAIN_PATH"
echo " Dev:    $DEV_PATH"
echo " Test:   $TEST_PATH"
echo " Output: $BASE_OUTPUT"
echo "========================================================"

# Verify split linkage before training. This reads identifiers/labels only for
# structural sanity and does not tune anything on the held-out test split.
python3 - "$TRAIN_PATH" "$DEV_PATH" "$TEST_PATH" <<'PY'
import json, sys
paths = dict(zip(("train", "dev", "test"), sys.argv[1:]))
data = {}
for name, path in paths.items():
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    if not rows:
        raise SystemExit(f"ERROR: {name} split is empty")
    labels = {int(r.get("label", -1)) for r in rows}
    if not {0, 1}.issubset(labels):
        raise SystemExit(f"ERROR: {name} split does not contain both labels: {labels}")
    data[name] = rows

for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
    ids_a = {str(r.get("conversation_id", "")) for r in data[a]}
    ids_b = {str(r.get("conversation_id", "")) for r in data[b]}
    dup_ids = (ids_a & ids_b) - {""}
    if dup_ids:
        raise SystemExit(f"ERROR: conversation leakage {a}<->{b}: {len(dup_ids)} ids")
    pairs_a = {str(r.get("pair_id", "")) for r in data[a] if r.get("pair_id")}
    pairs_b = {str(r.get("pair_id", "")) for r in data[b] if r.get("pair_id")}
    dup_pairs = pairs_a & pairs_b
    if dup_pairs:
        raise SystemExit(f"ERROR: pair leakage {a}<->{b}: {len(dup_pairs)} pair_ids")

print("Split leakage/class preflight: PASS")
for name, rows in data.items():
    print(name, len(rows), {k: sum(int(r.get('label',-1)) == k for r in rows) for k in (0,1)})
PY

# Shortcut gate uses TRAIN + DEV ONLY. The held-out test split is deliberately
# not opened by this probe. Full test AUC is reported later by eval_naacl.slurm.
python -m guardlens.evaluation.eval_length_probe_preflight \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --output "$PRECHECK_DIR/length_probe_dev.json"

python3 - "$PRECHECK_DIR/length_probe_dev.json" "$MAX_LENGTH_AUC" <<'PY'
import json, sys
path, limit = sys.argv[1], float(sys.argv[2])
r = json.load(open(path, encoding="utf-8"))
auc = float(r["dev"]["roc_auc"])
bal = float(r["dev"]["balanced_accuracy"])
print(f"Length shortcut preflight: dev AUC={auc:.3f}, balanced_acc={bal:.3f}, gate={limit:.3f}")
if auc > limit:
    raise SystemExit(
        f"ERROR: dev length-only AUC {auc:.3f} exceeds {limit:.3f}. "
        "Do not train until the residual shortcut is understood."
    )
PY

MODELS=("guardlens" "guardlens_no_fusion" "guardlens_no_cf" "turn_level" "conversation_deberta")
EPOCHS=(25 20 20 20 20)
BATCH=(4 4 4 4 2)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    EP="${EPOCHS[$i]}"
    BS="${BATCH[$i]}"
    OUTPUT="$BASE_OUTPUT/$MODEL"
    mkdir -p "$OUTPUT"

    EXTRA_FLAGS=()
    if [[ "$MODEL" == "conversation_deberta" || "$MODEL" == "turn_level" ]]; then
        EXTRA_FLAGS+=("--no-pivot-head")
    fi

    echo ""
    echo "Training $MODEL at $(date)"
    python -m guardlens.train \
        --train-path "$TRAIN_PATH" \
        --dev-path "$DEV_PATH" \
        --test-path "$TEST_PATH" \
        --output "$OUTPUT" \
        --model "$MODEL" \
        --backbone microsoft/deberta-v3-base \
        --batch-size "$BS" \
        --lr 2e-4 \
        --epochs "$EP" \
        --max-turns 32 \
        --max-tokens 192 \
        --seed 42 \
        "${EXTRA_FLAGS[@]}"
done

echo "NAACL retraining complete: $BASE_OUTPUT"
