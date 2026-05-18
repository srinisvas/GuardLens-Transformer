#!/bin/bash
#SBATCH --job-name=gl-eval-setup
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_stage1_%j.out
#SBATCH --error=logs/eval_stage1_%j.err

# Stage 1: Download external evaluation datasets (AdvBench, HarmBench, JailbreakBench)
# Run this first. All other stages depend on it.
# Submit: sbatch eval_stage1_setup.slurm

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
RESULTS_DIR="$HOME/work/results/guardlens_v11"
OUT_DIR="$RESULTS_DIR/results"
EVAL_DATA="$RESULTS_DIR/eval_data"

CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
mkdir -p "$TMPDIR" "$OUT_DIR" "$EVAL_DATA"

cd "$PROJECT_DIR"

echo "========================================================"
echo "  GuardLens Eval Stage 1: Data Setup"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"

# ---- Verify checkpoints ----
echo "Verifying training checkpoints..."
for model in guardlens guardlens_no_fusion guardlens_no_cf turn_level conversation_deberta; do
    ckpt="$RESULTS_DIR/checkpoints/$model/best.pt"
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Missing checkpoint: $ckpt"
        exit 1
    fi
    echo "  OK: $model/best.pt"
done

# ---- Check for causal eval results (optional, generated in Stage 2) ----
if [ -f "$OUT_DIR/causal_eval_results.json" ]; then
    echo "  OK: causal_eval_results.json (pre-existing)"
else
    echo "  NOTE: causal_eval_results.json not found yet (will be generated in Stage 2)"
fi

# ---- Download datasets ----
echo ""
echo "Downloading AdvBench..."
mkdir -p "$EVAL_DATA/advbench"
curl -fsSL \
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv" \
    -o "$EVAL_DATA/advbench/harmful_behaviors.csv" \
    && echo "  AdvBench: OK" || echo "  AdvBench: FAILED"

echo "Downloading HarmBench..."
mkdir -p "$EVAL_DATA/harmbench"
curl -fsSL \
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv" \
    -o "$EVAL_DATA/harmbench/harmbench_behaviors_text_test.csv" \
    && echo "  HarmBench: OK" || echo "  HarmBench: FAILED (will continue without it)"

echo "Downloading JailbreakBench..."
mkdir -p "$EVAL_DATA/jailbreakbench"
curl -fsSL \
    "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/src/jailbreakbench/data/behaviors.csv" \
    -o "$EVAL_DATA/jailbreakbench/behaviors.csv" \
    && echo "  JailbreakBench: OK" || echo "  JailbreakBench: FAILED (will continue without it)"

# ---- Build unified seed file ----
echo ""
echo "Building cross_dataset_combined.jsonl..."
python3 - <<'PYEOF'
import csv, json, os

DATA_DIR = os.path.expanduser("~/work/results/dataset_gen/eval_data")
output_path = os.path.join(DATA_DIR, "cross_dataset_combined.jsonl")
records = []

# AdvBench
p = os.path.join(DATA_DIR, "advbench/harmful_behaviors.csv")
if os.path.exists(p):
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("goal", row.get("Goal", "")).strip()
            if len(text) >= 20:
                records.append({"text": text, "source": "advbench",
                                "attack_type": row.get("category", "unknown"),
                                "category": row.get("category", "unknown")})

# HarmBench
p = os.path.join(DATA_DIR, "harmbench/harmbench_behaviors_text_test.csv")
if os.path.exists(p):
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("Behavior") or row.get("behavior") or
                    row.get("goal") or row.get("Goal") or "").strip()
            if len(text) >= 20:
                records.append({"text": text, "source": "harmbench",
                                "attack_type": row.get("FunctionalCategory", "unknown"),
                                "category": row.get("SemanticCategory", "unknown")})

# JailbreakBench
p = os.path.join(DATA_DIR, "jailbreakbench/behaviors.csv")
if os.path.exists(p):
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("Goal") or row.get("goal") or
                    row.get("Behavior") or row.get("behavior") or "").strip()
            if len(text) >= 20:
                records.append({"text": text, "source": "jailbreakbench",
                                "attack_type": row.get("Category", "unknown"),
                                "category": row.get("Category", "unknown")})

# Deduplicate
seen = set()
deduped = []
for r in records:
    key = r["text"].lower()[:100]
    if key not in seen:
        seen.add(key)
        deduped.append(r)

with open(output_path, "w") as f:
    for r in deduped:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"  Combined: {len(deduped)} behaviors")
from collections import Counter
srcs = Counter(r["source"] for r in deduped)
for s, n in srcs.items():
    print(f"    {s}: {n}")
PYEOF

echo ""
echo "========================================================"
echo "  Stage 1 complete: $(date)"
echo "  Files in: $EVAL_DATA"
ls -la "$EVAL_DATA/"
echo "========================================================"
