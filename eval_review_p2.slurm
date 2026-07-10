#!/bin/bash
#SBATCH --job-name=gl-review-p2
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/review_p2_%j.out
#SBATCH --error=logs/review_p2_%j.err

# ============================================================
# eval_review_p2.slurm — Phase 2 reviewer response experiments
#
# Requires: checkpoints from train_and_eval_review.slurm
#
# Steps:
#   1. Target-LLM counterfactual test  (~4h, needs Llama-3-8B)
#   2. LLM-as-judge attribution baseline (~2h)
#   3. Bootstrap CIs on existing results (~5min, CPU)
#   4. Decoy/false-lead specificity (~30min)
#
# Prerequisites:
#   - eval_target_llm_causal.py, eval_llm_judge_baseline.py
#     copied into guardlens/evaluation/
#   - Llama-3-8B-Instruct weights in HF cache
#     (download on login node first if needed)
#   - Trained checkpoints from phase 1
# ============================================================

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"

SPLIT_DIR="$HOME/staging/dataset_gen_output/splits"
TEST_PATH="${SPLIT_DIR}/test.jsonl"

CKPT_DIR="$HOME/work/results/guardlens_v11/checkpoints"
OUT_DIR="$HOME/work/results/guardlens_v11/results/review_response"

# ---- Environment ----
set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
mkdir -p "$TMPDIR" "$OUT_DIR" logs

cd "$PROJECT_DIR"

echo "========================================================"
echo "  Reviewer Response Phase 2"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  Python: $(which python) ($(python --version 2>&1))"

# Verify checkpoint
GL_ATTR=$(ls "$CKPT_DIR/guardlens/best_attribution.pt" 2>/dev/null \
       || ls "$CKPT_DIR/guardlens/best.pt" 2>/dev/null \
       || echo "")

if [ -z "$GL_ATTR" ]; then
    echo "FATAL: No GuardLens checkpoint. Run train_and_eval_review.slurm first."
    exit 1
fi
echo "  Checkpoint: $GL_ATTR"

# ---- Model availability ----
# Target MUST be Llama-3-8B-Instruct (the actual target from dataset
# generation). If not cached, skip the target-LLM test entirely.
# Using a different model would be a surrogate test, not a target test.
TARGET_MODEL="/gpfs/home/s001/ssubram7/work/hf_models/llama3-8b-instruct"
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
RUN_TARGET=1

echo ""
echo "  Checking HF cache..."

# Check target model
if python -c "
import os; os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('$TARGET_MODEL')
    print('cached')
except: print('missing')
" 2>/dev/null | grep -q "cached"; then
    echo "  Target cached: $TARGET_MODEL"
else
    echo "  Target NOT cached: $TARGET_MODEL"
    echo "  Attempting download..."
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
AutoTokenizer.from_pretrained('$TARGET_MODEL')
AutoModelForCausalLM.from_pretrained('$TARGET_MODEL', torch_dtype=torch.float16)
print('  Downloaded')
" 2>&1 || {
        echo "  Download failed. SKIPPING target-LLM test."
        echo "  To fix: download Llama-3-8B-Instruct on login node."
        RUN_TARGET=0
    }
fi

# Check judge model (independent from target)
if python -c "
import os; os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('$JUDGE_MODEL')
    print('cached')
except: print('missing')
" 2>/dev/null | grep -q "cached"; then
    echo "  Judge cached: $JUDGE_MODEL"
else
    echo "  Judge NOT cached: $JUDGE_MODEL"
    echo "  Attempting download..."
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
AutoTokenizer.from_pretrained('$JUDGE_MODEL')
AutoModelForCausalLM.from_pretrained('$JUDGE_MODEL', torch_dtype=torch.float16)
print('  Downloaded')
" 2>&1 || {
        echo "  Judge download failed. Will use target model as judge."
        JUDGE_MODEL="$TARGET_MODEL"
    }
fi

export HF_HUB_OFFLINE=1
echo "  Target: $TARGET_MODEL (run=$RUN_TARGET)"
echo "  Judge:  $JUDGE_MODEL"


# ============================================================
# 1. Target-LLM Counterfactual Test
# ============================================================

echo ""
echo "========================================================"
echo "  1. Target-LLM Counterfactual Test"
echo "  Target: $TARGET_MODEL"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

if [ "$RUN_TARGET" -eq 1 ]; then
    python -m guardlens.evaluation.eval_target_llm_causal \
        --test-path "$TEST_PATH" \
        --checkpoint "$GL_ATTR" \
        --target-model "$TARGET_MODEL" \
        --judge-model "$JUDGE_MODEL" \
        --output "$OUT_DIR/target_llm_causal.json" \
        --max-conversations 87 \
        --batch-size 4 \
        --device cuda \
    && echo "  1: SUCCESS ($(date +%H:%M:%S))" \
    || echo "  1: FAILED ($(date +%H:%M:%S))"
else
    echo "  1: SKIPPED (Llama-3-8B-Instruct not available)"
fi


# ============================================================
# 2. LLM-as-Judge Attribution Baseline
# ============================================================

echo ""
echo "========================================================"
echo "  2. LLM-as-Judge Attribution Baseline"
echo "  Judge: $JUDGE_MODEL"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python -m guardlens.evaluation.eval_llm_judge_baseline \
    --test-path "$TEST_PATH" \
    --checkpoint "$GL_ATTR" \
    --judge-model "$JUDGE_MODEL" \
    --output "$OUT_DIR/llm_judge_baseline.json" \
&& echo "  2: SUCCESS ($(date +%H:%M:%S))" \
|| echo "  2: FAILED ($(date +%H:%M:%S))"


# ============================================================
# 3. Bootstrap CIs on Existing Results (CPU only)
# ============================================================

echo ""
echo "========================================================"
echo "  3. Bootstrap CIs"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python3 << 'BOOTSTRAP_EOF'
import json
import numpy as np
import os

OUT_DIR = os.environ.get("HOME", "") + "/work/results/guardlens_v11/results/review_response"

# ---- Top-k turn hit rate CIs ----
topk_path = os.path.join(OUT_DIR, "topk_turn_hitrate.json")
if os.path.exists(topk_path):
    data = json.load(open(topk_path))
    results = data.get("results", {})

    print("  Top-k Turn Hit Rate Bootstrap CIs:")
    ci_results = {}
    rng = np.random.RandomState(42)

    for method in ["guardlens", "surface_risk", "random"]:
        m = results.get(method, {})
        ci_results[method] = {}
        for k in [1, 2, 3, 5]:
            d = m.get(f"top_{k}", {})
            hits = d.get("per_record_hits", [])
            floors = d.get("per_record_floors", [])
            if not hits:
                continue

            # Per-record margin: hit_i - floor_i
            if floors and len(floors) == len(hits):
                margins = np.array(hits, dtype=float) - np.array(floors, dtype=float)
            else:
                # Fallback to aggregate floor
                floor = d.get("random_floor", 0)
                margins = np.array(hits, dtype=float) - floor

            boot = []
            for _ in range(5000):
                sample = rng.choice(margins, size=len(margins), replace=True)
                boot.append(np.mean(sample))
            ci_low = np.percentile(boot, 2.5)
            ci_high = np.percentile(boot, 97.5)
            margin = np.mean(margins)

            ci_results[method][f"top_{k}"] = {
                "margin": float(margin),
                "ci_95": [float(ci_low), float(ci_high)],
            }
            print(f"    {method} top-{k}: margin={margin:+.4f} "
                  f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}]")

    ci_path = os.path.join(OUT_DIR, "topk_bootstrap_cis.json")
    with open(ci_path, "w") as f:
        json.dump(ci_results, f, indent=2)
    print(f"  Saved to {ci_path}")
else:
    print("  topk_turn_hitrate.json not found, skipping")

# ---- NoCF decomposition CIs ----
nocf_path = os.path.join(OUT_DIR, "nocf_decomposition.json")
if os.path.exists(nocf_path):
    data = json.load(open(nocf_path))
    gl = data.get("guardlens", {})
    nc = data.get("nocf", {})

    print("\n  NoCF Decomposition Bootstrap CIs (DD delta at 15%):")
    nocf_cis = {}

    for metric in ["dd_all", "dd_non_sr"]:
        gl_vals = gl.get("15%", {}).get(f"per_record_{metric}", [])
        nc_vals = nc.get("15%", {}).get(f"per_record_{metric}", [])

        if not gl_vals or not nc_vals:
            # Use aggregate values if per-record not available
            print(f"    {metric}: per-record data not available, skipping CI")
            continue

        # Paired bootstrap on delta (GL - NoCF)
        # Align by truncating to min length
        n = min(len(gl_vals), len(nc_vals))
        gl_arr = np.array(gl_vals[:n])
        nc_arr = np.array(nc_vals[:n])
        deltas = gl_arr - nc_arr

        boot = []
        for _ in range(5000):
            idx = rng.choice(n, size=n, replace=True)
            boot.append(np.mean(deltas[idx]))
        ci_low = np.percentile(boot, 2.5)
        ci_high = np.percentile(boot, 97.5)
        mean_delta = np.mean(deltas)

        nocf_cis[metric] = {
            "mean_delta_gl_minus_nocf": float(mean_delta),
            "ci_95": [float(ci_low), float(ci_high)],
        }
        sig = "significant" if ci_low > 0 or ci_high < 0 else "not significant"
        print(f"    {metric}: GL-NoCF={mean_delta:+.4f} "
              f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}] ({sig})")

    ci_path = os.path.join(OUT_DIR, "nocf_bootstrap_cis.json")
    with open(ci_path, "w") as f:
        json.dump(nocf_cis, f, indent=2)
    print(f"  Saved to {ci_path}")
else:
    print("  nocf_decomposition.json not found, skipping")

print("\n  3: DONE")
BOOTSTRAP_EOF

echo "  3 end: $(date +%H:%M:%S)"


# ============================================================
# 4. Decoy / False-Lead Attribution Specificity
# ============================================================

echo ""
echo "========================================================"
echo "  4. Decoy/False-Lead Attribution Specificity"
echo "  Start: $(date +%H:%M:%S)"
echo "========================================================"

python3 << 'DECOY_EOF'
import json
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.getcwd())

OUT_DIR = os.environ.get("HOME", "") + "/work/results/guardlens_v11/results/review_response"
TEST_PATH = os.environ.get("HOME", "") + "/staging/dataset_gen_output/splits/test.jsonl"
CKPT_DIR = os.environ.get("HOME", "") + "/work/results/guardlens_v11/checkpoints"

from guardlens.evaluation.eval_utils import load_jsonl
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from guardlens.evaluation.causal_eval import (
    guardlens_attribution,
    surface_risk_attribution,
    random_attribution,
    attention_attribution,
)
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
gl_ckpt_path = (
    f"{CKPT_DIR}/guardlens/best_attribution.pt"
    if os.path.exists(f"{CKPT_DIR}/guardlens/best_attribution.pt")
    else f"{CKPT_DIR}/guardlens/best.pt"
)
ckpt = torch.load(gl_ckpt_path, weights_only=False, map_location=device)
config = ckpt["config"]
model = MODEL_REGISTRY.get(ckpt.get("model_name", "guardlens"),
                           MODEL_REGISTRY["guardlens"])(config)
model.setup_backbone()
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device).eval()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
collator = GuardLensCollator(tokenizer, config)

records = load_jsonl(TEST_PATH)
print(f"  Records: {len(records)}")

# Identify false-lead and high-risk benign records
fl_records = [r for r in records if r.get("family") == "false_lead_benign"]
hr_records = [r for r in records if r.get("family") in ("hard_benign", "research_technical")]
print(f"  False-lead benign: {len(fl_records)}")
print(f"  Hard/research benign: {len(hr_records)}")

# For false-lead records: find which turns contain decoy vocabulary
# A turn with high surface-risk score in a benign conversation is a decoy
def get_decoy_turns(record):
    """Identify turns with high-risk vocabulary in benign records."""
    decoy_turns = set()
    for i, turn in enumerate(record.get("turns", [])):
        if turn.get("role") != "user":
            continue
        text = turn.get("text", "").lower()
        risk_terms = ["hack", "exploit", "vulnerability", "attack", "bypass",
                      "injection", "malware", "phishing", "payload", "backdoor",
                      "jailbreak", "weapon", "bomb", "poison", "kill", "drug"]
        if any(term in text for term in risk_terms):
            decoy_turns.add(i)
    return decoy_turns

# Process each subset
for subset_name, subset_records in [
    ("false_lead_benign", fl_records),
    ("hard_research_benign", hr_records),
]:
    if len(subset_records) < 3:
        print(f"\n  Skipping {subset_name} (only {len(subset_records)} records)")
        continue

    dataset = GuardLensDataset(subset_records, config)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator, num_workers=4)

    method_results = {m: {"top1_decoy": 0, "top3_decoy": 0, "total": 0}
                      for m in ["guardlens", "surface_risk", "attention", "random"]}

    for batch in loader:
        gl_scores = guardlens_attribution(model, batch, device)
        sr_scores = surface_risk_attribution(model, batch, device, tokenizer=tokenizer)
        rand_scores = random_attribution(model, batch, device)
        attn_scores = attention_attribution(model, batch, device)

        for i, meta in enumerate(batch["metadata"]):
            cid = meta.get("conversation_id", "")
            record = None
            for r in subset_records:
                if r.get("conversation_id") == cid:
                    record = r
                    break
            if record is None:
                continue

            decoy_turns = get_decoy_turns(record)
            if not decoy_turns:
                continue

            turns = record.get("turns", [])
            turn_mask = batch["turn_mask"][i]
            attn_mask = batch["attention_mask"][i]

            for method_name, scores in [
                ("guardlens", gl_scores),
                ("surface_risk", sr_scores),
                ("attention", attn_scores),
                ("random", rand_scores),
            ]:
                # Get per-user-turn mean attribution
                user_turn_scores = []
                for t in range(min(len(turns), turn_mask.size(0))):
                    if turn_mask[t] == 0 or turns[t].get("role") != "user":
                        continue
                    valid = attn_mask[t].bool()
                    s = scores[i][t][valid].mean().item() if valid.any() else 0
                    user_turn_scores.append((t, s))

                if not user_turn_scores:
                    continue

                ranked = sorted(user_turn_scores, key=lambda x: -x[1])

                # Top-1 on decoy?
                top1_turn = ranked[0][0]
                if top1_turn in decoy_turns:
                    method_results[method_name]["top1_decoy"] += 1

                # Top-3 on decoy?
                top3_turns = {t[0] for t in ranked[:3]}
                if top3_turns & decoy_turns:
                    method_results[method_name]["top3_decoy"] += 1

                method_results[method_name]["total"] += 1

    print(f"\n  {subset_name.upper()} — Decoy Attribution Rate:")
    print(f"  {'Method':<20} {'Top-1 Decoy':>12} {'Top-3 Decoy':>12} {'n':>6}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*6}")
    for method in ["guardlens", "surface_risk", "attention", "random"]:
        r = method_results[method]
        n = max(1, r["total"])
        t1 = r["top1_decoy"] / n
        t3 = r["top3_decoy"] / n
        print(f"  {method:<20} {t1:>12.3f} {t3:>12.3f} {r['total']:>6}")

    # Save
    decoy_path = os.path.join(OUT_DIR, f"decoy_specificity_{subset_name}.json")
    with open(decoy_path, "w") as f:
        json.dump(method_results, f, indent=2)
    print(f"  Saved to {decoy_path}")

print("\n  4: DONE")
DECOY_EOF

echo "  4 end: $(date +%H:%M:%S)"


# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================================"
echo "  Phase 2 Summary"
echo "  $(date)"
echo "========================================================"

echo ""
echo "  Output files:"
for f in target_llm_causal.json llm_judge_baseline.json \
         topk_bootstrap_cis.json nocf_bootstrap_cis.json \
         decoy_specificity_false_lead_benign.json; do
    if [ -f "$OUT_DIR/$f" ]; then
        echo "    OK: $f"
    else
        echo "    MISSING: $f"
    fi
done

# Target-LLM summary
if [ -f "$OUT_DIR/target_llm_causal.json" ]; then
    echo ""
    echo "  Target-LLM Counterfactual:"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/target_llm_causal.json'))
m = r.get('metrics', {})
print(f'    Total processed: {m.get(\"n_total\", 0)}')
for method in ['guardlens', 'surface_risk', 'random']:
    d = m.get(method, {})
    ci = d.get('ci_95', [0, 0])
    print(f'    {method:<20} origUnsafe={d.get(\"original_unsafe_rate\",0):.3f} '
          f'flip={d.get(\"flip_rate\",0):.3f} '
          f'95% CI [{ci[0]:.3f}, {ci[1]:.3f}] '
          f'n={d.get(\"n_valid\",0)}')
for control in ['surface_risk', 'random']:
    p = m.get(f'paired_vs_{control}', {})
    ci = p.get('ci_95', [0, 0])
    print(f'    GL - {control:<15} delta={p.get(\"mean_delta\",0):+.3f} '
          f'95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] '
          f'n={p.get(\"n_paired\",0)}')
" 2>/dev/null || echo "    parse error"
fi

# LLM judge summary
if [ -f "$OUT_DIR/llm_judge_baseline.json" ]; then
    echo ""
    echo "  LLM-as-Judge Baseline:"
    python3 -c "
import json
r = json.load(open('$OUT_DIR/llm_judge_baseline.json'))
d = r.get('detection', {})
print(f'    Detection: F1={d.get(\"f1\",0):.3f} P={d.get(\"precision\",0):.3f} R={d.get(\"recall\",0):.3f}')
h = r.get('turn_hit_rates', {})
for k in [1, 2, 3, 5]:
    t = h.get(f'top_{k}', {})
    print(f'    top-{k}: hit={t.get(\"hit_rate\",0):.3f} margin={t.get(\"margin\",0):+.3f}')
fl = r.get('false_lead_specificity', {})
print(f'    False-lead FP rate: {fl.get(\"adversarial_fp_rate\",0):.3f}')
print(f'    False-lead top1 decoy: {fl.get(\"top1_decoy_rate\",0):.3f}')
print(f'    False-lead top3 decoy: {fl.get(\"top3_decoy_rate\",0):.3f}')
" 2>/dev/null || echo "    parse error"
fi

echo ""
echo "========================================================"
echo "  Done: $(date)"
echo "========================================================"
