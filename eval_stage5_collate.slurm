#!/bin/bash
#SBATCH --job-name=gl-eval-collate
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_stage5_%j.out
#SBATCH --error=logs/eval_stage5_%j.err

# Stage 5: Collate all results into paper_tables.md
# Run LAST, after all other stages complete.
# Submit: sbatch eval_stage5_collate.slurm

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
RESULTS_DIR="$HOME/work/results/dataset_gen"
OUT_DIR="$RESULTS_DIR/results"

CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
mkdir -p "$TMPDIR"

cd "$PROJECT_DIR"

echo "========================================================"
echo "  GuardLens Eval Stage 5: Result Collation"
echo "  Job: ${SLURM_JOB_ID:-interactive}  Node: $(hostname)"
echo "  Start: $(date)"
echo "========================================================"

# Check which result files exist
echo "Checking result files..."
MISSING=0
for f in \
    causal_eval_results.json \
    implicit_explicit_eval.json \
    external_eval_full.json \
    cross_dataset_guardlens.json \
    paraphrase_eval_full.json \
    attribution_precision.json \
    cross_model_transfer_full.json; do
    if [ -f "$OUT_DIR/$f" ]; then
        echo "  OK:      $f"
    else
        echo "  MISSING: $f"
        MISSING=$((MISSING+1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "WARNING: $MISSING result file(s) missing. Tables will have gaps."
    echo "Continuing with available results..."
fi

python3 - <<'PYEOF'
import json, os

RES = os.path.expanduser("~/work/results/dataset_gen/results")

def load(fname):
    path = os.path.join(RES, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

lines = []

def section(title):
    lines.append(f"\n## {title}\n")

def table_header(*cols):
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

def table_row(*cols):
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")

def fmt(v, decimals=3):
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.{decimals}f}"
    return str(v)

lines.append("# GuardLens Evaluation Results\n")

# ============================================================
# TABLE 1: Main causal attribution results
# ============================================================
section("Table 1: Main Causal Attribution Results (In-Distribution)")
main = load("causal_eval_results.json")
if main:
    table_header("Method", "Token F1", "DevDrop@5%", "DevDrop@10%",
                 "DevDrop@15%", "DevDrop@20%", "Flip@15%",
                 "Nec@15%", "Suf@15%", "TrigSize")
    method_order = ["guardlens", "integrated_gradients", "grad_x_input",
                    "surface_risk", "attention", "random"]
    for method in method_order:
        if method not in main: continue
        d = main[method]
        tf1 = d.get("token_f1", {}).get("f1", 0)
        drops = d.get("deviation_drops", {})
        flips = d.get("flip_rates", {})
        nec   = d.get("necessity", {})
        suf   = d.get("sufficiency", {})
        trig  = d.get("trigger_size_stats", {}).get("median", 0)
        table_row(method, fmt(tf1),
                  fmt(drops.get("5%",0)), fmt(drops.get("10%",0)),
                  fmt(drops.get("15%",0)), fmt(drops.get("20%",0)),
                  fmt(flips.get("flip@15%",0)),
                  fmt(nec.get("15%",0)), fmt(suf.get("15%",0)),
                  str(int(trig)) if trig else "—")
else:
    lines.append("*causal_eval_results.json not found*")

# ============================================================
# TABLE 2: Self-eval vs External-eval
# ============================================================
section("Table 2: Self-Eval vs External-Eval (Circularity Check)")
ext = load("external_eval_full.json")
if ext:
    table_header("Evaluator", "DevDrop@5%", "DevDrop@10%",
                 "DevDrop@15%", "DevDrop@20%", "Flip@15%", "Nec@15%", "Suf@15%")
    k_strs = ["5%", "10%", "15%", "20%"]
    self_data = ext.get("self_eval", {})
    row = ["GuardLens (self-eval)"]
    for ks in k_strs:
        row.append(fmt(self_data.get(ks, {}).get("deviation_drop")))
    row.append(fmt(self_data.get("15%", {}).get("flip_rate")))
    row.append(fmt(self_data.get("15%", {}).get("necessity")))
    row.append(fmt(self_data.get("15%", {}).get("sufficiency")))
    table_row(*row)
    for name, data in ext.get("external_eval", {}).items():
        row = [name]
        for ks in k_strs:
            row.append(fmt(data.get(ks, {}).get("deviation_drop")))
        row.append(fmt(data.get("15%", {}).get("flip_rate")))
        row.append(fmt(data.get("15%", {}).get("necessity")))
        row.append(fmt(data.get("15%", {}).get("sufficiency")))
        table_row(*row)
else:
    lines.append("*external_eval_full.json not found*")

# ============================================================
# TABLE 3: Implicit vs Explicit Subset
# ============================================================
section("Table 3: Implicit vs Explicit Trigger Subset Analysis (@ 15%)")
imp_exp = load("implicit_explicit_eval.json")
if imp_exp:
    results = imp_exp.get("results", {})
    table_header("Method", "Implicit DD", "Implicit Flip",
                 "Explicit DD", "Explicit Flip", "HardNeg DD")
    method_order = ["guardlens", "surface_risk", "integrated_gradients",
                    "grad_x_input", "attention", "random"]
    for method in method_order:
        imp  = results.get("implicit", {}).get(method, {})
        expl = results.get("explicit", {}).get(method, {})
        hn   = results.get("hard_negative", {}).get(method, {})
        table_row(
            method,
            fmt(imp.get("deviation_drops",  {}).get("15%")),
            fmt(imp.get("flip_rates", {}).get("flip@15%")),
            fmt(expl.get("deviation_drops", {}).get("15%")),
            fmt(expl.get("flip_rates", {}).get("flip@15%")),
            fmt(hn.get("deviation_drops",   {}).get("15%")),
        )
    lines.append("\n*Key result: GuardLens - Surface Risk gap on Implicit triggers*")
else:
    lines.append("*implicit_explicit_eval.json not found*")

# ============================================================
# TABLE 4: Cross-Dataset Generalisation
# ============================================================
section("Table 4: Cross-Dataset Generalisation (Zero-Shot Transfer)")
table_header("Model", "F1", "Acc", "Precision", "Recall",
             "DevDrop@15%", "Flip@15%", "Suf@15%")
for model in ("guardlens", "guardlens_no_cf", "conversation_deberta"):
    data = load(f"cross_dataset_{model}.json")
    if not data:
        table_row(model, *["—"]*7); continue
    cls    = data.get("classification", {})
    causal = data.get("causal", {}).get("15%", {})
    table_row(model, fmt(cls.get("f1")), fmt(cls.get("accuracy")),
              fmt(cls.get("precision")), fmt(cls.get("recall")),
              fmt(causal.get("deviation_drop")),
              fmt(causal.get("flip_rate")),
              fmt(causal.get("sufficiency")))

# ============================================================
# TABLE 5: Paraphrase Robustness
# ============================================================
section("Table 5: Paraphrase Robustness (Semantic Attribution Stability)")
table_header("Subset", "N", "Turn Spearman ρ", "Token Spearman ρ",
             "Top-15% Stability", "Pivot Stability")
for label, fname in [
    ("All adversarial",   "paraphrase_eval_full.json"),
    ("Implicit triggers", "paraphrase_eval_implicit.json"),
    ("Explicit triggers", "paraphrase_eval_explicit.json"),
]:
    data = load(fname)
    if not data:
        table_row(label, *["—"]*5); continue
    agg = data.get("aggregate", {})
    n   = data.get("n_samples", 0)
    table_row(label, n,
              f"{agg.get('turn_rho_mean',0):.3f} ± {agg.get('turn_rho_std',0):.3f}",
              f"{agg.get('token_rho_mean',0):.3f} ± {agg.get('token_rho_std',0):.3f}",
              fmt(agg.get("topk_stability")),
              fmt(agg.get("pivot_stability")))
lines.append("\n*ρ > 0.60 = attribution tracks semantic content, not surface wording*")

# ============================================================
# TABLE 6: Hard Negative Attribution Precision
# ============================================================
section("Table 6: Hard Negative Attribution Precision (FPAR)")
prec = load("attribution_precision.json")
if prec:
    hn = prec.get("hard_negative_attribution_precision", {})
    table_header("Subset", "N", "FPAR (↓)", "Mean Attr (↓)", "Max Attr (↓)", "Entropy (↑)")
    for name in ["adversarial", "hard_negative", "borderline",
                 "false_pos_trap", "genuine_benign"]:
        d = hn.get(name)
        if not d: continue
        table_row(name, d.get("n_conversations","—"),
                  fmt(d.get("fpar")), fmt(d.get("mean_attribution")),
                  fmt(d.get("mean_max_attribution")), fmt(d.get("attribution_entropy")))
    lines.append("\n*FPAR = fraction of tokens with attribution > 0.5. Low on benign = no hallucinated causes.*")

    # Minimality curve
    lines.append("")
    section("Table 7: Minimality Sensitivity Curve (Flip Rate vs k%)")
    min_data = prec.get("minimality_curve", {})
    curves   = min_data.get("curves", {})
    sharpness = min_data.get("sharpness", {})
    k_fracs  = min_data.get("k_fracs", [])
    k_strs   = [f"{int(k*100)}%" for k in k_fracs]
    if curves:
        table_header("Method", *k_strs, "InflectionK", "Slope", "AUC")
        for method in ["guardlens", "integrated_gradients", "grad_x_input",
                       "surface_risk", "random"]:
            if method not in curves: continue
            vals = [fmt(curves[method].get(k, 0)) for k in k_strs]
            s    = sharpness.get(method, {})
            table_row(method, *vals,
                      fmt(s.get("inflection_k")),
                      fmt(s.get("slope_at_inflection")),
                      fmt(s.get("auc"), 4))
else:
    lines.append("*attribution_precision.json not found*")

# ============================================================
# TABLE 8: Cross-Model Transfer (ShieldGemma-9B)
# ============================================================
section("Table 8: Cross-Model Transfer (ShieldGemma-9B, Dangerous Content Policy)")
for label, fname in [
    ("Full test set",     "cross_model_transfer_full.json"),
    ("Implicit triggers", "cross_model_transfer_implicit.json"),
]:
    xfer = load(fname)
    if not xfer:
        lines.append(f"*{fname} not found*"); continue
    lines.append(f"\n**{label}** (k=15%, evaluator={xfer.get('shieldgemma_model','ShieldGemma-9B')}):\n")
    table_header("Method", "Transfer Flip Rate (↑)", "N Flipped", "N Tested")
    tfr = xfer.get("transfer_flip_rates", {})
    for method in ("guardlens", "surface_risk", "random"):
        d = tfr.get(method, {})
        table_row(method, fmt(d.get("transfer_flip_rate")),
                  d.get("n_flipped","—"), d.get("n_tested","—"))
    gl   = tfr.get("guardlens",     {}).get("transfer_flip_rate", 0)
    rand = tfr.get("random",        {}).get("transfer_flip_rate", 0)
    sr   = tfr.get("surface_risk",  {}).get("transfer_flip_rate", 0)
    lines.append(f"\nGuardLens vs Random: {gl-rand:+.3f} | GuardLens vs SurfaceRisk: {gl-sr:+.3f}")
lines.append("\n*Transfer Flip Rate: removing GuardLens-attributed tokens causes ShieldGemma to reclassify as safe.*")

# Write
output_path = os.path.join(RES, "paper_tables.md")
with open(output_path, "w") as f:
    f.write("\n".join(lines))
print(f"Written: {output_path}")
PYEOF

echo ""
echo "========================================================"
echo "  Stage 5 complete: $(date)"
echo "  Paper tables: $OUT_DIR/paper_tables.md"
echo "========================================================"
cat "$OUT_DIR/paper_tables.md"
