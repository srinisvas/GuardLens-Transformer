#!/bin/bash
#SBATCH --job-name=guardlens_collate
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/eval_collate_%j.out
#SBATCH --error=logs/eval_collate_%j.err

set -euo pipefail
mkdir -p logs

echo "========================================================"
echo "  GuardLens Eval Stage 5: Result Collation"
echo "  Job: $SLURM_JOB_ID"
echo "  Start: $(date)"
echo "========================================================"

source ~/.bashrc
conda activate guardlens 2>/dev/null || source ~/work/venv/guardlens/bin/activate

RESULTS_DIR=~/work/results/dataset_gen/results

python3 - <<'PYEOF'
"""
Collate all evaluation results into paper-ready tables.
Outputs a single markdown file with all key tables.
"""

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
    lines.append("")
    lines.append(f"## {title}")
    lines.append("")

def table_header(*cols):
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

def table_row(*cols):
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")

lines.append("# GuardLens Evaluation Results")
lines.append(f"")
lines.append("All results for paper submission.")
lines.append("")

# ============================================================
# TABLE 1: Main results (from causal_eval_results.json)
# ============================================================
section("Table 1: Main Causal Attribution Results (In-Distribution)")
main = load("../causal_eval_results.json")  # original results file
if main:
    table_header("Method", "Token F1", "DevDrop@5%", "DevDrop@10%", "DevDrop@15%",
                 "DevDrop@20%", "Flip@15%", "Nec@15%", "Suf@15%", "TrigSize (median)")
    for method, data in main.items():
        tf1 = data.get("token_f1", {}).get("f1", 0)
        drops = data.get("deviation_drops", {})
        flips = data.get("flip_rates", {})
        nec = data.get("necessity", {})
        suf = data.get("sufficiency", {})
        trig = data.get("trigger_size_stats", {}).get("median", 0)
        table_row(
            method,
            f"{tf1:.3f}",
            f"{drops.get('5%', 0):.3f}",
            f"{drops.get('10%', 0):.3f}",
            f"{drops.get('15%', 0):.3f}",
            f"{drops.get('20%', 0):.3f}",
            f"{flips.get('flip@15%', 0):.3f}",
            f"{nec.get('15%', 0):.3f}",
            f"{suf.get('15%', 0):.3f}",
            f"{trig}",
        )
else:
    lines.append("*causal_eval_results.json not found*")

# ============================================================
# TABLE 2: External Evaluator Comparison
# ============================================================
section("Table 2: Self-Eval vs External-Eval (Circularity Check)")
ext_full = load("external_eval_full.json")
if ext_full:
    table_header("Evaluator", "DevDrop@5%", "DevDrop@10%", "DevDrop@15%", "DevDrop@20%",
                 "Flip@15%", "Nec@15%", "Suf@15%")

    def ext_row(name, data, k_strs):
        row = [name]
        for ks in k_strs:
            row.append(f"{data.get(ks, {}).get('deviation_drop', 0):.3f}")
        row.append(f"{data.get('15%', {}).get('flip_rate', 0):.3f}")
        row.append(f"{data.get('15%', {}).get('necessity', 0):.3f}")
        row.append(f"{data.get('15%', {}).get('sufficiency', 0):.3f}")
        table_row(*row)

    k_strs = ["5%", "10%", "15%", "20%"]
    self_data = ext_full.get("self_eval", {})
    ext_row("GuardLens (self-eval)", self_data, k_strs)
    for name, data in ext_full.get("external_eval", {}).items():
        ext_row(name, data, k_strs)
else:
    lines.append("*external_eval_full.json not found*")

# ============================================================
# TABLE 3: Implicit vs Explicit Subset
# ============================================================
section("Table 3: Implicit vs Explicit Trigger Subset Analysis (@ 15%)")
imp_exp = load("implicit_explicit_eval.json")
if imp_exp:
    results = imp_exp.get("results", {})
    table_header("Method", "Implicit DD", "Implicit Flip", "Explicit DD", "Explicit Flip",
                 "HardNeg DD", "HardNeg Flip")
    methods_order = ["guardlens", "surface_risk", "integrated_gradients",
                     "grad_x_input", "attention", "random"]
    for method in methods_order:
        imp_data = results.get("implicit", {}).get(method, {})
        exp_data = results.get("explicit", {}).get(method, {})
        hn_data  = results.get("hard_negative", {}).get(method, {})
        imp_dd   = imp_data.get("deviation_drops", {}).get("15%", "—")
        imp_flip = imp_data.get("flip_rates", {}).get("flip@15%", "—")
        exp_dd   = exp_data.get("deviation_drops", {}).get("15%", "—")
        exp_flip = exp_data.get("flip_rates", {}).get("flip@15%", "—")
        hn_dd    = hn_data.get("deviation_drops",  {}).get("15%", "—")
        hn_flip  = hn_data.get("flip_rates", {}).get("flip@15%", "—")
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else str(v)
        table_row(method, fmt(imp_dd), fmt(imp_flip), fmt(exp_dd),
                  fmt(exp_flip), fmt(hn_dd), fmt(hn_flip))
else:
    lines.append("*implicit_explicit_eval.json not found*")

# ============================================================
# TABLE 4: Cross-Dataset Generalisation
# ============================================================
section("Table 4: Cross-Dataset Generalisation (Zero-Shot Transfer)")
models_order = ["guardlens", "guardlens_no_cf", "conversation_deberta"]
table_header("Model", "F1", "Acc", "Precision", "Recall",
             "DevDrop@15%", "Flip@15%", "Suf@15%")
for model in models_order:
    data = load(f"cross_dataset_{model}.json")
    if data is None:
        table_row(model, *["—"] * 7)
        continue
    cls = data.get("classification", {})
    causal = data.get("causal", {}).get("15%", {})
    table_row(
        model,
        f"{cls.get('f1', 0):.3f}",
        f"{cls.get('accuracy', 0):.3f}",
        f"{cls.get('precision', 0):.3f}",
        f"{cls.get('recall', 0):.3f}",
        f"{causal.get('deviation_drop', 0):.3f}",
        f"{causal.get('flip_rate', 0):.3f}",
        f"{causal.get('sufficiency', 0):.3f}",
    )

# Per-source breakdown
lines.append("")
lines.append("**Per-source breakdown (GuardLens F1):**")
cross_gl = load("cross_dataset_guardlens.json")
if cross_gl:
    per_src = cross_gl.get("per_source", {})
    table_header("Source", "N", "F1", "Precision", "Recall")
    for src, stats in sorted(per_src.items(), key=lambda x: -x[1].get("f1", 0)):
        table_row(src, stats["n"], f"{stats['f1']:.3f}",
                  f"{stats['precision']:.3f}", f"{stats['recall']:.3f}")

# ============================================================
# TABLE 5: Paraphrase Robustness
# ============================================================
section("Table 5: Paraphrase Robustness (Attribution Semantic Stability)")
table_header("Subset", "N", "Turn Spearman ρ", "Token Spearman ρ",
             "Top-15% Stability", "Pivot Stability")

for subset_name, fname in [
    ("All adversarial", "paraphrase_eval_full.json"),
    ("Implicit triggers", "paraphrase_eval_implicit.json"),
    ("Explicit triggers", "paraphrase_eval_explicit.json"),
]:
    data = load(fname)
    if data is None:
        table_row(subset_name, *["—"] * 5)
        continue
    agg = data.get("aggregate", {})
    n = data.get("n_samples", 0)
    table_row(
        subset_name, n,
        f"{agg.get('turn_rho_mean', 0):.3f} ± {agg.get('turn_rho_std', 0):.3f}",
        f"{agg.get('token_rho_mean', 0):.3f} ± {agg.get('token_rho_std', 0):.3f}",
        f"{agg.get('topk_stability', 0):.3f}",
        f"{agg.get('pivot_stability', 0):.3f}",
    )

lines.append("")
lines.append("*Spearman ρ > 0.60 indicates attribution tracks semantic content, not surface wording.*")

# ============================================================
# TABLE 6: Hard Negative Attribution Precision
# ============================================================
section("Table 6: Hard Negative Attribution Precision (False Positive Rate)")
prec = load("attribution_precision.json")
if prec:
    hn = prec.get("hard_negative_attribution_precision", {})
    table_header("Subset", "N Conversations", "FPAR (↓ better)",
                 "Mean Attribution (↓)", "Max Attribution (↓)", "Entropy (↑)")
    for subset_name in ["adversarial", "hard_negative", "borderline",
                        "false_pos_trap", "genuine_benign"]:
        d = hn.get(subset_name)
        if not d:
            continue
        table_row(
            subset_name,
            d.get("n_conversations", 0),
            f"{d.get('fpar', 0):.4f}",
            f"{d.get('mean_attribution', 0):.4f}",
            f"{d.get('mean_max_attribution', 0):.4f}",
            f"{d.get('attribution_entropy', 0):.4f}",
        )
    lines.append("")
    lines.append("*FPAR = fraction of tokens with attribution > 0.5. "
                 "Low FPAR on benign subsets = model does not hallucinate causes.*")
else:
    lines.append("*attribution_precision.json not found*")

# ============================================================
# TABLE 7: Minimality Sensitivity Curve
# ============================================================
section("Table 7: Minimality Sensitivity Curve (Flip Rate vs k%)")
if prec:
    min_data = prec.get("minimality_curve", {})
    curves = min_data.get("curves", {})
    sharpness = min_data.get("sharpness", {})
    k_fracs_used = min_data.get("k_fracs", [])
    k_strs_used = [f"{int(k*100)}%" for k in k_fracs_used]

    if curves:
        table_header("Method", *k_strs_used, "InflectionK", "Slope", "AUC")
        method_order = ["guardlens", "integrated_gradients", "grad_x_input",
                        "surface_risk", "random"]
        for method in method_order:
            if method not in curves:
                continue
            row_vals = [f"{curves[method].get(k, 0):.3f}" for k in k_strs_used]
            s = sharpness.get(method, {})
            table_row(
                method,
                *row_vals,
                f"{s.get('inflection_k', 0):.3f}",
                f"{s.get('slope_at_inflection', 0):.3f}",
                f"{s.get('auc', 0):.4f}",
            )
        lines.append("")
        lines.append("*Inflection K: k% at which flip rate first exceeds 50%. "
                     "Smaller = more compact trigger. "
                     "Slope: sharper inflection = near-minimal trigger set.*")
else:
    lines.append("*attribution_precision.json not found*")

# ============================================================
# TABLE 8: Cross-Model Transfer
# ============================================================
section("Table 8: Cross-Model Transfer (Attribution Transferability)")
for label, fname in [("Full Test Set", "cross_model_transfer_full.json"),
                     ("Implicit Triggers", "cross_model_transfer_implicit.json")]:
    xfer = load(fname)
    if not xfer:
        lines.append(f"*{fname} not found*")
        continue
    lines.append(f"**{label}** (k=15%, LLM={xfer.get('llm_provider','?')}/{xfer.get('llm_model','?')}):")
    lines.append("")
    table_header("Method", "Transfer Flip Rate (↑)", "N Flipped", "N Tested")
    tfr = xfer.get("transfer_flip_rates", {})
    for method in ("guardlens", "surface_risk", "random"):
        d = tfr.get(method, {})
        table_row(
            method,
            f"{d.get('transfer_flip_rate', 0):.3f}",
            d.get("n_flipped", 0),
            d.get("n_tested", 0),
        )
    gl = tfr.get("guardlens", {}).get("transfer_flip_rate", 0)
    rand = tfr.get("random", {}).get("transfer_flip_rate", 0)
    sr = tfr.get("surface_risk", {}).get("transfer_flip_rate", 0)
    lines.append("")
    lines.append(f"GuardLens vs Random: {gl - rand:+.3f} | "
                 f"GuardLens vs SurfaceRisk: {gl - sr:+.3f}")
    lines.append("")

lines.append("*Transfer Flip Rate: fraction of samples where removing GuardLens-attributed tokens "
             "changes external LLM response from compliant to refused. "
             "Demonstrates attribution is causally transferable across model boundaries.*")

# Write output
output_path = os.path.join(RES, "paper_tables.md")
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"Paper tables written to: {output_path}")
print(f"Total lines: {len(lines)}")
PYEOF

echo ""
echo "Paper tables:"
cat $RESULTS_DIR/paper_tables.md

echo ""
echo "========================================================"
echo "  Stage 5 complete: $(date)"
echo "  Final output: $RESULTS_DIR/paper_tables.md"
echo "========================================================"
