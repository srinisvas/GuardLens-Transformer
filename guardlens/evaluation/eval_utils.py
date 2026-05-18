"""
guardlens/evaluation/eval_utils.py

Shared utilities for v11 evaluation scripts.
Handles pre-split loading, v11 subset partitioning, LaTeX output.
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple


# =========================================================
# Data loading — pre-split or fallback
# =========================================================

def load_test_data(
    test_path: str = "",
    data_path: str = "",
    seed: int = 42,
) -> Tuple[List[Dict], List[int]]:
    """
    Load test data from pre-split file (preferred) or fallback to
    re-splitting a single file.

    Returns:
        records: list of all records (for subset indexing)
        test_idx: list of indices into records for test set
    """
    if test_path and os.path.exists(test_path):
        records = load_jsonl(test_path)
        test_idx = list(range(len(records)))
        print(f"  Loaded {len(records)} test records from {test_path}")
        return records, test_idx

    if data_path and os.path.exists(data_path):
        records = load_jsonl(data_path)
        from guardlens.data.splits import pair_aware_split
        _, _, test_idx = pair_aware_split(records, seed=seed)
        print(f"  Loaded {len(records)} records, test set: {len(test_idx)}")
        return records, test_idx

    raise FileNotFoundError(f"No test data found at test_path={test_path} or data_path={data_path}")


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def add_test_path_args(parser):
    """Add --test-path and --data args to an argparse parser."""
    parser.add_argument("--test-path", type=str, default="",
                        help="Path to pre-split test.jsonl (preferred)")
    parser.add_argument("--data", type=str, default="",
                        help="Fallback: single JSONL file to re-split")
    return parser


# =========================================================
# v11 subset partitioning
# =========================================================

def partition_test_set_v11(records: List[Dict]) -> Dict[str, List[int]]:
    """
    Partition test records into subsets using v11 fields.

    Uses: pivot_kind, transfer_tier, benign_status, family.

    Subsets:
      - contextual_pivot: pivot_kind == contextual_pivot (implicit/contextual attacks)
      - lexical_pivot: pivot_kind == lexical_pivot (explicit keyword attacks)
      - distributed: transfer_tier in {transfer_success, target_only, cross_only}
                     AND pivot_kind in {none, distributed}
      - hard_benign: benign_status == validated_benign_twin OR family in hard_benign/false_lead
      - false_lead: family == false_lead_benign
      - clean_benign: benign_status == clean_benign
      - transfer_success: transfer_tier == transfer_success (cross-model jailbreaks)
      - target_only: transfer_tier == target_only
      - cross_only: transfer_tier == cross_only
    """
    subsets = {
        "contextual_pivot": [],
        "lexical_pivot": [],
        "distributed": [],
        "hard_benign": [],
        "false_lead": [],
        "clean_benign": [],
        "transfer_success": [],
        "target_only": [],
        "cross_only": [],
    }

    for idx, record in enumerate(records):
        label = record.get("label", 0)
        pivot_kind = record.get("pivot_kind", "none")
        transfer_tier = record.get("transfer_tier", "unknown")
        benign_status = record.get("benign_status", "none")
        family = record.get("family", "unknown")

        if label == 1:
            # Malicious subsets
            if pivot_kind == "contextual_pivot":
                subsets["contextual_pivot"].append(idx)
            elif pivot_kind == "lexical_pivot":
                subsets["lexical_pivot"].append(idx)

            if pivot_kind in ("none", "distributed", "distributed_or_unclear"):
                subsets["distributed"].append(idx)

            if transfer_tier == "transfer_success":
                subsets["transfer_success"].append(idx)
            elif transfer_tier == "target_only":
                subsets["target_only"].append(idx)
            elif transfer_tier == "cross_only":
                subsets["cross_only"].append(idx)

        elif label == 0:
            # Benign subsets
            if benign_status == "clean_benign":
                subsets["clean_benign"].append(idx)
            elif benign_status == "validated_benign_twin":
                subsets["hard_benign"].append(idx)

            if family == "false_lead_benign":
                subsets["false_lead"].append(idx)
            elif family in ("hard_benign", "topic_matched_safe"):
                if idx not in subsets["hard_benign"]:
                    subsets["hard_benign"].append(idx)

    return subsets


def partition_by_supervision_tier(records: List[Dict]) -> Dict[str, List[int]]:
    """Partition by supervision tier for tier-stratified evaluation."""
    subsets = {}
    for idx, record in enumerate(records):
        tier = record.get("supervision_tier", "unknown")
        subsets.setdefault(tier, []).append(idx)
    return subsets


def print_subset_summary(subsets: Dict[str, List[int]], records: List[Dict]):
    """Print summary of subset sizes and properties."""
    print(f"\n  Subset summary:")
    for name, indices in sorted(subsets.items(), key=lambda x: -len(x[1])):
        if not indices:
            continue
        labels = Counter(records[i].get("label", -1) for i in indices)
        print(f"    {name:<25} n={len(indices):4d}  mal={labels.get(1, 0):3d}  ben={labels.get(0, 0):3d}")


# =========================================================
# LaTeX table output
# =========================================================

def results_to_latex_table(
    results: Dict,
    caption: str = "Causal Attribution Evaluation",
    label: str = "tab:causal_eval",
    focus_k: str = "15%",
) -> str:
    """Convert causal eval results dict to a LaTeX table string."""
    methods = list(results.keys())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"Method & DD@{focus_k} & Flip@{focus_k} & Nec@{focus_k} & Suf@{focus_k} \\",
        r"\midrule",
    ]

    for method in methods:
        data = results[method]
        dd = data.get("deviation_drops", {}).get(focus_k, 0)
        flip = data.get("flip_rates", {}).get(f"flip@{focus_k}", 0)
        nec = data.get("necessity", {}).get(focus_k, 0)
        suf = data.get("sufficiency", {}).get(focus_k, 0)

        method_display = method.replace("_", " ").title()
        if method == "guardlens":
            method_display = r"\textbf{GuardLens}"
        elif method == "surface_risk":
            method_display = "Surface Risk"
        elif method == "integrated_gradients":
            method_display = "Int. Gradients"
        elif method == "grad_x_input":
            method_display = r"Grad$\times$Input"

        lines.append(
            rf"{method_display} & {dd:.3f} & {flip:.3f} & {nec:.3f} & {suf:.3f} \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def comparison_to_latex(
    all_results: Dict[str, Dict],
    caption: str = "Subset Comparison",
    label: str = "tab:subset_comparison",
    focus_k: str = "15%",
    methods: List[str] = None,
) -> str:
    """Convert multi-subset results to a LaTeX table."""
    if methods is None:
        methods = ["guardlens", "surface_risk", "integrated_gradients", "attention", "random"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{ll" + "c" * len(methods) + "}",
        r"\toprule",
    ]

    header = "Subset & Metric & " + " & ".join(
        m.replace("_", " ").title() for m in methods
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for subset_name, subset_results in all_results.items():
        subset_display = subset_name.replace("_", " ").title()
        for metric_name, metric_key in [("DD", "deviation_drops"), ("Flip", "flip_rates")]:
            vals = []
            for m in methods:
                data = subset_results.get(m, {})
                if metric_key == "flip_rates":
                    v = data.get(metric_key, {}).get(f"flip@{focus_k}", 0)
                else:
                    v = data.get(metric_key, {}).get(focus_k, 0)
                vals.append(f"{v:.3f}")
            lines.append(
                rf"{subset_display} & {metric_name}@{focus_k} & " + " & ".join(vals) + r" \\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)
