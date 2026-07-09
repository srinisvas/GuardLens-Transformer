#!/usr/bin/env python3
"""
Bootstrap confidence intervals for GuardLens paper metrics.

Computes 95% bootstrap CIs for:
  - DD@15 (GuardLens and surface-risk)
  - Flip@15 (GuardLens and surface-risk)
  - Boundary FPR
  - Attribution Utility = DD@15 - Boundary FPR
  - Human top-5 turn hit rate (both annotators)

Usage:
    python compute_bootstrap_cis.py \
        --causal results/causal_eval_results.json \
        --boundary results/boundary_stress.json \
        --human results/human_benchmark_eval.json \
        --output results/bootstrap_cis.json \
        --n-bootstrap 10000 \
        --seed 42

All CIs are two-sided 95% percentile bootstrap intervals.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
#  Core bootstrap engine
# ═══════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    point_est = float(stat_fn(values))

    boots = np.array([
        stat_fn(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci
    ci_low  = float(np.percentile(boots, 100 * alpha / 2))
    ci_high = float(np.percentile(boots, 100 * (1 - alpha / 2)))

    return {
        "mean":    point_est,
        "ci_low":  ci_low,
        "ci_high": ci_high,
        "se":      float(np.std(boots)),
        "n":       n,
    }


def bootstrap_paired_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    stat_fn_a,
    stat_fn_b,
    combine_fn,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    na, nb = len(values_a), len(values_b)
    point_est = float(combine_fn(stat_fn_a(values_a), stat_fn_b(values_b)))

    boots = np.array([
        combine_fn(
            stat_fn_a(rng.choice(values_a, size=na, replace=True)),
            stat_fn_b(rng.choice(values_b, size=nb, replace=True)),
        )
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci
    ci_low  = float(np.percentile(boots, 100 * alpha / 2))
    ci_high = float(np.percentile(boots, 100 * (1 - alpha / 2)))

    return {
        "mean":    point_est,
        "ci_low":  ci_low,
        "ci_high": ci_high,
        "se":      float(np.std(boots)),
        "na":      na,
        "nb":      nb,
    }


# ═══════════════════════════════════════════════════════════════════
#  DD@15 extraction
# ═══════════════════════════════════════════════════════════════════

def extract_dd15_per_record(causal_json: dict, method: str) -> np.ndarray:
    """
    Extract per-record deviation drop at 15% for a given method.

    The causal eval JSON has structure:
      Top-level keys are method names: "guardlens", "surface_risk", etc.
      Each method has "per_record_dd" which is a dict keyed by k-level:
        {"5%": [0.1, 0.2, ...], "10%": [...], "15%": [...], "20%": [...]}

    Also handles full_test wrapper if present.
    """
    # Locate the method block (may be at top level or under full_test)
    method_data = None
    for root in [causal_json, causal_json.get("full_test", {})]:
        if method in root:
            method_data = root[method]
            break

    if method_data is None:
        raise KeyError(
            f"Method '{method}' not found. "
            f"Available keys: {list(causal_json.keys())}"
        )

    # per_record_dd is a dict keyed by k-level: {"5%": [...], "15%": [...]}
    if "per_record_dd" in method_data:
        prd = method_data["per_record_dd"]
        if isinstance(prd, dict):
            # Normal case: pick the 15% key
            arr = prd.get("15%", prd.get("0.15", []))
            if arr:
                return np.array(arr, dtype=float)
            raise KeyError(
                f"per_record_dd for '{method}' has no '15%' key. "
                f"Available: {list(prd.keys())}"
            )
        elif isinstance(prd, list):
            # Legacy: flat list assumed to be 15%
            return np.array(prd, dtype=float)

    # Fallback: per_record list of dicts
    if "per_record" in method_data:
        records = method_data["per_record"]
        if isinstance(records, list) and records:
            return np.array([
                r.get("deviation_drop_15", r.get("dd_15", r.get("absolute_drop", 0.0)))
                for r in records
            ], dtype=float)

    raise KeyError(
        f"Cannot find per-record DD@15 for method '{method}'. "
        f"Method keys: {list(method_data.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════
#  Flip@15 extraction
# ═══════════════════════════════════════════════════════════════════

def extract_flip15_per_record(causal_json: dict, method: str) -> np.ndarray:
    """
    Extract per-record flip indicators at 15% for a given method.

    Structure: method_data["flip_rates"]["per_record_flips"]["flip@15%"] → [0,1,0,1,...]
    """
    method_data = None
    for root in [causal_json, causal_json.get("full_test", {})]:
        if method in root:
            method_data = root[method]
            break

    if method_data is None:
        raise KeyError(f"Method '{method}' not found.")

    fr = method_data.get("flip_rates", {})
    prf = fr.get("per_record_flips", {})
    if isinstance(prf, dict):
        arr = prf.get("flip@15%", [])
        if arr:
            return np.array(arr, dtype=float)

    raise KeyError(
        f"Cannot find per-record Flip@15 for method '{method}'. "
        f"flip_rates keys: {list(fr.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════
#  Boundary FPR extraction
# ═══════════════════════════════════════════════════════════════════

def extract_boundary_fpr_per_record(boundary_json: dict) -> np.ndarray:
    """
    Extract per-record FP indicators from boundary stress JSON.

    Structure: boundary_json["per_record_fp"] → [0, 0, 1, 0, ...]
    """
    if "per_record_fp" in boundary_json:
        return np.array(boundary_json["per_record_fp"], dtype=float)

    # Fallback: reconstruct from per_record_prob and threshold
    if "per_record_prob" in boundary_json:
        threshold = boundary_json.get("threshold", 0.5)
        probs = np.array(boundary_json["per_record_prob"], dtype=float)
        return (probs > threshold).astype(float)

    raise KeyError(
        f"Cannot find per_record_fp in boundary stress JSON. "
        f"Available keys: {list(boundary_json.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════
#  Human top-5 hit extraction
# ═══════════════════════════════════════════════════════════════════

def extract_top5_hit_per_record(human_json: dict, annotator: str) -> np.ndarray:
    """
    Extract per-record top-5 turn hit (0/1) for a given annotator.

    Structure: human_json[annotator]["per_record_guardlens"] → [{"top5_hit": 1}, ...]
    """
    ann_data = human_json.get(annotator, {})
    records = ann_data.get("per_record_guardlens", [])
    if not records:
        raise KeyError(
            f"Cannot find per_record_guardlens for annotator '{annotator}'. "
            f"Available keys: {list(ann_data.keys())}"
        )
    return np.array([r.get("top5_hit", 0) for r in records], dtype=float)


# ═══════════════════════════════════════════════════════════════════
#  Formatting helpers
# ═══════════════════════════════════════════════════════════════════

def fmt(ci_dict: dict, decimals: int = 3) -> str:
    m  = round(ci_dict["mean"],    decimals)
    lo = round(ci_dict["ci_low"],  decimals)
    hi = round(ci_dict["ci_high"], decimals)
    n_key = "n" if "n" in ci_dict else "na"
    n  = ci_dict.get(n_key, "?")
    return f"{m:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]  (n={n})"


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for GuardLens paper metrics"
    )
    parser.add_argument("--causal",    required=True,
                        help="causal_eval_results.json")
    parser.add_argument("--boundary",  required=True,
                        help="boundary_stress.json")
    parser.add_argument("--human",     default=None,
                        help="human_benchmark_eval.json (optional)")
    parser.add_argument("--output",    required=True,
                        help="Output JSON path for CIs")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--ci",          type=float, default=0.95)
    args = parser.parse_args()

    P = sys.stderr
    B = args.n_bootstrap
    S = args.seed
    C = args.ci

    print(f"Bootstrap CIs ({int(C*100)}%, B={B}, seed={S})", file=P)

    with open(args.causal)   as f: causal_json   = json.load(f)
    with open(args.boundary) as f: boundary_json = json.load(f)

    human_json = None
    if args.human and os.path.exists(args.human):
        with open(args.human) as f: human_json = json.load(f)

    results = {}

    # ── 1. DD@15 ──────────────────────────────────────────────────
    print_section("DD@15")

    for method in ["guardlens", "surface_risk"]:
        try:
            vals = extract_dd15_per_record(causal_json, method)
            ci_result = bootstrap_ci(vals, np.mean, n_bootstrap=B, ci=C, seed=S)
            label = "GuardLens" if method == "guardlens" else "Surface-risk"
            print(f"  {label:<20} {fmt(ci_result)}")
            results[f"dd15_{method}"] = ci_result
        except KeyError as e:
            print(f"  SKIP DD@15 {method}: {e}", file=P)

    # ── 2. Flip@15 ────────────────────────────────────────────────
    print_section("Flip@15")

    for method in ["guardlens", "surface_risk"]:
        try:
            vals = extract_flip15_per_record(causal_json, method)
            ci_result = bootstrap_ci(vals, np.mean, n_bootstrap=B, ci=C, seed=S)
            label = "GuardLens" if method == "guardlens" else "Surface-risk"
            print(f"  {label:<20} {fmt(ci_result)}")
            results[f"flip15_{method}"] = ci_result
        except KeyError as e:
            print(f"  SKIP Flip@15 {method}: {e}", file=P)

    # ── 3. Boundary FPR ──────────────────────────────────────────
    print_section("Boundary FPR")

    try:
        vals = extract_boundary_fpr_per_record(boundary_json)
        ci_result = bootstrap_ci(vals, np.mean, n_bootstrap=B, ci=C, seed=S)
        print(f"  {'GuardLens':<20} {fmt(ci_result)}")
        results["boundary_fpr"] = ci_result
    except KeyError as e:
        print(f"  SKIP Boundary FPR: {e}", file=P)

    # ── 4. Utility = DD@15 - Boundary FPR ────────────────────────
    print_section("Attribution Utility = DD@15 − Boundary FPR")

    if "dd15_guardlens" in results and "boundary_fpr" in results:
        try:
            dd_vals  = extract_dd15_per_record(causal_json, "guardlens")
            fpr_vals = extract_boundary_fpr_per_record(boundary_json)
            ci_result = bootstrap_paired_ci(
                values_a=dd_vals,
                values_b=fpr_vals,
                stat_fn_a=np.mean,
                stat_fn_b=np.mean,
                combine_fn=lambda a, b: a - b,
                n_bootstrap=B, ci=C, seed=S,
            )
            print(f"  {'GuardLens':<20} {fmt(ci_result)}")
            results["utility_guardlens"] = ci_result
        except KeyError as e:
            # Fallback: delta of independent CIs
            dd  = results["dd15_guardlens"]
            fpr = results["boundary_fpr"]
            util_mean = dd["mean"] - fpr["mean"]
            util_se   = (dd["se"]**2 + fpr["se"]**2) ** 0.5
            z = 1.96
            ci_result = {
                "mean":    util_mean,
                "ci_low":  util_mean - z * util_se,
                "ci_high": util_mean + z * util_se,
                "se":      util_se,
                "note":    "delta of independent CIs (fallback)",
            }
            print(f"  {'GuardLens':<20} {fmt(ci_result)}  [delta fallback]")
            results["utility_guardlens"] = ci_result

    # ── 5. Human top-5 hit rate ───────────────────────────────────
    if human_json:
        print_section("Human top-5 turn hit rate")

        # Try common annotator name patterns
        annotator_names = []
        for key in human_json.keys():
            if key.startswith("Ann") or key.startswith("Annotator"):
                annotator_names.append(key)

        if not annotator_names:
            print(f"  No annotator keys found. Keys: {list(human_json.keys())}", file=P)

        for annotator in annotator_names:
            try:
                vals = extract_top5_hit_per_record(human_json, annotator)
                ci_result = bootstrap_ci(vals, np.mean, n_bootstrap=B, ci=C, seed=S)
                print(f"  {annotator:<20} {fmt(ci_result)}")
                results[f"human_top5_{annotator}"] = ci_result
            except KeyError as e:
                print(f"  SKIP {annotator}: {e}", file=P)

    # ── Summary table ─────────────────────────────────────────────
    print_section("PAPER-READY SUMMARY")
    print()
    print(f"  {'Metric':<30} {'Point est':>10} {'95% CI':>24} {'n':>6}")
    print(f"  {'-'*30} {'-'*10} {'-'*24} {'-'*6}")

    for label, key in [
        ("DD@15 (GuardLens)",     "dd15_guardlens"),
        ("DD@15 (Surface-risk)",  "dd15_surface_risk"),
        ("Flip@15 (GuardLens)",   "flip15_guardlens"),
        ("Flip@15 (Surface-risk)","flip15_surface_risk"),
        ("Boundary FPR",          "boundary_fpr"),
        ("Utility (GuardLens)",   "utility_guardlens"),
    ]:
        ci_r = results.get(key)
        if ci_r:
            n = ci_r.get("n", ci_r.get("na", "?"))
            print(f"  {label:<30} {ci_r['mean']:>10.4f} "
                  f"[{ci_r['ci_low']:.4f}, {ci_r['ci_high']:.4f}] {n:>6}")

    # Human annotators
    for key, val in results.items():
        if key.startswith("human_top5_"):
            ann = key.replace("human_top5_", "")
            print(f"  {'Top-5 hit (' + ann + ')':<30} {val['mean']:>10.4f} "
                  f"[{val['ci_low']:.4f}, {val['ci_high']:.4f}] {val['n']:>6}")

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved → {args.output}", file=P)


if __name__ == "__main__":
    main()