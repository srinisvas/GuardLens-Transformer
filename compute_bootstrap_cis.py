#!/usr/bin/env python3
"""
Bootstrap confidence intervals for GuardLens paper metrics.

Computes 95% bootstrap CIs for:
  - DD@15 (GuardLens and surface-risk)
  - Boundary FPR (GuardLens and surface-risk)
  - SR-injected FPR (GuardLens and surface-risk)
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
    """
    Compute a bootstrap confidence interval for stat_fn(values).

    Returns dict with keys: mean, ci_low, ci_high, se, n
    """
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
    """
    Bootstrap CI for a statistic that combines two independent samples.
    E.g. Utility = DD@15(adversarial) - FPR(benign).

    values_a and values_b are bootstrapped independently since they come
    from different subsets (adversarial vs benign conversations).
    """
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
#  DD@15 from causal eval results
# ═══════════════════════════════════════════════════════════════════

def extract_dd15_per_record(causal_json: dict, method: str) -> np.ndarray:
    """
    Extract per-record deviation drop at 15% for a given method.

    The causal eval JSON stores per-record results under:
      causal_json["per_record"][method] → list of dicts with "deviation_drop_15"
    or
      causal_json["full_test"][method]["per_record_dd"] → list of floats

    Falls back to reconstructing from aggregate if per-record not available.
    """
    # Try per_record path first
    if "per_record" in causal_json:
        records = causal_json["per_record"].get(method, [])
        if records:
            return np.array([r.get("deviation_drop_15", r.get("dd_15", 0.0))
                             for r in records], dtype=float)

    # Try full_test per_record path
    full = causal_json.get("full_test", {})
    if method in full:
        method_data = full[method]
        if "per_record_dd" in method_data:
            return np.array(method_data["per_record_dd"], dtype=float)
        if "per_record" in method_data:
            records = method_data["per_record"]
            return np.array([r.get("dd_15", r.get("deviation_drop", 0.0))
                             for r in records], dtype=float)

    # Last resort: try top-level per_record_results
    if "per_record_results" in causal_json:
        records = causal_json["per_record_results"].get(method, [])
        if records:
            return np.array([r.get("dd_15", 0.0) for r in records], dtype=float)

    raise KeyError(
        f"Cannot find per-record DD@15 for method '{method}'. "
        f"Available keys: {list(causal_json.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════
#  FPR from boundary stress results
# ═══════════════════════════════════════════════════════════════════

def extract_fpr_per_record(boundary_json: dict, method: str, subset: str) -> np.ndarray:
    """
    Extract per-record binary FP indicators for a method on a benign subset.

    Returns array of 0/1 values (1 = false positive, 0 = true negative).

    Expected JSON structure:
      boundary_json["per_record"][subset][method] → list of {"pred": bool, "label": 0}
    or
      boundary_json[subset]["per_record"][method] → list of {"pred": bool}
    """
    # Path 1: per_record → subset → method
    if "per_record" in boundary_json:
        subset_data = boundary_json["per_record"].get(subset, {})
        records = subset_data.get(method, [])
        if records:
            return np.array([int(r.get("pred", r.get("predicted_adversarial", 0)))
                             for r in records], dtype=float)

    # Path 2: subset → per_record → method
    if subset in boundary_json:
        subset_block = boundary_json[subset]
        if "per_record" in subset_block:
            records = subset_block["per_record"].get(method, [])
            if records:
                return np.array([int(r.get("pred", 0)) for r in records], dtype=float)

    # Path 3: flat per_record list with method and subset fields
    if "records" in boundary_json:
        records = [
            r for r in boundary_json["records"]
            if r.get("method") == method and r.get("subset") == subset
        ]
        if records:
            return np.array([int(r.get("pred", 0)) for r in records], dtype=float)

    raise KeyError(
        f"Cannot find per-record FPR for method='{method}', subset='{subset}'. "
        f"Top-level keys: {list(boundary_json.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════
#  Human top-5 hit from human benchmark results
# ═══════════════════════════════════════════════════════════════════

def extract_top5_hit_per_record(human_json: dict, annotator: str) -> np.ndarray:
    """
    Extract per-record top-5 turn hit (0/1) for a given annotator.

    Expected structure:
      human_json[annotator]["per_record_guardlens"] → list of {"top5_hit": int}
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
    """Format as 'mean [ci_low, ci_high]' for printing."""
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
    parser.add_argument("--human",     required=True,
                        help="human_benchmark_eval.json")
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

    # Load JSON files
    with open(args.causal)   as f: causal_json   = json.load(f)
    with open(args.boundary) as f: boundary_json = json.load(f)
    with open(args.human)    as f: human_json    = json.load(f)

    results = {}

    # ── 1. DD@15 ────────────────────────────────────────────────────────────
    print_section("DD@15")

    for method in ["guardlens", "surface_risk"]:
        try:
            vals = extract_dd15_per_record(causal_json, method)
            ci   = bootstrap_ci(vals, np.mean, n_bootstrap=B, ci=C, seed=S)
            label = "Proposed" if method == "guardlens" else "Surface-risk"
            print(f"  {label:<20} {fmt(ci)}")
            results[f"dd15_{method}"] = ci
        except KeyError as e:
            print(f"  SKIP {method}: {e}", file=P)

    # ── 2. Boundary FPR ─────────────────────────────────────────────────────
    print_section("Boundary FPR (boundary_benign subset)")

    for method, label in [("guardlens", "Proposed"), ("surface_risk", "Surface-risk")]:
        for subset in ["boundary_benign", "all_boundary"]:
            try:
                vals = extract_fpr_per_record(boundary_json, method, subset)
                ci   = bootstrap_ci(vals, np.mean, n_bootstrap=B, ci=C, seed=S)
                print(f"  {label:<20} subset={subset:<20} {fmt(ci)}")
                results[f"boundary_fpr_{method}"] = ci
                break  # use first subset that works
            except KeyError:
                continue

    # ── 3. SR-injected FPR ──────────────────────────────────────────────────
    print_section("SR-injected FPR (high-risk vocab benign)")

    for method, label in [("guardlens", "Proposed"), ("surface_risk", "Surface-risk")]:
        for subset in ["sr_injected", "high_risk_benign", "sr_injected_benign"]:
            try:
                vals = extract_fpr_per_record(boundary_json, method, subset)
                ci   = bootstrap_ci(vals, np.mean, n_bootstrap=B, ci=C, seed=S)
                print(f"  {label:<20} subset={subset:<20} {fmt(ci)}")
                results[f"sr_injected_fpr_{method}"] = ci
                break
            except KeyError:
                continue

    # ── 4. Attribution Utility = DD@15 - Boundary FPR ───────────────────────
    print_section("Attribution Utility = DD@15 − Boundary FPR")

    for method, label in [("guardlens", "Proposed"), ("surface_risk", "Surface-risk")]:
        if f"dd15_{method}" in results and f"boundary_fpr_{method}" in results:
            # Need per-record arrays for both to bootstrap jointly
            try:
                dd_vals  = extract_dd15_per_record(causal_json, method)
                fpr_vals = extract_fpr_per_record(
                    boundary_json, method,
                    "boundary_benign" if method == "guardlens" else "boundary_benign"
                )
                ci = bootstrap_paired_ci(
                    values_a=dd_vals,
                    values_b=fpr_vals,
                    stat_fn_a=np.mean,
                    stat_fn_b=np.mean,
                    combine_fn=lambda a, b: a - b,
                    n_bootstrap=B, ci=C, seed=S,
                )
                print(f"  {label:<20} {fmt(ci)}")
                results[f"utility_{method}"] = ci
            except KeyError as e:
                # Fall back to delta of point estimates with pooled SE
                dd  = results[f"dd15_{method}"]
                fpr = results[f"boundary_fpr_{method}"]
                util_mean = dd["mean"] - fpr["mean"]
                util_se   = (dd["se"]**2 + fpr["se"]**2) ** 0.5
                z = 1.96  # 95%
                ci = {
                    "mean":    util_mean,
                    "ci_low":  util_mean - z * util_se,
                    "ci_high": util_mean + z * util_se,
                    "se":      util_se,
                    "note":    "delta of independent CIs (per-record FPR unavailable)",
                }
                print(f"  {label:<20} {fmt(ci)}  [delta fallback]")
                results[f"utility_{method}"] = ci

    # ── 5. Human top-5 hit rate ──────────────────────────────────────────────
    print_section("Human top-5 turn hit rate")

    for annotator, label in [("Ann_A", "Ann A"), ("Ann_B", "Ann B")]:
        try:
            vals = extract_top5_hit_per_record(human_json, annotator)
            ci   = bootstrap_ci(vals, np.mean, n_bootstrap=B, ci=C, seed=S)
            print(f"  {label:<20} {fmt(ci)}")
            results[f"human_top5_{annotator}"] = ci
        except KeyError as e:
            print(f"  SKIP {annotator}: {e}", file=P)

    # ── Summary table for paper ──────────────────────────────────────────────
    print_section("PAPER-READY SUMMARY")
    print()
    print(f"  {'Metric':<40} {'Proposed':<28} {'Surface-risk'}")
    print(f"  {'-'*40} {'-'*28} {'-'*20}")

    def row(metric, key_gl, key_sr, decimals=3):
        gl = results.get(key_gl)
        sr = results.get(key_sr)
        gl_str = f"{gl['mean']:.{decimals}f} [{gl['ci_low']:.{decimals}f}, {gl['ci_high']:.{decimals}f}]" if gl else "—"
        sr_str = f"{sr['mean']:.{decimals}f} [{sr['ci_low']:.{decimals}f}, {sr['ci_high']:.{decimals}f}]" if sr else "—"
        print(f"  {metric:<40} {gl_str:<28} {sr_str}")

    row("DD@15",              "dd15_guardlens",           "dd15_surface_risk")
    row("Boundary FPR",       "boundary_fpr_guardlens",   "boundary_fpr_surface_risk")
    row("SR-injected FPR",    "sr_injected_fpr_guardlens","sr_injected_fpr_surface_risk")
    row("Attribution Utility","utility_guardlens",         "utility_surface_risk")

    print()
    for ann in ["Ann_A", "Ann_B"]:
        ci = results.get(f"human_top5_{ann}")
        if ci:
            label = ann.replace("_", " ")
            print(f"  Human top-5 hit ({label:<6})              "
                  f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved → {args.output}", file=P)


if __name__ == "__main__":
    main()
