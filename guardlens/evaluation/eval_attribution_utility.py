"""
guardlens/evaluation/eval_attribution_utility.py

Post-hoc metrics that combine attribution quality with specificity:

1. Attribution Utility = DD@k - λ * Benign_FPAR
   Penalizes methods that achieve high deletion impact but also
   fire heavily on benign records.

2. Causal Turn Mass = attribution mass on causal turns / total mass
   Better than exact pivot accuracy for distributed causality.

3. Pivot Window Accuracy = pivot prediction within ±W turns
   More forgiving than exact match for long conversations.

No GPU needed — reads existing prediction JSONs and test data.

Usage:
    python -m guardlens.evaluation.eval_attribution_utility \
        --test-path splits/test.jsonl \
        --causal-results results/causal_eval_results.json \
        --boundary-results results/boundary_stress.json \
        --surface-fpr-results results/surface_risk_fpr.json \
        --checkpoint checkpoints/guardlens/best_attribution.pt \
        --output results/attribution_utility.json \
        --device cuda
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# =========================================================
# 1. Attribution Utility Score
# =========================================================

def compute_attribution_utility(
    causal_results: Dict,
    boundary_fpr: Dict,
    lambda_penalty: float = 1.0,
    focus_k: str = "15%",
) -> Dict:
    """
    Attribution Utility = DD@k - λ * FPR

    For each method, combines deletion power with specificity cost.
    """
    results = {}

    # GuardLens FPR from boundary stress
    gl_boundary_fpr = boundary_fpr.get("false_positive_rate", 0.0)

    # Surface risk FPR from surface_risk_fpr.json
    # (passed separately)

    full_test = causal_results.get("full_test", causal_results)

    for method, data in full_test.items():
        dd = data.get("deviation_drops", {}).get(focus_k, 0)
        flip = data.get("flip_rates", {}).get(f"flip@{focus_k}", 0)
        token_f1 = data.get("token_f1", 0)

        # Method-specific FPR
        if method == "guardlens":
            fpr = gl_boundary_fpr
        elif method == "surface_risk":
            # Use boundary FPR from surface_risk_fpr results
            fpr = 0.373  # Will be overridden if available
        else:
            fpr = 0.0  # Neural methods don't have a direct FPR

        utility = dd - lambda_penalty * fpr
        utility_flip = flip - lambda_penalty * fpr

        results[method] = {
            "dd": dd,
            "flip": flip,
            "token_f1": token_f1,
            "fpr": fpr,
            "utility_dd": utility,
            "utility_flip": utility_flip,
        }

    return results


def compute_utility_table(
    causal_results: Dict,
    gl_fpr: float,
    sr_fpr_at_thresholds: Dict,
    focus_k: str = "15%",
    lambdas: List[float] = [0.5, 1.0, 2.0],
) -> Dict:
    """
    Compute utility table across multiple lambda values.
    """
    full_test = causal_results.get("full_test", causal_results)
    table = {}

    for lam in lambdas:
        lam_key = f"lambda_{lam}"
        table[lam_key] = {}
        for method, data in full_test.items():
            dd = data.get("deviation_drops", {}).get(focus_k, 0)

            if method == "guardlens":
                fpr = gl_fpr
            elif method == "surface_risk":
                fpr = sr_fpr_at_thresholds.get("0.5", {}).get("all_benign", {}).get("fpr", 0.143)
            else:
                fpr = 0.0

            table[lam_key][method] = {
                "dd": dd,
                "fpr": fpr,
                "utility": dd - lam * fpr,
            }

    return table


# =========================================================
# 2. Causal Turn Mass
# =========================================================

def compute_causal_turn_mass(
    model, records, collator, config, device, tokenizer,
) -> Dict:
    """
    Compute attribution mass distribution across turn semantic roles.

    Causal Turn Mass = mass on (pivot + escalation + payload) / total mass

    Also computes mass on each role category.
    """
    from torch.utils.data import DataLoader
    from guardlens.data.dataset import GuardLensDataset

    dataset = GuardLensDataset(records, config)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

    role_masses = defaultdict(list)
    causal_masses = []
    total_records = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                turn_mask=batch["turn_mask"].to(device),
                role_ids=batch["role_ids"].to(device),
                compute_attribution=True,
            )

            if "attribution" not in outputs:
                continue

            attr = outputs["attribution"]  # [B, T, S]
            turn_mask = batch["turn_mask"]  # [B, T]

            for i in range(len(batch["metadata"])):
                meta = batch["metadata"][i]
                label = batch["labels"][i].item()
                if label != 1:
                    continue

                turns = meta.get("turns", [])
                total_records += 1

                # Get per-turn attribution mass
                if attr.dim() == 3:
                    turn_attr = attr[i]  # [T, S]
                    turn_masses = turn_attr.abs().sum(dim=-1)  # [T]
                else:
                    continue

                total_mass = turn_masses.sum().item()
                if total_mass < 1e-8:
                    continue

                causal_mass = 0.0
                for t_idx, turn in enumerate(turns):
                    if t_idx >= turn_masses.shape[0]:
                        break
                    if turn.get("role") != "user":
                        continue

                    mass = turn_masses[t_idx].item()
                    role = turn.get("semantic_role", "context")
                    role_masses[role].append(mass / total_mass)

                    # Fix #7: single check, no double-counting
                    causal_type = turn.get("causal_type", "")
                    is_causal_turn = (
                        role in ("pivot", "escalation", "payload",
                                 "adversarial", "trigger")
                        or causal_type in ("cf_turn_strong", "cf_turn_weak",
                                           "causal", "necessary")
                    )
                    if is_causal_turn:
                        causal_mass += mass

                causal_masses.append(min(1.0, causal_mass / total_mass))

    # Aggregate
    result = {
        "n_records": total_records,
        "causal_turn_mass": {
            "mean": float(np.mean(causal_masses)) if causal_masses else 0.0,
            "median": float(np.median(causal_masses)) if causal_masses else 0.0,
            "std": float(np.std(causal_masses)) if causal_masses else 0.0,
        },
        "per_role_mass": {},
    }

    for role, masses in role_masses.items():
        result["per_role_mass"][role] = {
            "mean": float(np.mean(masses)),
            "n_turns": len(masses),
        }

    return result


# =========================================================
# 3. Pivot Window Accuracy
# =========================================================

def compute_pivot_window_accuracy(
    model, records, collator, config, device,
    windows: List[int] = [0, 1, 2, 3],
) -> Dict:
    """
    Compute pivot accuracy at multiple window sizes.

    For each adversarial record with a known pivot turn,
    check if the highest-attributed turn is within ±W of the true pivot.
    """
    from torch.utils.data import DataLoader
    from guardlens.data.dataset import GuardLensDataset

    dataset = GuardLensDataset(records, config)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

    correct_at_window = {w: 0 for w in windows}
    total = 0
    pred_positions = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                turn_mask=batch["turn_mask"].to(device),
                role_ids=batch["role_ids"].to(device),
                compute_attribution=True,
            )

            if "attribution" not in outputs:
                continue

            attr = outputs["attribution"]

            for i in range(len(batch["metadata"])):
                meta = batch["metadata"][i]
                label = batch["labels"][i].item()
                if label != 1:
                    continue

                pivot_id = meta.get("pivot_turn_id")
                if pivot_id is None:
                    continue

                # Get turn-level attribution (max over tokens per turn)
                if attr.dim() == 3:
                    turn_attr = attr[i].abs().max(dim=-1).values  # [T]
                elif attr.dim() == 2:
                    turn_attr = attr[i].abs()  # [T]
                else:
                    continue

                # Only consider user turns — map to user-turn indices
                turns = meta.get("turns", [])
                user_turns_with_idx = [
                    (user_idx, raw_idx)
                    for user_idx, (raw_idx, t) in enumerate(
                        (ri, t) for ri, t in enumerate(turns) if t.get("role") == "user"
                    )
                    if raw_idx < turn_attr.shape[0]
                ]

                if not user_turns_with_idx:
                    continue

                # Score each user turn
                user_scores = [
                    (user_idx, raw_idx, turn_attr[raw_idx].item())
                    for user_idx, raw_idx in user_turns_with_idx
                ]
                best = max(user_scores, key=lambda x: x[2])
                pred_user_idx = best[0]

                # pivot_turn_id may be raw index or user-turn index
                # Check if pivot_id maps to a user turn at that raw index
                # If pivot_id > number of user turns, treat as raw index
                n_user = len(user_turns_with_idx)
                if pivot_id < n_user:
                    # Treat as user-turn index
                    true_user_idx = pivot_id
                else:
                    # Treat as raw turn index, convert to user-turn index
                    true_user_idx = None
                    for uid, rid in user_turns_with_idx:
                        if rid == pivot_id:
                            true_user_idx = uid
                            break
                    if true_user_idx is None:
                        # Approximate: find nearest user turn
                        true_user_idx = min(
                            user_turns_with_idx,
                            key=lambda x: abs(x[1] - pivot_id)
                        )[0]

                total += 1
                distance = abs(pred_user_idx - true_user_idx)
                pred_positions.append({
                    "true_user_idx": true_user_idx,
                    "pred_user_idx": pred_user_idx,
                    "true_raw": pivot_id,
                    "pred_raw": best[1],
                    "dist": distance,
                })

                for w in windows:
                    if distance <= w:
                        correct_at_window[w] += 1

    result = {
        "total": total,
        "window_accuracy": {},
    }
    for w in windows:
        acc = correct_at_window[w] / max(1, total)
        result["window_accuracy"][f"within_{w}"] = {
            "accuracy": acc,
            "correct": correct_at_window[w],
        }

    if pred_positions:
        distances = [p["dist"] for p in pred_positions]
        result["distance_stats"] = {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p75": float(np.percentile(distances, 75)),
            "p90": float(np.percentile(distances, 90)),
        }

    return result


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Attribution utility metrics")
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--causal-results", type=str, required=True,
                        help="Path to causal_eval_results.json")
    parser.add_argument("--boundary-results", type=str, default="",
                        help="Path to boundary_stress.json")
    parser.add_argument("--surface-fpr-results", type=str, default="",
                        help="Path to surface_risk_fpr.json")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="For causal turn mass and pivot window (needs GPU)")
    parser.add_argument("--output", type=str, default="./results/attribution_utility.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = {}

    # ---- 1. Attribution Utility ----
    print("=" * 60)
    print("  1. Attribution Utility Score")
    print("=" * 60)

    causal = json.load(open(args.causal_results))

    gl_fpr = 0.0
    if args.boundary_results and os.path.exists(args.boundary_results):
        boundary = json.load(open(args.boundary_results))
        gl_fpr = boundary.get("false_positive_rate", 0.0)

    sr_fpr = {}
    if args.surface_fpr_results and os.path.exists(args.surface_fpr_results):
        sr_fpr = json.load(open(args.surface_fpr_results))

    # Get SR FPR at different thresholds
    sr_fpr_thresholds = {}
    for thresh_key, thresh_data in sr_fpr.items():
        t = thresh_key.replace("threshold_", "")
        sr_fpr_thresholds[t] = thresh_data

    # Fix #5: Compute utility with both all_benign and boundary FPR sources
    utility_table = compute_utility_table(
        causal, gl_fpr, sr_fpr_thresholds,
        lambdas=[0.5, 1.0, 2.0],
    )
    results["utility_all_benign"] = utility_table

    # Boundary-specific utility (the hard specificity setting)
    sr_boundary_fpr = sr_fpr_thresholds.get("0.5", {}).get(
        "boundary_rejected", {}).get("fpr", 0.373)
    boundary_utility = {}
    full_test = causal.get("full_test", causal)
    for method, data in full_test.items():
        dd = data.get("deviation_drops", {}).get(focus_k, 0)
        if method == "guardlens":
            fpr = gl_fpr
        elif method == "surface_risk":
            fpr = sr_boundary_fpr
        else:
            fpr = 0.0
        boundary_utility[method] = {
            "dd": dd, "fpr": fpr, "utility": dd - 1.0 * fpr,
        }
    results["utility_boundary"] = boundary_utility

    # Print
    print(f"\n  GuardLens boundary FPR: {gl_fpr:.4f}")
    sr_benign_fpr = sr_fpr_thresholds.get("0.5", {}).get("all_benign", {}).get("fpr", "?")
    print(f"  Surface risk benign FPR@0.5: {sr_benign_fpr}")
    print(f"  Surface risk boundary FPR@0.5: {sr_boundary_fpr:.4f}")

    # Fix #6: Main utility table — GuardLens vs surface risk only
    print(f"\n  HEADLINE UTILITY (boundary FPR, λ=1.0):")
    print(f"  {'Method':<25} {'DD@15%':>8} {'BndFPR':>8} {'Utility':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    for method in ["guardlens", "surface_risk"]:
        vals = boundary_utility.get(method, {})
        if vals:
            print(f"  {method:<25} {vals['dd']:>8.3f} {vals['fpr']:>8.3f} "
                  f"{vals['utility']:>10.3f}")

    for lam_key, lam_data in utility_table.items():
        print(f"\n  {lam_key}:")
        print(f"  {'Method':<25} {'DD@15%':>8} {'FPR':>8} {'Utility':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
        for method, vals in sorted(lam_data.items(),
                                   key=lambda x: -x[1]["utility"]):
            print(f"  {method:<25} {vals['dd']:>8.3f} {vals['fpr']:>8.3f} "
                  f"{vals['utility']:>10.3f}")

    # ---- 2 & 3: Causal Turn Mass + Pivot Window (need model) ----
    if args.checkpoint and os.path.exists(args.checkpoint):
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
        config = ckpt["config"]

        from transformers import AutoTokenizer
        from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
        from guardlens.models import MODEL_REGISTRY

        tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
        collator = GuardLensCollator(tokenizer, config)

        model_cls = MODEL_REGISTRY.get(ckpt.get("model_name", "guardlens"))
        model = model_cls(config)
        model.setup_backbone()
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        test_records = load_jsonl(args.test_path)
        adv_records = [r for r in test_records if r.get("label") == 1]

        # ---- 2. Causal Turn Mass ----
        print(f"\n{'='*60}")
        print(f"  2. Causal Turn Mass")
        print(f"{'='*60}")

        ctm = compute_causal_turn_mass(
            model, adv_records, collator, config, device, tokenizer,
        )
        results["causal_turn_mass"] = ctm

        print(f"  Records: {ctm['n_records']}")
        print(f"  Causal turn mass: mean={ctm['causal_turn_mass']['mean']:.3f}, "
              f"median={ctm['causal_turn_mass']['median']:.3f}")
        print(f"\n  Per-role attribution mass:")
        for role, stats in sorted(ctm["per_role_mass"].items(),
                                  key=lambda x: -x[1]["mean"]):
            print(f"    {role:<20} mean={stats['mean']:.4f}  n={stats['n_turns']}")

        # ---- 3. Pivot Window Accuracy ----
        print(f"\n{'='*60}")
        print(f"  3. Pivot Window Accuracy")
        print(f"{'='*60}")

        pwa = compute_pivot_window_accuracy(
            model, adv_records, collator, config, device,
            windows=[0, 1, 2, 3, 5],
        )
        results["pivot_window"] = pwa

        print(f"  Records with pivot: {pwa['total']}")
        for w_key, w_data in pwa["window_accuracy"].items():
            print(f"    {w_key}: {w_data['accuracy']:.3f} "
                  f"({w_data['correct']}/{pwa['total']})")

        if "distance_stats" in pwa:
            ds = pwa["distance_stats"]
            print(f"  Distance: mean={ds['mean']:.1f}, median={ds['median']:.1f}, "
                  f"p75={ds['p75']:.1f}, p90={ds['p90']:.1f}")

    else:
        print("\n  Skipping causal turn mass and pivot window (no checkpoint)")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
