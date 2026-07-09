"""
guardlens/evaluation/eval_topk_turn_hitrate.py

Top-k turn hit rate with random floor and baseline comparison.

Reviewer aeLx's concern: "Top-five actually gives the model many
chances to hit" -- with ~7 user turns and multiple annotated causal
turns, top-5 is extremely generous.

This script computes:
  1. Hit rate at top-1, top-2, top-3, top-5 for all attribution methods
  2. Random floor: expected hit rate at each k given T user turns and
     M marked causal turns per conversation
  3. Margin above random for each method
  4. Per-conversation breakdown for bootstrap CI computation

The margin above random is the defensible metric, not the absolute
hit rate.

Usage:
    python -m guardlens.evaluation.eval_topk_turn_hitrate \\
        --test-path splits/test.jsonl \\
        --checkpoint checkpoints/guardlens/best.pt \\
        --output results/topk_turn_hitrate.json \\
        --device cuda
"""

import argparse
import json
import os
import sys
from math import comb
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from guardlens.evaluation.causal_eval import (
    guardlens_attribution,
    attention_attribution,
    integrated_gradients_attribution,
    gradient_x_input_attribution,
    surface_risk_attribution,
    random_attribution,
    ATTRIBUTION_METHODS,
)


# -------------------------------------------------------
# Robust causal span detection
#
# The dataset uses multiple overlapping fields to indicate
# causality: causal_type, label/span_type, supervision_tier,
# causal_role. A span is causal if ANY of these signals
# indicates causality, unless it is an explicit negative
# (DECOY, BENIGN_CONTEXT, incidental).
# -------------------------------------------------------

CAUSAL_SPAN_TYPES = {
    "MALICIOUS_TRIGGER",
    "PAYLOAD_SPAN",
    "CONTEXT_BRIDGE",
    "STRUCTURAL_TRIGGER",
    "IMPLICIT_TRIGGER",
}

CAUSAL_TIERS = {
    "cf_strong",
    "cf_weak",
    "llm_confirmed",
    "construction",
}

NON_CAUSAL_SPAN_TYPES = {
    "DECOY",
    "BENIGN_CONTEXT",
    "SAFE_CONSTRAINT",
    "QUOTED_UNSAFE_CONTENT",
}

NON_CAUSAL_ROLES = {
    "decoy",
    "benign_context",
    "incidental",
}


def is_causal_span(span: Dict) -> bool:
    """
    Determine if a span annotation indicates causal attribution.

    Checks multiple fields in priority order:
      1. causal_type == "causal" (v11 primary signal)
      2. label/span_type in CAUSAL_SPAN_TYPES
      3. supervision_tier in CAUSAL_TIERS AND label not in NON_CAUSAL_SPAN_TYPES
      4. causal_role not in NON_CAUSAL_ROLES

    Returns False for explicit negatives (DECOY, BENIGN_CONTEXT, incidental).
    """
    # Explicit causal_type field (v11 primary)
    if span.get("causal_type") == "causal":
        return True
    if span.get("causal_type") == "incidental":
        return False

    # Label / span_type field
    span_type = span.get("span_type") or span.get("label", "")
    if span_type in NON_CAUSAL_SPAN_TYPES:
        return False
    if span_type in CAUSAL_SPAN_TYPES:
        return True

    # Supervision tier (only if span type is not an explicit negative)
    tier = span.get("supervision_tier") or span.get("tier", "")
    if tier in CAUSAL_TIERS and span_type not in NON_CAUSAL_SPAN_TYPES:
        return True

    # Causal role field
    causal_role = span.get("causal_role", "")
    if causal_role and causal_role not in NON_CAUSAL_ROLES:
        return True

    return False


def turn_has_causal_spans(turn: Dict) -> bool:
    """Check if a turn contains at least one causal span annotation."""
    for span in turn.get("span_annotations", []):
        if is_causal_span(span):
            return True
    return False


def expected_random_hit_rate(n_user_turns: int, n_causal: int, k: int) -> float:
    """
    Expected hit rate for random top-k selection.

    P(at least one causal turn in top-k) = 1 - C(T-M, k) / C(T, k)
    where T = n_user_turns, M = n_causal, k = top-k.

    Returns 1.0 if k >= T or k >= T - M + 1 (guaranteed hit).
    """
    T = n_user_turns
    M = min(n_causal, T)
    k = min(k, T)

    if k <= 0 or T <= 0 or M <= 0:
        return 0.0
    if k >= T:
        return 1.0
    if T - M < k:
        # Not enough non-causal turns to fill top-k without a causal one
        return 1.0

    # P(miss) = C(T-M, k) / C(T, k)
    try:
        p_miss = comb(T - M, k) / comb(T, k)
    except (ValueError, ZeroDivisionError):
        return 1.0

    return 1.0 - p_miss


def compute_topk_hit_rates(
    attr_scores: torch.Tensor,
    batch: Dict,
    records: List[Dict],
    k_values: List[int] = [1, 2, 3, 5],
) -> Dict:
    """
    Compute top-k turn hit rates for a batch.

    For each adversarial conversation:
      1. Compute per-USER-turn mean attribution score
      2. Rank user turns by attribution score
      3. Check if any marked causal turn appears in top-k
      4. Compute the random floor for this specific T, M, k

    Returns per-record results for aggregation.
    """
    per_record = []

    for i, meta in enumerate(batch["metadata"]):
        label = batch["labels"][i].item()
        if label != 1:
            continue

        pivot_id = meta.get("pivot_turn_id")
        if pivot_id is None:
            continue

        # Find the original record to get turn info
        cid = meta.get("conversation_id", "")
        record = None
        for r in records:
            if r.get("conversation_id") == cid:
                record = r
                break

        if record is None:
            continue

        turns = record.get("turns", [])
        turn_mask = batch["turn_mask"][i]
        attn_mask = batch["attention_mask"][i]
        scores_i = attr_scores[i]

        # Get per-turn attribution scores for USER turns only
        user_turn_scores = []  # (user_turn_index, raw_turn_index, mean_attr_score)
        user_idx = 0
        for t in range(min(len(turns), turn_mask.size(0))):
            if turn_mask[t] == 0:
                continue
            if t < len(turns) and turns[t].get("role") == "user":
                valid = attn_mask[t].bool()
                if valid.any():
                    score = scores_i[t][valid].mean().item()
                else:
                    score = 0.0
                user_turn_scores.append((user_idx, t, score))
                user_idx += 1

        if not user_turn_scores:
            continue

        n_user = len(user_turn_scores)

        # Identify causal user turns (from pivot_turn_id)
        # pivot_turn_id is the raw turn index
        causal_raw_indices = set()
        causal_raw_indices.add(pivot_id)

        # Also check for additional causal turns from span annotations
        for t_idx, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue
            if turn_has_causal_spans(turn):
                causal_raw_indices.add(t_idx)

        # Map causal raw indices to user-turn indices
        causal_user_indices = set()
        for uid, rid, _ in user_turn_scores:
            if rid in causal_raw_indices:
                causal_user_indices.add(uid)

        n_causal = len(causal_user_indices)
        if n_causal == 0:
            continue

        # Sort user turns by attribution score (descending)
        ranked = sorted(user_turn_scores, key=lambda x: -x[2])
        ranked_user_indices = [x[0] for x in ranked]

        # Compute hit rate at each k
        record_result = {
            "conversation_id": cid,
            "n_user_turns": n_user,
            "n_causal_turns": n_causal,
            "hits": {},
            "random_floor": {},
        }

        for k in k_values:
            if k > n_user:
                # Can always hit if k > n_user
                record_result["hits"][k] = 1
                record_result["random_floor"][k] = 1.0
                continue

            top_k_set = set(ranked_user_indices[:k])
            hit = int(len(top_k_set & causal_user_indices) > 0)
            random_p = expected_random_hit_rate(n_user, n_causal, k)

            record_result["hits"][k] = hit
            record_result["random_floor"][k] = random_p

        per_record.append(record_result)

    return per_record


def main():
    parser = argparse.ArgumentParser(
        description="Top-k turn hit rate with random floor",
    )
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="./results/topk_turn_hitrate.json")
    parser.add_argument("--methods", nargs="+",
                        default=["guardlens", "surface_risk",
                                 "attention", "random"])
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=[1, 2, 3, 5])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")

    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = GuardLensCollator(tokenizer, config)

    from guardlens.evaluation.eval_utils import load_jsonl
    records = load_jsonl(args.test_path)
    print(f"Test records: {len(records)}")
    n_adv = sum(1 for r in records if r.get("label") == 1)
    print(f"Adversarial: {n_adv}")

    dataset = GuardLensDataset(records, config)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=collator, num_workers=args.workers,
    )

    # Compute distribution stats
    user_turn_counts = []
    causal_turn_counts = []
    for r in records:
        if r.get("label") != 1:
            continue
        n_user = sum(1 for t in r.get("turns", []) if t.get("role") == "user")
        user_turn_counts.append(n_user)
        n_causal = 0
        for t in r.get("turns", []):
            if t.get("role") != "user":
                continue
            if turn_has_causal_spans(t):
                n_causal += 1
        if r.get("pivot_turn_id") is not None:
            n_causal = max(n_causal, 1)
        causal_turn_counts.append(n_causal)

    if user_turn_counts:
        print(f"\nUser turns per adversarial conversation:")
        print(f"  mean={np.mean(user_turn_counts):.1f}  "
              f"median={np.median(user_turn_counts):.0f}  "
              f"range=[{min(user_turn_counts)}, {max(user_turn_counts)}]")
        print(f"Causal turns per conversation:")
        print(f"  mean={np.mean(causal_turn_counts):.1f}  "
              f"median={np.median(causal_turn_counts):.0f}  "
              f"range=[{min(causal_turn_counts)}, {max(causal_turn_counts)}]")

        # Compute expected random floor
        print(f"\nExpected random floor (averaged over dataset):")
        for k in args.k_values:
            floors = [
                expected_random_hit_rate(t, c, k)
                for t, c in zip(user_turn_counts, causal_turn_counts)
                if c > 0
            ]
            if floors:
                print(f"  top-{k}: {np.mean(floors):.4f}")

    # Run for each attribution method
    all_results = {}

    for method_name in args.methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method_name}")
        print(f"{'='*60}")

        attr_fn = ATTRIBUTION_METHODS.get(method_name)
        if attr_fn is None:
            print(f"  Skipped (not registered)")
            continue

        all_per_record = []

        for batch in loader:
            if method_name == "surface_risk":
                attr_scores = attr_fn(model, batch, device, tokenizer=tokenizer)
            elif method_name in ("grad_x_input", "integrated_gradients"):
                with torch.enable_grad():
                    attr_scores = attr_fn(model, batch, device)
            else:
                attr_scores = attr_fn(model, batch, device)

            per_record = compute_topk_hit_rates(
                attr_scores, batch, records, args.k_values,
            )
            all_per_record.extend(per_record)

        # Aggregate
        method_results = {
            "n_evaluated": len(all_per_record),
        }

        for k in args.k_values:
            hits = [r["hits"].get(k, 0) for r in all_per_record]
            floors = [r["random_floor"].get(k, 0) for r in all_per_record]

            hit_rate = np.mean(hits) if hits else 0
            random_floor = np.mean(floors) if floors else 0
            margin = hit_rate - random_floor

            method_results[f"top_{k}"] = {
                "hit_rate": float(hit_rate),
                "random_floor": float(random_floor),
                "margin": float(margin),
                "n": len(hits),
                "per_record_hits": [int(h) for h in hits],
            }

            print(f"  top-{k}: hit={hit_rate:.4f}  "
                  f"random={random_floor:.4f}  "
                  f"margin={margin:+.4f}")

        all_results[method_name] = method_results

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  TOP-K TURN HIT RATE COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Method':<22}", end="")
    for k in args.k_values:
        print(f" {'top-'+str(k):>12}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in args.k_values:
        print(f" {'-'*12}", end="")
    print()

    # Random floor row
    print(f"  {'Random floor':<22}", end="")
    for k in args.k_values:
        # Average random floor across all methods (should be the same)
        floors = []
        for m in all_results.values():
            f = m.get(f"top_{k}", {}).get("random_floor", 0)
            if f > 0:
                floors.append(f)
        floor = np.mean(floors) if floors else 0
        print(f" {floor:>12.4f}", end="")
    print()

    # Method rows
    for method_name in args.methods:
        if method_name not in all_results:
            continue
        m = all_results[method_name]
        print(f"  {method_name:<22}", end="")
        for k in args.k_values:
            hr = m.get(f"top_{k}", {}).get("hit_rate", 0)
            print(f" {hr:>12.4f}", end="")
        print()

    # Margin rows
    print()
    print(f"  MARGIN ABOVE RANDOM:")
    for method_name in args.methods:
        if method_name not in all_results:
            continue
        m = all_results[method_name]
        print(f"  {method_name:<22}", end="")
        for k in args.k_values:
            margin = m.get(f"top_{k}", {}).get("margin", 0)
            print(f" {margin:>+12.4f}", end="")
        print()

    # Diagnostic
    print(f"\n  DIAGNOSTIC:")
    for method_name in args.methods:
        if method_name not in all_results:
            continue
        m = all_results[method_name]
        top5 = m.get("top_5", {})
        if top5.get("margin", 0) <= 0:
            print(f"  WARNING: {method_name} top-5 margin is non-positive "
                  f"({top5.get('margin',0):+.4f}). "
                  f"Top-5 is uninformative for this method.")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "methods": args.methods,
        "k_values": args.k_values,
        "results": all_results,
        "dataset_stats": {
            "n_adversarial": n_adv,
            "mean_user_turns": float(np.mean(user_turn_counts)) if user_turn_counts else 0,
            "median_user_turns": float(np.median(user_turn_counts)) if user_turn_counts else 0,
            "mean_causal_turns": float(np.mean(causal_turn_counts)) if causal_turn_counts else 0,
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
