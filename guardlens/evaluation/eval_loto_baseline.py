"""
guardlens/evaluation/eval_loto_baseline.py

Leave-One-Turn-Out (LOTO) occlusion baseline for turn-level
attribution.

Reviewer i7qz: "a simple but directly relevant multi-turn
localization baseline, such as leave-one-turn-out occlusion
using the same conversation classifier."

For each adversarial conversation:
  1. Get the model's baseline adversarial score P(adv | full)
  2. For each user turn t:
     a. Mask turn t (zero its embeddings)
     b. Get P(adv | full minus turn t)
     c. Score_drop(t) = P(adv | full) - P(adv | full minus t)
  3. Rank user turns by score_drop (highest = most important)
  4. Compare against GuardLens attribution using top-k hit rate

This is a strong baseline because it uses the same model and
the same classification head. If LOTO matches GuardLens, the
attribution architecture adds nothing over simple occlusion.
If GuardLens beats LOTO, the cross-turn attention and gated
fusion are learning something occlusion misses.

Usage:
    python -m guardlens.evaluation.eval_loto_baseline \
        --test-path splits/test.jsonl \
        --checkpoint checkpoints/guardlens/best_attribution.pt \
        --output results/loto_baseline.json
"""

import argparse
import json
import os
import sys
from math import comb
from typing import Dict, List, Optional

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
    surface_risk_attribution,
    random_attribution,
    ATTRIBUTION_METHODS,
)


# ================================================================
# Causal span detection (shared with eval_topk_turn_hitrate)
# ================================================================

CAUSAL_SPAN_TYPES = {
    "MALICIOUS_TRIGGER", "PAYLOAD_SPAN", "CONTEXT_BRIDGE",
    "STRUCTURAL_TRIGGER", "IMPLICIT_TRIGGER",
}
CAUSAL_TIERS = {"cf_strong", "cf_weak", "llm_confirmed", "construction"}
NON_CAUSAL_SPAN_TYPES = {
    "DECOY", "BENIGN_CONTEXT", "SAFE_CONSTRAINT", "QUOTED_UNSAFE_CONTENT",
}
NON_CAUSAL_ROLES = {"decoy", "benign_context", "incidental"}


def is_causal_span(span: Dict) -> bool:
    if span.get("causal_type") == "causal":
        return True
    if span.get("causal_type") == "incidental":
        return False
    span_type = span.get("span_type") or span.get("label", "")
    if span_type in NON_CAUSAL_SPAN_TYPES:
        return False
    if span_type in CAUSAL_SPAN_TYPES:
        return True
    tier = span.get("supervision_tier") or span.get("tier", "")
    if tier in CAUSAL_TIERS and span_type not in NON_CAUSAL_SPAN_TYPES:
        return True
    causal_role = span.get("causal_role", "")
    if causal_role and causal_role not in NON_CAUSAL_ROLES:
        return True
    return False


def turn_has_causal_spans(turn: Dict) -> bool:
    for span in turn.get("span_annotations", []):
        if is_causal_span(span):
            return True
    return False


def expected_random_hit_rate(T: int, M: int, k: int) -> float:
    T, M, k = max(T, 0), min(max(M, 0), T), min(max(k, 0), T)
    if k <= 0 or T <= 0 or M <= 0:
        return 0.0
    if k >= T or T - M < k:
        return 1.0
    try:
        return 1.0 - comb(T - M, k) / comb(T, k)
    except (ValueError, ZeroDivisionError):
        return 1.0


# ================================================================
# LOTO computation
# ================================================================

def compute_loto_scores(
    model, batch: Dict, device: torch.device,
) -> torch.Tensor:
    """Compute leave-one-turn-out score drops for each turn.

    For each sample in the batch, masks one user turn at a time
    and measures the drop in P(adversarial).

    Returns:
        [B, T] tensor of score drops per turn (higher = more important).
        Non-user turns and padded turns get 0.
    """
    B = batch["input_ids"].size(0)
    T = batch["input_ids"].size(1)

    # Baseline: full conversation score
    with torch.no_grad():
        base_out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            turn_mask=batch["turn_mask"].to(device),
            role_ids=batch["role_ids"].to(device),
            compute_attribution=False,
        )
    base_probs = torch.sigmoid(base_out["cls_logits"]).cpu()  # [B]

    drops = torch.zeros(B, T)

    for t in range(T):
        # Check if any sample has a valid user turn at position t
        has_valid = False
        for b in range(B):
            if (batch["turn_mask"][b, t] == 1 and
                    batch["metadata"][b].get("turns_roles", [None] * T)[t] == "user"
                    if "turns_roles" in batch["metadata"][b]
                    else True):  # fallback: try all turns
                has_valid = True
                break
        if not has_valid:
            continue

        # Create masked turn_mask: zero out turn t
        masked_turn_mask = batch["turn_mask"].clone()
        masked_turn_mask[:, t] = 0

        with torch.no_grad():
            masked_out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                turn_mask=masked_turn_mask.to(device),
                role_ids=batch["role_ids"].to(device),
                compute_attribution=False,
            )
        masked_probs = torch.sigmoid(masked_out["cls_logits"]).cpu()  # [B]

        drops[:, t] = base_probs - masked_probs

    return drops


def compute_loto_hit_rates(
    model,
    loader: DataLoader,
    records: List[Dict],
    device: torch.device,
    k_values: List[int] = [1, 2, 3, 5],
) -> Dict:
    """Compute top-k turn hit rates using LOTO occlusion."""
    model.eval()
    per_record = []

    for batch in loader:
        # LOTO scores
        loto_drops = compute_loto_scores(model, batch, device)

        for i, meta in enumerate(batch["metadata"]):
            if batch["labels"][i].item() != 1:
                continue

            cid = meta.get("conversation_id", "")
            record = next(
                (r for r in records if r.get("conversation_id") == cid),
                None,
            )
            if record is None:
                continue

            turns = record.get("turns", [])
            turn_mask = batch["turn_mask"][i]

            # Get per-user-turn LOTO drop scores
            user_turn_scores = []
            for t in range(min(len(turns), turn_mask.size(0))):
                if turn_mask[t] == 0 or turns[t].get("role") != "user":
                    continue
                user_turn_scores.append((t, loto_drops[i, t].item()))

            if not user_turn_scores:
                continue

            n_user = len(user_turn_scores)

            # Ground truth
            gt_causal = set()
            pivot_id = record.get("pivot_turn_id")
            if pivot_id is not None:
                gt_causal.add(pivot_id)
            for t_idx, turn in enumerate(turns):
                if turn.get("role") == "user" and turn_has_causal_spans(turn):
                    gt_causal.add(t_idx)

            if not gt_causal:
                continue

            n_causal = len(gt_causal)

            # Rank by LOTO drop (highest first)
            ranked = sorted(user_turn_scores, key=lambda x: -x[1])
            ranked_indices = [x[0] for x in ranked]

            result = {
                "conversation_id": cid,
                "n_user": n_user,
                "n_causal": n_causal,
                "hits": {},
                "floors": {},
                "base_prob": None,
                "top_drop": ranked[0][1] if ranked else 0,
            }

            for k in k_values:
                top_k_set = set(ranked_indices[:min(k, n_user)])
                hit = int(len(top_k_set & gt_causal) > 0)
                floor = expected_random_hit_rate(n_user, n_causal, k)
                result["hits"][k] = hit
                result["floors"][k] = floor

            per_record.append(result)

    # Aggregate
    metrics = {"n_evaluated": len(per_record), "method": "loto"}
    for k in k_values:
        hits = [r["hits"][k] for r in per_record]
        floors = [r["floors"][k] for r in per_record]
        hit_rate = float(np.mean(hits)) if hits else 0
        floor = float(np.mean(floors)) if floors else 0
        metrics[f"top_{k}"] = {
            "hit_rate": hit_rate,
            "random_floor": floor,
            "margin": hit_rate - floor,
            "per_record_hits": [int(h) for h in hits],
            "per_record_floors": [float(f) for f in floors],
        }

    return metrics


def compute_loto_dd(
    model,
    loader: DataLoader,
    records: List[Dict],
    device: torch.device,
    k_fracs: List[float] = [0.05, 0.10, 0.15, 0.20],
) -> Dict:
    """Compute DD@k using LOTO-ranked tokens for deletion.

    LOTO ranks turns, then within each turn uses uniform token
    weight. The top-k% tokens are selected by first taking the
    highest-LOTO-drop turns, then all tokens within those turns.
    """
    from guardlens.evaluation.causal_eval import _get_prob, _single_batch

    model.eval()
    results = {}

    for k_frac in k_fracs:
        all_drops = []

        for batch in loader:
            loto_drops = compute_loto_scores(model, batch, device)

            for i in range(len(batch["labels"])):
                if batch["labels"][i].item() != 1:
                    continue

                cid = batch["metadata"][i].get("conversation_id", "")
                record = next(
                    (r for r in records if r.get("conversation_id") == cid),
                    None,
                )
                if record is None:
                    continue

                turns = record.get("turns", [])
                turn_mask = batch["turn_mask"][i]
                attn_mask = batch["attention_mask"][i]

                # Build per-token LOTO scores: each token gets its turn's drop score
                T, S = attn_mask.shape
                token_scores = torch.zeros(T, S)
                user_turn_valid = torch.zeros(T, dtype=torch.bool)

                for t in range(min(len(turns), T)):
                    if turn_mask[t] == 0:
                        continue
                    if turns[t].get("role") != "user":
                        continue
                    token_scores[t, :] = loto_drops[i, t]
                    user_turn_valid[t] = True

                valid = (attn_mask.bool() & user_turn_valid.unsqueeze(-1))
                flat_scores = token_scores[valid].flatten()
                n_tokens = flat_scores.numel()
                if n_tokens == 0:
                    continue

                k = max(1, int(n_tokens * k_frac))
                topk_idx = torch.topk(flat_scores, k).indices
                flat_mask = torch.ones(n_tokens, dtype=torch.bool)
                flat_mask[topk_idx] = False

                full_mask = torch.ones_like(token_scores)
                full_mask[valid] = flat_mask.float()

                # Get baseline prob
                sb = _single_batch(batch, i)
                orig_prob = _get_prob(model, sb, device)[0].item()
                if orig_prob < 0.5:
                    continue

                masked_prob = _get_prob(
                    model, sb, device,
                    attribution_mask=full_mask.unsqueeze(0),
                )[0].item()

                all_drops.append(orig_prob - masked_prob)

        k_str = f"{int(k_frac * 100)}%"
        results[k_str] = {
            "dd": float(np.mean(all_drops)) if all_drops else 0,
            "n": len(all_drops),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-turn-out occlusion baseline",
    )
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="./results/loto_baseline.json")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--top-k-fracs", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")
    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = GuardLensCollator(tokenizer, config)

    from guardlens.evaluation.eval_utils import load_jsonl
    records = load_jsonl(args.test_path)
    n_adv = sum(1 for r in records if r.get("label") == 1)
    print(f"Records: {len(records)} ({n_adv} adversarial)")

    dataset = GuardLensDataset(records, config)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=collator, num_workers=4)

    # LOTO turn hit rates
    print("\n  Computing LOTO turn hit rates...")
    hit_rates = compute_loto_hit_rates(
        model, loader, records, device, args.k_values,
    )

    print(f"\n  LOTO Turn Hit Rates:")
    for k in args.k_values:
        d = hit_rates.get(f"top_{k}", {})
        print(f"    top-{k}: hit={d.get('hit_rate',0):.3f}  "
              f"random={d.get('random_floor',0):.3f}  "
              f"margin={d.get('margin',0):+.3f}")

    # LOTO DD
    print("\n  Computing LOTO DD...")
    dd_results = compute_loto_dd(
        model, loader, records, device, args.top_k_fracs,
    )

    print(f"\n  LOTO Deviation Drop:")
    for k_str, dd in dd_results.items():
        print(f"    DD@{k_str}: {dd['dd']:.3f} (n={dd['n']})")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "method": "loto",
        "turn_hit_rates": hit_rates,
        "deviation_drop": dd_results,
        "k_values": args.k_values,
        "top_k_fracs": args.top_k_fracs,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
