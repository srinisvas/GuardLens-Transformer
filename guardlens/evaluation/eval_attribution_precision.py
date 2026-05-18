"""
guardlens/evaluation/eval_attribution_precision.py

Two analyses not yet in the evaluation suite:

1. HARD NEGATIVE ATTRIBUTION PRECISION
   On benign samples with adversarial-sounding vocabulary (hard negatives,
   borderline samples, false positive traps), does GuardLens hallucinate
   causal tokens? Measures:
     - False Positive Attribution Rate (FPAR): fraction of tokens in benign
       samples that receive high attribution (> 0.5). Ideally near 0.
     - Attribution Sparsity on Benign: entropy / concentration of attribution
       mass. A good model should have diffuse, low-confidence attribution on
       benign samples, not concentrated high scores.
     - Specificity: FPAR(hard_negative) vs FPAR(genuine_benign). Hard negatives
       should be harder -- the gap quantifies how well the model handles them.

2. MINIMALITY SENSITIVITY CURVE
   Current results show median trigger size = 7 tokens (GuardLens) but this
   alone doesn't prove minimality. The sensitivity curve tests:
     - Compute flip rate at k, k-Δ, k-2Δ, ...
     - A sharp drop from k → k-Δ means the model found a compact, necessary
       set. Flat curve means the trigger set is over-estimated.
   Formally: if FlipRate(k%) >> FlipRate((k-5)%) for small k, the trigger is
   near-minimal. If the curve is flat, attribution includes redundant tokens.

   We plot: flip rate vs k for {guardlens, IG, surface_risk, random}
   GuardLens should have the sharpest inflection point, demonstrating the
   trigger identification is both precise and near-minimal.

Usage:
    python -m guardlens.evaluation.eval_attribution_precision \\
        --data ~/work/results/dataset_gen/splits/test.jsonl \\
        --checkpoint ~/work/results/dataset_gen/checkpoints/guardlens/best.pt \\
        --output ~/work/results/dataset_gen/results/attribution_precision.json \\
        --device cuda
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.data.splits import pair_aware_split
from guardlens.models import MODEL_REGISTRY


# -------------------------------------------------------
# Attribution extraction (all methods)
# -------------------------------------------------------

def get_guardlens_scores(model, batch, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            turn_mask=batch["turn_mask"].to(device),
            role_ids=batch["role_ids"].to(device),
            compute_attribution=True,
        )
    return out["attr_probs"].cpu() if out["attr_probs"] is not None else \
        torch.zeros_like(batch["input_ids"], dtype=torch.float32)


def get_random_scores(batch) -> torch.Tensor:
    scores = torch.rand_like(batch["input_ids"].float())
    mask = (batch["attention_mask"] * batch["turn_mask"].unsqueeze(-1)).float()
    return scores * mask


def get_surface_risk_scores(batch, tokenizer) -> torch.Tensor:
    """Surface risk: score each token by membership in SR vocabulary."""
    _SR = {
        "improve", "success", "rate", "detectable", "blend", "convincing",
        "scrutiny", "reduce", "suspicion", "bypass", "exploit", "override",
        "jailbreak", "payload", "adversarial", "evade", "evasion",
        "credential", "harvest", "phishing", "malware", "inject",
        "realistic", "credible", "persuasive", "plausibility", "stealth",
        "optimize", "refine", "operational", "effective",
    }
    B, T, S = batch["input_ids"].shape
    scores = torch.zeros(B, T, S)
    for b in range(B):
        for t in range(T):
            for s in range(S):
                tid = batch["input_ids"][b, t, s].item()
                if tid == 0:
                    continue
                word = tokenizer.decode([tid], skip_special_tokens=True)
                if word.lower().strip(".,!?") in _SR:
                    scores[b, t, s] = 1.0
    mask = (batch["attention_mask"] * batch["turn_mask"].unsqueeze(-1)).float()
    return scores * mask


ATTRIBUTION_FNS = {
    "guardlens": get_guardlens_scores,
    "random": lambda model, batch, device: get_random_scores(batch),
}


# -------------------------------------------------------
# 1. Hard Negative Attribution Precision
# -------------------------------------------------------

def compute_fpar(
    attr_scores: torch.Tensor,   # [B, T, S]
    attention_mask: torch.Tensor,# [B, T, S]
    turn_mask: torch.Tensor,     # [B, T]
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    False Positive Attribution Rate:
      Fraction of valid tokens in the batch with attr_score > threshold.

    Also computes:
      - Mean attribution score (lower = model is uncertain, good for benign)
      - Attribution entropy (higher = more diffuse, good for benign)
      - Max attribution score
    """
    valid_mask = (attention_mask * turn_mask.unsqueeze(-1)).bool()  # [B, T, S]

    all_scores = attr_scores[valid_mask]  # [N_valid]
    if all_scores.numel() == 0:
        return {"fpar": 0.0, "mean_score": 0.0, "entropy": 0.0, "max_score": 0.0}

    fpar = (all_scores > threshold).float().mean().item()
    mean_score = all_scores.mean().item()
    max_score = all_scores.max().item()

    # Attribution entropy: treat per-sample attr as a distribution
    # High entropy = diffuse (good for benign), low = concentrated (good for adversarial)
    entropies = []
    B = attr_scores.shape[0]
    for b in range(B):
        valid_b = valid_mask[b]
        scores_b = attr_scores[b][valid_b]
        if scores_b.numel() < 2:
            continue
        # Normalize to probability distribution
        p = F.softmax(scores_b.float() * 5.0, dim=0)  # temperature=5 for sharper dist
        ent = -(p * (p + 1e-10).log()).sum().item()
        entropies.append(ent)

    return {
        "fpar": fpar,
        "mean_score": mean_score,
        "max_score": max_score,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "n_tokens": all_scores.numel(),
    }


@torch.no_grad()
def run_hard_negative_analysis(
    model: torch.nn.Module,
    records: List[Dict],
    test_idx: List[int],
    collator: GuardLensCollator,
    config: GuardLensConfig,
    device: torch.device,
    batch_size: int = 8,
) -> Dict:
    """
    Partition test set into subsets and compute attribution statistics for each.
    """
    dataset = GuardLensDataset(records, config)

    # Partition test indices — v11 compatible
    all_records = [records[i] for i in test_idx]
    subsets = {
        "adversarial":     [],  # label=1, genuine adversarial
        "hard_negative":   [],  # label=0, validated_benign_twin or hard_benign/topic_matched
        "borderline":      [],  # label=0, false_lead_benign
        "genuine_benign":  [],  # label=0, clean_benign from separate pool
        "false_pos_trap":  [],  # label=0, research_technical (uses risky vocabulary safely)
    }

    for local_i, global_i in enumerate(test_idx):
        r = records[global_i]
        label = r.get("label", 0)
        family = r.get("family", "")
        benign_status = r.get("benign_status", "none")

        if label == 1:
            subsets["adversarial"].append(global_i)
        elif benign_status == "validated_benign_twin":
            subsets["hard_negative"].append(global_i)
        elif family in ("hard_benign", "topic_matched_safe"):
            subsets["hard_negative"].append(global_i)
        elif family == "false_lead_benign":
            subsets["borderline"].append(global_i)
        elif family == "research_technical":
            subsets["false_pos_trap"].append(global_i)
        elif benign_status == "clean_benign":
            subsets["genuine_benign"].append(global_i)
        else:
            subsets["genuine_benign"].append(global_i)

    print(f"\n  Subset sizes:")
    for name, indices in subsets.items():
        print(f"    {name:<20}: {len(indices)}")

    results = {}
    for subset_name, global_indices in subsets.items():
        if len(global_indices) < 3:
            print(f"\n  Skipping {subset_name} (too few samples)")
            continue

        print(f"\n  Processing {subset_name} ({len(global_indices)} samples)...")
        loader = DataLoader(
            Subset(dataset, global_indices),
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

        all_fpar_stats = []
        for batch in loader:
            attr = get_guardlens_scores(model, batch, device)
            stats = compute_fpar(
                attr,
                batch["attention_mask"],
                batch["turn_mask"],
                threshold=0.5,
            )
            all_fpar_stats.append(stats)

        # Aggregate
        mean_fpar = np.mean([s["fpar"] for s in all_fpar_stats])
        mean_score = np.mean([s["mean_score"] for s in all_fpar_stats])
        mean_entropy = np.mean([s["entropy"] for s in all_fpar_stats])
        mean_max = np.mean([s["max_score"] for s in all_fpar_stats])
        total_tokens = sum(s["n_tokens"] for s in all_fpar_stats)

        results[subset_name] = {
            "n_conversations": len(global_indices),
            "total_tokens": total_tokens,
            "fpar": float(mean_fpar),           # FALSE POSITIVE ATTRIBUTION RATE
            "mean_attribution": float(mean_score),
            "mean_max_attribution": float(mean_max),
            "attribution_entropy": float(mean_entropy),
        }

        print(f"    FPAR:              {mean_fpar:.4f}")
        print(f"    Mean attribution:  {mean_score:.4f}")
        print(f"    Max attribution:   {mean_max:.4f}")
        print(f"    Attr entropy:      {mean_entropy:.4f}")

    return results


# -------------------------------------------------------
# 2. Minimality Sensitivity Curve
# -------------------------------------------------------

@torch.no_grad()
def compute_flip_rate_at_k(
    model: torch.nn.Module,
    batch: Dict,
    attr_scores: torch.Tensor,
    device: torch.device,
    k_frac: float,
) -> Tuple[int, int]:
    """Compute (n_flipped, n_tested) at a given k fraction."""
    labels = batch["labels"]
    adv_idx = (labels == 1).nonzero(as_tuple=True)[0]
    flips = 0
    tested = 0

    for i in adv_idx:
        i = i.item()
        # Check original prediction
        single_input = {
            "input_ids": batch["input_ids"][i:i+1].to(device),
            "attention_mask": batch["attention_mask"][i:i+1].to(device),
            "turn_mask": batch["turn_mask"][i:i+1].to(device),
            "role_ids": batch["role_ids"][i:i+1].to(device),
        }
        orig_out = model(**single_input, compute_attribution=False)
        orig_prob = torch.sigmoid(orig_out["cls_logits"])[0].item()
        if orig_prob < 0.5:
            continue  # Not classified as adversarial, skip

        # Build top-k mask
        valid = (batch["attention_mask"][i] * batch["turn_mask"][i].unsqueeze(-1)).bool()
        flat_scores = attr_scores[i][valid]
        n = flat_scores.numel()
        if n == 0:
            continue
        k = max(1, int(n * k_frac))
        threshold = flat_scores.topk(k).values[-1].item()
        mask = (attr_scores[i] < threshold).float().to(device)

        # Counterfactual forward
        cf_out = model(**single_input, compute_attribution=False,
                       attribution_mask=mask.unsqueeze(0))
        cf_prob = torch.sigmoid(cf_out["cls_logits"])[0].item()

        if cf_prob < 0.5:
            flips += 1
        tested += 1

    return flips, tested


def run_minimality_curve(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    methods: Dict[str, callable],
    k_fracs: List[float],
) -> Dict:
    """
    Compute flip rate at each k fraction for each attribution method.
    Returns {method: {k_str: flip_rate}} for plotting.
    """
    # Accumulate (flips, tested) per method per k
    counts = {
        method: {f"{int(k*100)}%": [0, 0] for k in k_fracs}
        for method in methods
    }

    n_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{n_batches}...")

        for method_name, attr_fn in methods.items():
            # Get attribution scores
            if method_name in ("grad_x_input", "integrated_gradients"):
                with torch.enable_grad():
                    attr_scores = attr_fn(model, batch, device)
            else:
                attr_scores = attr_fn(model, batch, device)

            # Compute flip rate at each k
            for k_frac in k_fracs:
                k_str = f"{int(k_frac*100)}%"
                flips, tested = compute_flip_rate_at_k(
                    model, batch, attr_scores, device, k_frac
                )
                counts[method_name][k_str][0] += flips
                counts[method_name][k_str][1] += tested

    # Compute rates
    curves = {}
    for method_name, k_counts in counts.items():
        curves[method_name] = {}
        for k_str, (flips, tested) in k_counts.items():
            curves[method_name][k_str] = flips / max(1, tested)

    return curves


def compute_sharpness(curve: Dict[str, float], k_fracs: List[float]) -> Dict:
    """
    Quantify how 'sharp' the flip rate curve is.

    A near-minimal trigger set has a sharp inflection: flip rate is low at
    small k, then jumps sharply at the trigger size. Flat curves indicate
    redundancy (many tokens can each independently flip the prediction).

    Metrics:
      - inflection_k: k where flip rate first exceeds 0.5
      - slope_at_inflection: average slope around that k (larger = sharper)
      - auc: area under the curve (lower AUC at small k = more compact trigger)
    """
    rates = [curve.get(f"{int(k*100)}%", 0.0) for k in k_fracs]

    # Inflection point: first k where flip rate >= 0.5
    inflection_idx = next((i for i, r in enumerate(rates) if r >= 0.5), len(rates) - 1)
    inflection_k = k_fracs[inflection_idx] if inflection_idx < len(k_fracs) else k_fracs[-1]

    # Slope at inflection
    if 0 < inflection_idx < len(rates) - 1:
        slope = (rates[inflection_idx + 1] - rates[inflection_idx - 1]) / \
                (k_fracs[inflection_idx + 1] - k_fracs[inflection_idx - 1])
    elif inflection_idx > 0:
        slope = (rates[inflection_idx] - rates[inflection_idx - 1]) / \
                (k_fracs[inflection_idx] - k_fracs[inflection_idx - 1])
    else:
        slope = 0.0

    # AUC (trapezoidal)
    auc = float(np.trapezoid(rates, k_fracs))

    return {
        "inflection_k": float(inflection_k),
        "slope_at_inflection": float(slope),
        "auc": auc,
        "rates": {f"{int(k*100)}%": r for k, r in zip(k_fracs, rates)},
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hard negative attribution precision and minimality curve"
    )
    parser.add_argument("--test-path", type=str, default="",
                        help="Path to pre-split test.jsonl (preferred)")
    parser.add_argument("--data", default="",
                        help="Fallback: single JSONL file to re-split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",
                        default="./results/attribution_precision.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    # Minimality curve: fine-grained k values
    parser.add_argument("--k-fracs", nargs="+", type=float,
                        default=[0.02, 0.05, 0.08, 0.10, 0.12, 0.15,
                                 0.18, 0.20, 0.25, 0.30])
    parser.add_argument("--methods", nargs="+",
                        default=["guardlens", "random"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")
    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded {model_name}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = GuardLensCollator(tokenizer, config)

    # Load data (pre-split or fallback)
    if args.test_path and os.path.exists(args.test_path):
        print(f"\nLoading test data from {args.test_path}...")
        records = []
        with open(args.test_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        test_idx = list(range(len(records)))
        print(f"  {len(records)} test records")
    else:
        print(f"\nLoading data {args.data}...")
        records = []
        with open(args.data) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        _, _, test_idx = pair_aware_split(records, seed=config.seed)
        print(f"  Test set: {len(test_idx)} samples")

    # ============================================================
    # Analysis 1: Hard negative attribution precision
    # ============================================================
    print("\n" + "=" * 65)
    print("  Analysis 1: Hard Negative Attribution Precision")
    print("=" * 65)

    hn_results = run_hard_negative_analysis(
        model=model,
        records=records,
        test_idx=test_idx,
        collator=collator,
        config=config,
        device=device,
        batch_size=args.batch_size,
    )

    # Print comparison table
    print(f"\n  {'Subset':<22} {'FPAR':>8} {'MeanAttr':>10} {'MaxAttr':>8} {'Entropy':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    subset_order = ["adversarial", "hard_negative", "borderline",
                    "false_pos_trap", "genuine_benign"]
    for name in subset_order:
        if name not in hn_results:
            continue
        d = hn_results[name]
        print(f"  {name:<22} {d['fpar']:>8.4f} {d['mean_attribution']:>10.4f} "
              f"{d['mean_max_attribution']:>8.4f} {d['attribution_entropy']:>8.4f}")

    # Key metric: FPAR gap between hard negatives and genuine benign
    hn_fpar = hn_results.get("hard_negative", {}).get("fpar", None)
    gb_fpar = hn_results.get("genuine_benign", {}).get("fpar", None)
    adv_fpar = hn_results.get("adversarial", {}).get("fpar", None)
    if hn_fpar is not None and gb_fpar is not None:
        print(f"\n  Hard Neg FPAR vs Genuine Benign FPAR: "
              f"{hn_fpar:.4f} vs {gb_fpar:.4f} "
              f"(gap={hn_fpar - gb_fpar:+.4f})")
    if adv_fpar is not None and hn_fpar is not None:
        print(f"  Adversarial FPAR vs Hard Neg FPAR: "
              f"{adv_fpar:.4f} vs {hn_fpar:.4f} "
              f"(gap={adv_fpar - hn_fpar:+.4f})")

    # ============================================================
    # Analysis 2: Minimality sensitivity curve
    # ============================================================
    print("\n" + "=" * 65)
    print("  Analysis 2: Minimality Sensitivity Curve")
    print(f"  k values: {args.k_fracs}")
    print("=" * 65)

    # Import attribution methods from causal_eval
    try:
        from guardlens.evaluation.causal_eval import (
            guardlens_attribution,
            gradient_x_input_attribution,
            integrated_gradients_attribution,
            surface_risk_attribution,
            random_attribution,
        )
        _METHODS = {
            "guardlens": guardlens_attribution,
            "grad_x_input": gradient_x_input_attribution,
            "integrated_gradients": integrated_gradients_attribution,
            "random": random_attribution,
        }
        if tokenizer:
            _METHODS["surface_risk"] = lambda m, b, d: surface_risk_attribution(
                m, b, d, tokenizer=tokenizer
            )
    except ImportError:
        print("  Warning: causal_eval attribution methods not importable. "
              "Using guardlens and random only.")
        _METHODS = {
            "guardlens": get_guardlens_scores,
            "random": lambda m, b, d: get_random_scores(b),
        }

    methods_to_run = {k: v for k, v in _METHODS.items() if k in args.methods}
    if not methods_to_run:
        print("  No valid methods specified. Using guardlens + random.")
        methods_to_run = {
            "guardlens": get_guardlens_scores,
            "random": lambda m, b, d: get_random_scores(b),
        }

    # Test set loader (adversarial only for flip rate measurement)
    adv_test_idx = [
        test_idx[i] for i, r in enumerate([records[j] for j in test_idx])
        if r.get("label") == 1
    ]
    print(f"\n  Adversarial test samples: {len(adv_test_idx)}")

    dataset = GuardLensDataset(records, config)
    test_loader = DataLoader(
        Subset(dataset, adv_test_idx),
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
    )

    print(f"\n  Running minimality curve ({len(methods_to_run)} methods × "
          f"{len(args.k_fracs)} k values)...")
    curves = run_minimality_curve(
        model=model,
        loader=test_loader,
        device=device,
        methods=methods_to_run,
        k_fracs=args.k_fracs,
    )

    # Print curve table
    k_strs = [f"{int(k*100)}%" for k in args.k_fracs]
    print(f"\n  Flip rate curve:")
    header = f"  {'Method':<25}" + "".join(f" {k:>7}" for k in k_strs)
    print(header)
    print(f"  {'-'*25}" + "".join(" -------" for _ in k_strs))
    for method in args.methods:
        if method not in curves:
            continue
        row = f"  {method:<25}" + "".join(
            f" {curves[method].get(k, 0):>7.3f}" for k in k_strs
        )
        print(row)

    # Compute sharpness for each method
    print(f"\n  Sharpness analysis:")
    sharpness = {}
    print(f"  {'Method':<25} {'InflectionK':>12} {'Slope':>8} {'AUC':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*8}")
    for method in args.methods:
        if method not in curves:
            continue
        s = compute_sharpness(curves[method], args.k_fracs)
        sharpness[method] = s
        print(f"  {method:<25} {s['inflection_k']:>12.3f} "
              f"{s['slope_at_inflection']:>8.3f} {s['auc']:>8.4f}")

    print("\n  Interpretation:")
    print("    Inflection K: smaller = trigger identified with fewer tokens")
    print("    Slope at inflection: larger = sharper, more near-minimal trigger")
    print("    AUC: smaller = flip achieved with fewer tokens (more compact)")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        "hard_negative_attribution_precision": hn_results,
        "minimality_curve": {
            "curves": curves,
            "sharpness": sharpness,
            "k_fracs": args.k_fracs,
            "methods": args.methods,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()