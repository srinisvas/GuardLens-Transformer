import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.data.splits import pair_aware_split
from guardlens.models import MODEL_REGISTRY
from guardlens.evaluation.causal_eval import (
    run_causal_evaluation,
    print_comparison_table,
)


def partition_test_set(
    test_records: List[Dict],
    test_idx: List[int],
) -> Dict[str, List[int]]:

    implicit_global = []
    explicit_global = []
    hard_neg_global = []
    benign_global = []

    for local_i, global_i in enumerate(test_idx):
        record = test_records[local_i]
        label = record.get("label", 0)
        family = record.get("family", "")
        turns = record.get("turns", [])

        # Implicit: has at least one implicit_trigger turn
        has_implicit = any(t.get("implicit_trigger") for t in turns)

        # Hard negative: explicitly constructed to look adversarial but benign
        is_hard_neg = family in ("hard_negative", "borderline_benign", "false_positive_trap")

        if label == 1 and has_implicit:
            implicit_global.append(global_i)
        elif label == 1 and not has_implicit:
            explicit_global.append(global_i)
        elif label == 0 and is_hard_neg:
            hard_neg_global.append(global_i)
        elif label == 0:
            benign_global.append(global_i)

    return {
        "implicit": implicit_global,
        "explicit": explicit_global,
        "hard_negative": hard_neg_global,
        "benign": benign_global,
    }


def compute_surface_risk_score(text: str) -> float:

    _SR_HIGH = {
        "improve success rate": 0.80, "less detectable": 0.75,
        "blend in": 0.65, "more convincing": 0.55,
        "pass casual scrutiny": 0.70, "reduce suspicion": 0.65,
        "bypass": 0.60, "exploit": 0.55, "override": 0.55,
        "credential harvesting": 0.60, "jailbreak": 0.65,
        "prompt injection": 0.55, "payload": 0.50,
    }
    txt = text.lower()
    scores = sorted([w for p, w in _SR_HIGH.items() if p in txt], reverse=True)
    if not scores:
        return 0.0
    score = scores[0]
    for w in scores[1:]:
        score += w * 0.3
    return round(min(1.0, score), 3)


def verify_subset_properties(
    test_records: List[Dict],
    subsets: Dict[str, List[int]],
    all_records: List[Dict],
):

    record_by_idx = {i: all_records[i] for subset in subsets.values() for i in subset}

    print("\n  Subset verification:")
    for subset_name, global_indices in subsets.items():
        if not global_indices:
            print(f"    {subset_name}: EMPTY")
            continue

        records_in_subset = [all_records[i] for i in global_indices]
        avg_sr = []
        for record in records_in_subset:
            user_turns = [t for t in record.get("turns", []) if t["role"] == "user"]
            for t in user_turns:
                avg_sr.append(compute_surface_risk_score(t["text"]))

        implicit_turns = sum(
            1 for r in records_in_subset
            for t in r.get("turns", [])
            if t.get("implicit_trigger")
        )

        print(f"    {subset_name:<20}: n={len(global_indices):4d}  "
              f"mean_SR={sum(avg_sr)/max(1,len(avg_sr)):.3f}  "
              f"implicit_turns={implicit_turns}")


def run_subset_eval(
    model: torch.nn.Module,
    global_indices: List[int],
    all_records: List[Dict],
    collator: GuardLensCollator,
    config: GuardLensConfig,
    device: torch.device,
    top_k_fractions: List[float],
    batch_size: int,
    methods: List[str],
    tokenizer,
    subset_name: str,
) -> Dict:
    if not global_indices:
        return {"error": "empty subset"}

    dataset = GuardLensDataset(all_records, config)
    loader = DataLoader(
        Subset(dataset, global_indices),
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\n  Running {subset_name} subset ({len(global_indices)} samples)...")
    results = run_causal_evaluation(
        model, loader, device,
        methods=methods,
        top_k_fractions=top_k_fractions,
        tokenizer=tokenizer,
    )
    return results


def print_subset_comparison(
    all_results: Dict[str, Dict],
    top_k_fractions: List[float],
    focus_k: float = 0.15,
):

    k_str = f"{int(focus_k*100)}%"
    methods = list(next(iter(all_results.values())).keys()) if all_results else []

    print(f"\n{'='*80}")
    print(f"  Subset Analysis at top-{k_str} token removal")
    print(f"{'='*80}")

    for subset_name, subset_results in all_results.items():
        print(f"\n  Subset: {subset_name.upper()}")
        print(f"  {'Method':<25} {'DevDrop':>10} {'Flip':>10} {'Nec':>10} {'Suf':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for method, data in subset_results.items():
            dd = data.get("deviation_drops", {}).get(k_str, 0)
            flip = data.get("flip_rates", {}).get(f"flip@{k_str}", 0)
            nec = data.get("necessity", {}).get(k_str, 0)
            suf = data.get("sufficiency", {}).get(k_str, 0)
            print(f"  {method:<25} {dd:>10.3f} {flip:>10.3f} {nec:>10.3f} {suf:>10.3f}")

    # Key contrast: GuardLens vs Surface Risk on implicit vs explicit
    if "implicit" in all_results and "explicit" in all_results:
        print(f"\n  {'='*60}")
        print(f"  KEY CONTRAST: GuardLens vs Surface Risk")
        print(f"  {'='*60}")
        print(f"  {'Subset':<20} {'GuardLens DD':>15} {'SurfRisk DD':>15} {'Delta':>10}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")

        for subset_name in ["implicit", "explicit", "hard_negative"]:
            if subset_name not in all_results:
                continue
            sr = all_results[subset_name]
            gl_dd = sr.get("guardlens", {}).get("deviation_drops", {}).get(k_str, None)
            surf_dd = sr.get("surface_risk", {}).get("deviation_drops", {}).get(k_str, None)
            if gl_dd is not None and surf_dd is not None:
                delta = gl_dd - surf_dd
                flag = " ← KEY RESULT" if subset_name == "implicit" else ""
                print(f"  {subset_name:<20} {gl_dd:>15.3f} {surf_dd:>15.3f} {delta:>+10.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Implicit vs explicit trigger analysis")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="./results/implicit_explicit_eval.json")
    parser.add_argument("--methods", nargs="+",
                        default=["guardlens", "surface_risk", "integrated_gradients",
                                 "grad_x_input", "attention", "random"])
    parser.add_argument("--top-k", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
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

    # Load data
    print(f"\nLoading data from {args.data}...")
    records = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"  {len(records)} records")

    _, _, test_idx = pair_aware_split(records, seed=config.seed)
    test_records = [records[i] for i in test_idx]
    print(f"  Test set: {len(test_idx)} samples")

    # Partition
    subsets = partition_test_set(test_records, test_idx)
    for name, indices in subsets.items():
        print(f"  Subset '{name}': {len(indices)} samples")

    # Verify subset properties
    verify_subset_properties(test_records, subsets, records)

    # Run evaluation on each subset
    all_results = {}
    # Order: implicit first (the key result), then explicit, then others
    eval_order = ["implicit", "explicit", "hard_negative", "benign"]
    for subset_name in eval_order:
        global_indices = subsets.get(subset_name, [])
        if len(global_indices) < 5:
            print(f"\n  Skipping '{subset_name}' (only {len(global_indices)} samples)")
            continue

        results = run_subset_eval(
            model=model,
            global_indices=global_indices,
            all_records=records,
            collator=collator,
            config=config,
            device=device,
            top_k_fractions=args.top_k,
            batch_size=args.batch_size,
            methods=args.methods,
            tokenizer=tokenizer,
            subset_name=subset_name,
        )
        all_results[subset_name] = results

        # Print intermediate table
        print_comparison_table(results)

    # Print combined comparison
    print_subset_comparison(all_results, args.top_k, focus_k=0.15)

    # Save
    def make_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'item'):
            return obj.item()
        return str(obj)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
        "methods": args.methods,
        "top_k_fractions": args.top_k,
        "results": make_serializable(all_results),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
