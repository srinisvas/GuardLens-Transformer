"""
Causal attribution evaluation — v11 compatible.

Usage:
    python -m guardlens.eval_causal \
        --test-path splits/test.jsonl \
        --checkpoint ./checkpoints/guardlens/best.pt \
        --output ./results/causal_eval.json \
        --methods guardlens attention grad_x_input random
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from guardlens.evaluation.causal_eval import (
    run_causal_evaluation,
    print_comparison_table,
)
from guardlens.evaluation.eval_utils import (
    load_test_data, add_test_path_args,
    partition_by_supervision_tier, partition_test_set_v11,
    print_subset_summary, results_to_latex_table,
)


def make_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


def run_subset_causal_eval(
    model, records, indices, collator, config, device,
    methods, top_k, tokenizer, batch_size, subset_name,
):
    """Run causal eval on a subset of records."""
    if len(indices) < 3:
        print(f"  Skipping {subset_name} (only {len(indices)} records)")
        return None

    dataset = GuardLensDataset(records, config)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=4,
    )

    print(f"\n  {subset_name} ({len(indices)} records)...")
    results = run_causal_evaluation(
        model, loader, device,
        methods=methods,
        top_k_fractions=top_k,
        tokenizer=tokenizer,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Causal attribution evaluation")
    parser = add_test_path_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./causal_eval_results.json")
    parser.add_argument("--methods", nargs="+",
                        default=["guardlens", "attention", "random"])
    parser.add_argument("--top-k", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tier-eval", action="store_true", default=True,
                        help="Run per-supervision-tier attribution eval")
    parser.add_argument("--transfer-eval", action="store_true", default=True,
                        help="Run per-transfer-tier attribution eval")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")
    print(f"Model: {model_name}, epoch {ckpt.get('epoch', '?')}, phase {ckpt.get('phase', '?')}")

    # Load data
    records, test_idx = load_test_data(
        test_path=args.test_path, data_path=args.data, seed=config.seed,
    )

    # Build model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = GuardLensCollator(tokenizer, config)

    dataset = GuardLensDataset(records, config)
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.workers,
    )

    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---- Full test set eval ----
    print(f"\nRunning causal evaluation with methods: {args.methods}")
    print(f"Top-k fractions: {args.top_k}")
    results = run_causal_evaluation(
        model, test_loader, device,
        methods=args.methods,
        top_k_fractions=args.top_k,
        tokenizer=tokenizer,
    )
    print_comparison_table(results)

    output = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "full_test": make_serializable(results),
    }

    # ---- Per-supervision-tier eval (issue 11) ----
    if args.tier_eval:
        print(f"\n{'='*60}")
        print(f"  Per-supervision-tier attribution evaluation")
        print(f"{'='*60}")

        test_records = [records[i] for i in test_idx]
        tier_subsets = partition_by_supervision_tier(test_records)
        print_subset_summary(tier_subsets, test_records)

        tier_results = {}
        for tier_name, local_indices in tier_subsets.items():
            if tier_name in ("ignore", "boundary_benign"):
                continue
            r = run_subset_causal_eval(
                model, test_records, local_indices, collator, config, device,
                args.methods, args.top_k, tokenizer, args.batch_size, f"tier:{tier_name}",
            )
            if r:
                tier_results[tier_name] = make_serializable(r)

        output["tier_eval"] = tier_results

    # ---- Per-transfer-tier eval (issue 15) ----
    if args.transfer_eval:
        print(f"\n{'='*60}")
        print(f"  Per-transfer-tier attribution evaluation")
        print(f"{'='*60}")

        test_records = [records[i] for i in test_idx]
        v11_subsets = partition_test_set_v11(test_records)
        print_subset_summary(v11_subsets, test_records)

        transfer_results = {}
        for subset_name in ["transfer_success", "target_only", "cross_only",
                            "contextual_pivot", "lexical_pivot", "distributed"]:
            indices = v11_subsets.get(subset_name, [])
            r = run_subset_causal_eval(
                model, test_records, indices, collator, config, device,
                args.methods, args.top_k, tokenizer, args.batch_size, f"transfer:{subset_name}",
            )
            if r:
                transfer_results[subset_name] = make_serializable(r)

        output["transfer_eval"] = transfer_results

    # ---- Generate LaTeX table (issue 16) ----
    latex = results_to_latex_table(results, caption="Causal Attribution Evaluation",
                                  label="tab:causal_eval")
    output["latex_table"] = latex

    latex_path = args.output.replace(".json", ".tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {latex_path}")

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()