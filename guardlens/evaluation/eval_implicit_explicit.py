"""
guardlens/evaluation/eval_implicit_explicit.py

Implicit vs explicit trigger subset analysis — v11 compatible.

v11 changes:
  - Uses pivot_kind for implicit/explicit partitioning:
    contextual_pivot / distributed → implicit (context-dependent attacks)
    lexical_pivot → explicit (keyword-based attacks)
  - Uses benign_status for hard negative partitioning:
    validated_benign_twin, hard_benign, false_lead_benign → hard negative
    clean_benign → clean benign
  - Loads pre-split test data instead of re-splitting

Usage:
    python -m guardlens.evaluation.eval_implicit_explicit \
        --test-path splits/test.jsonl \
        --checkpoint checkpoints/guardlens/best.pt \
        --output results/implicit_explicit_eval.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from guardlens.evaluation.causal_eval import (
    run_causal_evaluation,
    print_comparison_table,
)
from guardlens.evaluation.eval_utils import (
    load_test_data, add_test_path_args,
    partition_test_set_v11, print_subset_summary,
    comparison_to_latex,
)


def print_subset_comparison(
    all_results: Dict[str, Dict],
    top_k_fractions,
    focus_k: float = 0.15,
):
    """Print table comparing methods across subsets."""
    k_str = f"{int(focus_k*100)}%"

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

    # Key contrast: GuardLens vs Surface Risk on contextual vs lexical pivots
    ctx_key = "contextual_pivot"
    lex_key = "lexical_pivot"
    if ctx_key in all_results and lex_key in all_results:
        print(f"\n  {'='*60}")
        print(f"  KEY CONTRAST: GuardLens vs Surface Risk")
        print(f"  {'='*60}")
        print(f"  {'Subset':<25} {'GuardLens DD':>15} {'SurfRisk DD':>15} {'Delta':>10}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")

        for subset_name in [ctx_key, lex_key, "hard_benign", "false_lead"]:
            if subset_name not in all_results:
                continue
            sr = all_results[subset_name]
            gl_dd = sr.get("guardlens", {}).get("deviation_drops", {}).get(k_str, None)
            surf_dd = sr.get("surface_risk", {}).get("deviation_drops", {}).get(k_str, None)
            if gl_dd is not None and surf_dd is not None:
                delta = gl_dd - surf_dd
                flag = " <-- KEY RESULT" if subset_name == ctx_key else ""
                print(f"  {subset_name:<25} {gl_dd:>15.3f} {surf_dd:>15.3f} {delta:>+10.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Implicit vs explicit trigger analysis")
    parser = add_test_path_args(parser)
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
    records, test_idx = load_test_data(
        test_path=args.test_path, data_path=args.data, seed=config.seed,
    )
    test_records = [records[i] for i in test_idx]

    # Partition using v11 fields
    subsets = partition_test_set_v11(test_records)
    print_subset_summary(subsets, test_records)

    # Run evaluation on each subset
    all_results = {}
    eval_order = ["contextual_pivot", "lexical_pivot", "distributed",
                  "hard_benign", "false_lead", "clean_benign"]

    for subset_name in eval_order:
        indices = subsets.get(subset_name, [])
        if len(indices) < 5:
            print(f"\n  Skipping '{subset_name}' (only {len(indices)} samples)")
            continue

        dataset = GuardLensDataset(test_records, config)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )

        print(f"\n  Running {subset_name} ({len(indices)} samples)...")
        results = run_causal_evaluation(
            model, loader, device,
            methods=args.methods,
            top_k_fractions=args.top_k,
            tokenizer=tokenizer,
        )
        all_results[subset_name] = results
        print_comparison_table(results)

    # Print combined comparison
    print_subset_comparison(all_results, args.top_k, focus_k=0.15)

    # Generate LaTeX
    latex = comparison_to_latex(
        all_results,
        caption="Attribution Quality by Trigger Type",
        label="tab:implicit_explicit",
    )

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
        "latex_table": latex,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    latex_path = args.output.replace(".json", ".tex")
    with open(latex_path, "w") as f:
        f.write(latex)

    print(f"\nResults saved to {args.output}")
    print(f"LaTeX saved to {latex_path}")


if __name__ == "__main__":
    main()