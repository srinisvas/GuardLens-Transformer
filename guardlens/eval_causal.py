"""
Causal attribution evaluation.

Usage:
    python -m guardlens.eval_causal \
        --data semantic_multiturn_v10_augmented.jsonl \
        --checkpoint ./checkpoints/guardlens/best.pt \
        --output ./results/causal_eval.json \
        --methods guardlens attention grad_x_input random
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.data.splits import pair_aware_split
from guardlens.models import MODEL_REGISTRY
from guardlens.evaluation.causal_eval import (
    run_causal_evaluation,
    print_comparison_table,
)


def main():
    parser = argparse.ArgumentParser(description="Causal attribution evaluation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./causal_eval_results.json")
    parser.add_argument("--methods", nargs="+",
                        default=["guardlens", "attention", "random"])
    parser.add_argument("--top-k", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20])
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
    print(f"Model: {model_name}, epoch {ckpt.get('epoch', '?')}, phase {ckpt.get('phase', '?')}")

    # Load data
    records = []
    with open(args.data, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    _, _, test_idx = pair_aware_split(records, seed=config.seed)
    print(f"Test set: {len(test_idx)} samples")

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

    # Run evaluation
    print(f"\nRunning causal evaluation with methods: {args.methods}")
    print(f"Top-k fractions: {args.top_k}")
    results = run_causal_evaluation(
        model, test_loader, device,
        methods=args.methods,
        top_k_fractions=args.top_k,
        tokenizer=tokenizer,
    )

    # Print comparison table
    print_comparison_table(results)

    # Save results
    # Convert numpy types for JSON serialization
    def make_serializable(obj):
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
        return obj

    import numpy as np
    serializable = make_serializable(results)

    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()