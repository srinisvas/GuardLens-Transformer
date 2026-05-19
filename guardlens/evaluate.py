"""
Evaluation entry point — v11 compatible.

Usage:
    python -m guardlens.evaluate \
        --test-path splits/test.jsonl \
        --checkpoint ./checkpoints/best.pt \
        --output results/classification.json
"""

import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader, Subset

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.trainer import evaluate
from guardlens.evaluation.eval_utils import load_test_data, add_test_path_args


def main():
    parser = argparse.ArgumentParser(description="Evaluate GuardLens")
    parser = add_test_path_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON path (if empty, prints to stdout)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")
    threshold = ckpt.get("threshold", 0.5)
    print(f"Model: {model_name}, threshold: {threshold:.2f}", file=sys.stderr)

    # Disable features for baselines
    if model_name in ("turn_level", "conversation_deberta"):
        config.use_pivot_head = False

    # Load data
    records, test_idx = load_test_data(
        test_path=args.test_path, data_path=args.data, seed=config.seed,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    if model_name == "conversation_deberta":
        from guardlens.data.dataset import FlatConversationCollator
        collator = FlatConversationCollator(tokenizer, config)
    else:
        collator = GuardLensCollator(tokenizer, config)

    dataset = GuardLensDataset(records, config)
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.workers,
    )

    # Build model
    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    loss_fn = GuardLensLoss(config)
    results = evaluate(model, test_loader, loss_fn, config, device, threshold=threshold)

    payload = {k: v for k, v in results.items() if not isinstance(v, torch.Tensor)}

    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()