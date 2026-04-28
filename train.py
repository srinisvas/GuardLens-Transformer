"""
Training entry point.

Usage:
    python -m guardlens.train --data data.jsonl --output ./checkpoints
    python -m guardlens.train --data data.jsonl --output ./checkpoints --model turn_level
"""

import argparse

from guardlens.config import GuardLensConfig
from guardlens.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train GuardLens")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--model", type=str, default="guardlens",
                        choices=["guardlens", "guardlens_no_fusion",
                                 "guardlens_no_cf", "turn_level",
                                 "conversation_deberta"])
    parser.add_argument("--backbone", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = GuardLensConfig(
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        num_workers=args.workers,
    )

    train(config, args.data, args.output, model_name=args.model)


if __name__ == "__main__":
    main()
