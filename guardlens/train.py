"""
Training entry point — NAACL validity-repair compatible.

Usage:
    python -m guardlens.train \
        --train-path splits/train.jsonl \
        --dev-path splits/dev.jsonl \
        --test-path splits/test.jsonl \
        --output ./checkpoints
"""

import argparse

from guardlens.config import GuardLensConfig
from guardlens.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train GuardLens")

    # Data paths (pre-split, preferred)
    parser.add_argument("--train-path", type=str, default="")
    parser.add_argument("--dev-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="")

    # Fallback single file
    parser.add_argument("--data", type=str, default="")

    # Model
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--model", type=str, default="guardlens",
                        choices=["guardlens", "guardlens_no_fusion",
                                 "guardlens_no_cf", "turn_level",
                                 "conversation_deberta"])

    # Hyperparameters
    parser.add_argument("--backbone", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-turns", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)

    # Feature flags
    parser.add_argument("--no-pivot-head", action="store_true", default=False)
    parser.add_argument("--no-oversample", action="store_true", default=False)
    parser.add_argument("--no-threshold-tune", action="store_true", default=False)

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.grad_accumulation <= 0:
        parser.error("--grad-accumulation must be positive")
    if args.max_turns <= 0 or args.max_tokens <= 0:
        parser.error("--max-turns and --max-tokens must be positive")

    config = GuardLensConfig(
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accumulation,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens,
        seed=args.seed,
        device=args.device,
        num_workers=args.workers,
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        use_pivot_head=not args.no_pivot_head,
        oversample_cf=not args.no_oversample,
        tune_threshold=not args.no_threshold_tune,
    )

    data_path = args.data or ""
    train(config, data_path, args.output, model_name=args.model)


if __name__ == "__main__":
    main()
