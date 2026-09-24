"""NAACL causal-localization training entry point.

Only frozen train/dev files are accepted. Test evaluation is a separate stage.
"""
import argparse

from guardlens.config import GuardLensConfig
from guardlens.training.trainer import train


def main():
    parser = argparse.ArgumentParser(
        description="Train redesigned GuardLens on frozen train/dev partitions"
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--dev-path", required=True)
    parser.add_argument(
        "--train-variant",
        choices=["primary", "primary_plus_auxiliary"],
        default="primary_plus_auxiliary",
        help="Frozen training population contract. The canonical run includes auxiliaries.",
    )
    parser.add_argument("--output", default="./checkpoints")
    parser.add_argument(
        "--model",
        default="guardlens",
        choices=["guardlens"],
        help="Only the redesigned GuardLens is enabled until baseline migration.",
    )
    parser.add_argument("--backbone", default="answerdotai/ModernBERT-large")
    parser.add_argument(
        "--backbone-revision",
        default="45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
    )
    parser.add_argument("--backbone-turn-microbatch", type=int, default=8)
    parser.add_argument("--backbone-trainable-layers", type=int, default=0)
    parser.add_argument(
        "--architecture-mode",
        choices=["hierarchical_turn", "cross_token"],
        default="hierarchical_turn",
    )
    parser.add_argument(
        "--attribution-fusion",
        action="store_true",
        help="feed attribution-weighted cross-token states to detection",
    )
    parser.add_argument(
        "--turn-pooling", choices=["mean", "attention"], default="attention"
    )
    parser.add_argument(
        "--input-view", choices=["pre_response", "retrospective"],
        default="retrospective",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--localization-ramp-epochs", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-threshold-tune", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume exactly from OUTPUT/last.pt after validating code, data, runtime, and config",
    )

    args = parser.parse_args()
    for name, value in [
        ("batch-size", args.batch_size),
        ("grad-accumulation", args.grad_accumulation),
        ("epochs", args.epochs),
        ("backbone-turn-microbatch", args.backbone_turn_microbatch),
        ("localization-ramp-epochs", args.localization_ramp_epochs),
        ("max-turns", args.max_turns),
        ("max-tokens", args.max_tokens),
    ]:
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.phase1_epochs < 0 or args.phase1_epochs >= args.epochs:
        parser.error("--phase1-epochs must be >=0 and smaller than --epochs")
    if args.backbone_trainable_layers < 0:
        parser.error("--backbone-trainable-layers must be nonnegative")
    if args.attribution_fusion and args.architecture_mode != "cross_token":
        parser.error("--attribution-fusion requires --architecture-mode cross_token")
    if args.lr <= 0 or args.backbone_lr <= 0:
        parser.error("--lr and --backbone-lr must be positive")

    config = GuardLensConfig(
        backbone_name=args.backbone,
        backbone_revision=args.backbone_revision,
        backbone_turn_microbatch=args.backbone_turn_microbatch,
        backbone_trainable_layers=args.backbone_trainable_layers,
        architecture_mode=args.architecture_mode,
        use_attribution_fusion=args.attribution_fusion,
        turn_pooling=args.turn_pooling,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accumulation,
        learning_rate=args.lr,
        backbone_learning_rate=args.backbone_lr,
        max_epochs=args.epochs,
        phase1_epochs=args.phase1_epochs,
        localization_ramp_epochs=args.localization_ramp_epochs,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens,
        seed=args.seed,
        device=args.device,
        num_workers=args.workers,
        train_path=args.train_path,
        dev_path=args.dev_path,
        train_variant=args.train_variant,
        input_view=args.input_view,
        tune_threshold=not args.no_threshold_tune,
    )
    train(config, args.output, model_name=args.model, resume=args.resume)


if __name__ == "__main__":
    main()
