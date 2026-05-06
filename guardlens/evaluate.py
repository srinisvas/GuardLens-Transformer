import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.data.splits import pair_aware_split
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate GuardLens")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")

    # Load data
    records = []
    with open(args.data, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    _, _, test_idx = pair_aware_split(records, seed=config.seed)

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

    # Build model
    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    loss_fn = GuardLensLoss(config)
    results = evaluate(model, test_loader, loss_fn, config, device)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
