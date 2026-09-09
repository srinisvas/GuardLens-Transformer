#!/usr/bin/env python3
"""One-batch GPU smoke for the repaired 48-turn GuardLens training path.

This deliberately exercises phase 3, including attribution and counterfactual
loss, on the longest available primary training records. It is a runtime gate,
not a benchmark and never reads dev/test labels or metrics.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensCollator, GuardLensDataset
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.trainer import train_epoch


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--backbone", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-turns", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training-window smoke")
    if min(args.max_turns, args.max_tokens, args.batch_size) <= 0:
        raise ValueError("max-turns, max-tokens, and batch-size must be positive")

    records = load_jsonl(args.train)
    if not records:
        raise RuntimeError("training split is empty")
    over = [r for r in records if len(r.get("turns", [])) > args.max_turns]
    if over:
        raise RuntimeError(
            f"{len(over)} training records exceed max_turns={args.max_turns}; run model-window audit first"
        )

    # Stable worst-case selection by realized length, then conversation ID.
    selected = sorted(
        records,
        key=lambda r: (-len(r.get("turns", [])), str(r.get("conversation_id", ""))),
    )[: args.batch_size]
    if len(selected) < args.batch_size:
        raise RuntimeError(
            f"need at least batch_size={args.batch_size} training records for smoke"
        )

    config = GuardLensConfig(
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        gradient_accumulation=8,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens,
        seed=args.seed,
        device="cuda",
        num_workers=0,
        use_pivot_head=True,
        oversample_cf=False,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    dataset = GuardLensDataset(selected, config)
    collator = GuardLensCollator(tokenizer, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )

    model_cls = MODEL_REGISTRY["guardlens"]
    model = model_cls(config)
    model.setup_backbone()
    model = model.to("cuda")

    n_pos = sum(int(r.get("label", -1)) == 1 for r in selected)
    n_neg = len(selected) - n_pos
    config.pos_weight = n_neg / max(1, n_pos)
    loss_fn = GuardLensLoss(config)
    loss_fn.set_pos_weight(config.pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    torch.cuda.reset_peak_memory_stats()
    # Epoch 20 is phase 3 under the locked 5/15/5 schedule.
    metrics = train_epoch(
        model,
        loader,
        optimizer,
        None,
        loss_fn,
        config,
        epoch=config.phase1_epochs + config.phase2_epochs,
        device=torch.device("cuda"),
    )
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("=== GuardLens 48-turn training smoke ===")
    print("Selected records:")
    for r in selected:
        print(
            f"  {r.get('conversation_id')} label={r.get('label')} "
            f"turns={len(r.get('turns', []))} source={r.get('corpus_source', 'unknown')}"
        )
    print(f"phase={metrics['phase']} loss={metrics['loss']:.6f}")
    print(f"peak_cuda_memory_allocated_gib={peak_gib:.2f}")
    print("TRAINING WINDOW SMOKE PASSED")


if __name__ == "__main__":
    main()
