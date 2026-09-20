#!/usr/bin/env python3
"""One-batch GPU smoke for the redesigned joint causal-localization path."""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensCollator, GuardLensDataset
from guardlens.data.training_contract import (
    classification_loss_weight,
    is_auxiliary_detection_record,
    training_label,
)
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.trainer import train_epoch


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def footprint(record):
    turns = record.get("turns", [])
    texts = [str(t.get("text", "")) for t in turns]
    return (
        len(turns),
        sum(len(text) for text in texts),
        max([len(text) for text in texts] or [0]),
    )


def select_smoke_records(records, batch_size):
    primary = [
        r for r in records if not is_auxiliary_detection_record(r)
    ]
    localizable_positive = [
        r for r in primary
        if training_label(r) == 1
        and (
            r.get("evidence_turn_ids")
            or str(r.get("supervision_tier", "")) in {"cf_strong", "cf_weak"}
        )
    ]
    negatives = [r for r in primary if training_label(r) == 0]
    if not localizable_positive or not negatives:
        raise RuntimeError(
            "joint smoke requires a localizable malicious record and a benign record"
        )

    selected = [
        max(localizable_positive, key=footprint),
        max(negatives, key=footprint),
    ]
    if batch_size > 2:
        used = {str(r.get("conversation_id", "")) for r in selected}
        remaining = [
            r for r in sorted(primary, key=footprint, reverse=True)
            if str(r.get("conversation_id", "")) not in used
        ]
        selected.extend(remaining[: batch_size - 2])
    return selected[:batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--backbone", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-turns", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training smoke")
    if args.batch_size < 2:
        raise ValueError("batch-size must be >=2")

    records = load_jsonl(args.train)
    selected = select_smoke_records(records, args.batch_size)

    config = GuardLensConfig(
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        gradient_accumulation=8,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens,
        max_epochs=20,
        phase1_epochs=5,
        seed=args.seed,
        device="cuda",
        num_workers=0,
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
    batch = next(iter(loader))
    if int((batch["turn_labels"] == 1).sum()) == 0:
        raise RuntimeError("smoke batch has no positive evidence-turn target")
    if int((batch["token_labels"] == 1).sum()) == 0:
        raise RuntimeError("smoke batch has no positive causal-span target")

    model = MODEL_REGISTRY["guardlens"](config)
    model.setup_backbone()
    model = model.to("cuda")

    pos_mass = sum(
        classification_loss_weight(r)
        for r in selected if training_label(r) == 1
    )
    neg_mass = sum(
        classification_loss_weight(r)
        for r in selected if training_label(r) == 0
    )
    loss_fn = GuardLensLoss(config)
    loss_fn.set_pos_weight(neg_mass / max(1e-8, pos_mass))

    turn_pos_mass = float(
        batch["turn_weights"][batch["turn_labels"] == 1].sum()
    )
    turn_neg_mass = float(
        batch["turn_weights"][batch["turn_labels"] == 0].sum()
    )
    if turn_pos_mass > 0 and turn_neg_mass > 0:
        loss_fn.set_turn_pos_weight(
            turn_neg_mass / turn_pos_mass
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    torch.cuda.reset_peak_memory_stats()
    metrics = train_epoch(
        model,
        loader,
        optimizer,
        None,
        loss_fn,
        config,
        epoch=config.phase1_epochs,
        device=torch.device("cuda"),
    )
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("=== GuardLens causal-localization training smoke ===")
    for record in selected:
        print(
            f"{record.get('conversation_id')} "
            f"label={training_label(record)} "
            f"turns={len(record.get('turns', []))} "
            f"tier={record.get('supervision_tier')}"
        )
    print(
        f"phase={metrics['phase']} loss={metrics['loss']:.6f} "
        f"turn_loss={metrics['turn_loss']:.6f} "
        f"span_loss={metrics['span_loss']:.6f}"
    )
    print(f"peak_cuda_memory_allocated_gib={peak_gib:.2f}")
    print("TRAINING ARCHITECTURE SMOKE PASSED")


if __name__ == "__main__":
    main()
