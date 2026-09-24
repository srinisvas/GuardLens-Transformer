#!/usr/bin/env python3
"""One-batch GPU smoke for the redesigned joint causal-localization path."""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensCollator, GuardLensDataset
from guardlens.data.causal_targets import span_supervision_target
from guardlens.data.training_contract import (
    classification_loss_weight,
    is_auxiliary_detection_record,
    source_family,
    training_label,
)
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.trainer import train_epoch


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def token_footprint(record, tokenizer):
    lengths = [
        len(tokenizer(
            str(turn.get("text", "")),
            truncation=False,
            add_special_tokens=True,
        )["input_ids"])
        for turn in record.get("turns", []) or []
    ]
    max_len = max(lengths or [0])
    # First key approximates the dense post-backbone tensor footprint used by
    # the current hierarchical heads; later keys break ties toward more real
    # tokens and more turns.
    return (
        max_len * len(lengths),
        max_len,
        sum(lengths),
        len(lengths),
    )


def select_smoke_records(records, batch_size, tokenizer):
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
    negatives = []
    for record in primary:
        if training_label(record) != 0:
            continue
        has_negative_span = any(
            (target := span_supervision_target(span)) is not None
            and target[0] == 0
            for turn in record.get("turns", []) or []
            for span in turn.get("span_annotations", []) or []
        )
        if has_negative_span:
            negatives.append(record)

    if not localizable_positive or not negatives:
        raise RuntimeError(
            "joint smoke requires a localizable malicious record and a benign "
            "record with explicit negative span supervision"
        )

    selected = [
        max(localizable_positive, key=lambda r: token_footprint(r, tokenizer)),
        max(negatives, key=lambda r: token_footprint(r, tokenizer)),
    ]
    if batch_size > 2:
        used = {str(r.get("conversation_id", "")) for r in selected}
        remaining = [
            r for r in sorted(primary, key=lambda r: token_footprint(r, tokenizer), reverse=True)
            if str(r.get("conversation_id", "")) not in used
        ]
        selected.extend(remaining[: batch_size - 2])
    return selected[:batch_size]


def select_worst_case_records(records, batch_size, tokenizer):
    return sorted(
        records,
        key=lambda r: token_footprint(r, tokenizer),
        reverse=True,
    )[:batch_size]


def select_auxiliary_records(records, batch_size, tokenizer):
    auxiliary = [r for r in records if is_auxiliary_detection_record(r)]
    by_family = {
        family: [r for r in auxiliary if source_family(r) == family]
        for family in ("A", "B")
    }
    if not all(by_family.values()):
        raise RuntimeError(
            "canonical smoke requires detection-only auxiliaries from A and B"
        )

    selected = [
        max(by_family["A"], key=lambda r: token_footprint(r, tokenizer)),
        max(
            by_family["B"],
            key=lambda r: (
                training_label(r) == 1,
                token_footprint(r, tokenizer),
            ),
        ),
    ]
    if batch_size > 2:
        used = {str(r.get("conversation_id", "")) for r in selected}
        remaining = [
            r for r in sorted(
                auxiliary,
                key=lambda r: token_footprint(r, tokenizer),
                reverse=True,
            )
            if str(r.get("conversation_id", "")) not in used
        ]
        selected.extend(remaining[: batch_size - 2])
    return selected[:batch_size]


def make_loader(records, config, tokenizer):
    dataset = GuardLensDataset(records, config)
    collator = GuardLensCollator(tokenizer, config)
    return DataLoader(
        dataset,
        batch_size=len(records),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument(
        "--train-variant",
        choices=["primary", "primary_plus_auxiliary"],
        default="primary_plus_auxiliary",
    )
    parser.add_argument("--backbone", default="answerdotai/ModernBERT-large")
    parser.add_argument(
        "--backbone-revision",
        default="45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
    )
    parser.add_argument("--backbone-turn-microbatch", type=int, default=8)
    parser.add_argument("--backbone-trainable-layers", type=int, default=0)
    parser.add_argument(
        "--turn-pooling", choices=["mean", "attention"], default="attention"
    )
    parser.add_argument(
        "--input-view", choices=["pre_response", "retrospective"],
        default="pre_response",
    )
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training smoke")
    if args.batch_size < 2:
        raise ValueError("batch-size must be >=2")

    records = load_jsonl(args.train)

    config = GuardLensConfig(
        backbone_name=args.backbone,
        backbone_revision=args.backbone_revision,
        backbone_turn_microbatch=args.backbone_turn_microbatch,
        backbone_trainable_layers=args.backbone_trainable_layers,
        turn_pooling=args.turn_pooling,
        batch_size=args.batch_size,
        gradient_accumulation=8,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens,
        max_epochs=20,
        phase1_epochs=5,
        seed=args.seed,
        device="cuda",
        num_workers=0,
        input_view=args.input_view,
        train_variant=args.train_variant,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.backbone_name,
        revision=config.backbone_revision,
        use_fast=True,
    )
    selected = select_smoke_records(
        records, args.batch_size, tokenizer
    )
    worst_case = select_worst_case_records(
        records, args.batch_size, tokenizer
    )
    auxiliary = (
        select_auxiliary_records(records, args.batch_size, tokenizer)
        if args.train_variant == "primary_plus_auxiliary"
        else []
    )
    loader = make_loader(selected, config, tokenizer)
    worst_loader = make_loader(worst_case, config, tokenizer)
    auxiliary_loader = (
        make_loader(auxiliary, config, tokenizer) if auxiliary else None
    )
    batch = next(iter(loader))
    if int((batch["turn_labels"] == 1).sum()) == 0:
        raise RuntimeError("smoke batch has no positive evidence-turn target")
    if int((batch["token_labels"] == 1).sum()) == 0:
        raise RuntimeError("smoke batch has no positive causal-span target")
    if int((batch["token_labels"] == 0).sum()) == 0:
        raise RuntimeError("smoke batch has no explicit negative span target")
    if auxiliary_loader is not None:
        auxiliary_batch = next(iter(auxiliary_loader))
        if int((auxiliary_batch["turn_labels"] >= 0).sum()) != 0:
            raise RuntimeError("auxiliary smoke batch leaked evidence-turn targets")
        if int((auxiliary_batch["token_labels"] >= 0).sum()) != 0:
            raise RuntimeError("auxiliary smoke batch leaked span targets")

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

    def run_batch(name, records_for_batch, batch_loader):
        torch.cuda.reset_peak_memory_stats()
        metrics = train_epoch(
            model,
            batch_loader,
            optimizer,
            None,
            loss_fn,
            config,
            epoch=config.phase1_epochs,
            device=torch.device("cuda"),
        )
        torch.cuda.synchronize()
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"=== {name} ===")
        for record in records_for_batch:
            footprint = token_footprint(record, tokenizer)
            print(
                f"{record.get('conversation_id')} "
                f"label={training_label(record)} "
                f"turns={len(record.get('turns', []))} "
                f"max_turn_tokens={footprint[1]} "
                f"dense_token_slots={footprint[0]} "
                f"tier={record.get('supervision_tier')} "
                f"source={record.get('corpus_source')} "
                f"detection_weight={classification_loss_weight(record)}"
            )
        print(
            f"phase={metrics['phase']} loss={metrics['loss']:.6f} "
            f"turn_loss={metrics['turn_loss']:.6f} "
            f"span_loss={metrics['span_loss']:.6f}"
        )
        print(f"peak_cuda_memory_allocated_gib={peak_gib:.2f}")
        return peak_gib

    print("=== GuardLens causal-localization training smoke ===")
    joint_peak = run_batch("joint-supervision batch", selected, loader)
    auxiliary_peak = None
    if auxiliary_loader is not None:
        auxiliary_peak = run_batch(
            "auxiliary-detection-only batch", auxiliary, auxiliary_loader
        )
    worst_peak = run_batch("worst-token-footprint batch", worst_case, worst_loader)
    print(f"joint_peak_gib={joint_peak:.2f}")
    if auxiliary_peak is not None:
        print(f"auxiliary_peak_gib={auxiliary_peak:.2f}")
    print(f"worst_case_peak_gib={worst_peak:.2f}")
    print("TRAINING ARCHITECTURE SMOKE PASSED")


if __name__ == "__main__":
    main()
