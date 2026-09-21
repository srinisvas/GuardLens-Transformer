"""Fail-closed tokenizer/representation coverage audit for train and dev."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

from guardlens.config import GuardLensConfig
from guardlens.data.causal_targets import span_supervision_target
from guardlens.data.dataset import GuardLensDataset


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit_split(name, path, tokenizer, max_turns, max_tokens):
    records = load_jsonl(path)
    dataset = GuardLensDataset(
        records,
        GuardLensConfig(
            max_turns=max_turns,
            max_tokens_per_turn=max_tokens,
        ),
    )

    token_lengths = []
    errors = []
    over_token_cap = 0
    positive_spans = 0
    negative_spans = 0
    ignored_spans = 0
    positive_spans_beyond_cap = 0
    turn_targets = {"positive": 0, "negative": 0, "ignored": 0}
    max_tokens_seen = 0
    max_turns_seen = 0

    for idx, record in enumerate(records):
        cid = str(record.get("conversation_id", "")) or "<missing>"
        turns = list(record.get("turns", []) or [])
        max_turns_seen = max(max_turns_seen, len(turns))

        structural_errors = []
        if not turns:
            structural_errors.append(f"{cid}: empty realized trajectory")
        if len(turns) > max_turns:
            structural_errors.append(
                f"{cid}: {len(turns)} turns exceed max_turns={max_turns}"
            )
        expected_ids = list(range(len(turns)))
        realized_ids = [turn.get("turn_id") for turn in turns]
        if turns and realized_ids != expected_ids:
            structural_errors.append(
                f"{cid}: realized turn_id values must equal indices 0..{len(turns)-1}"
            )
        for t_idx, turn in enumerate(turns):
            role = str(turn.get("role", "")).lower()
            if role not in {"user", "assistant"}:
                structural_errors.append(
                    f"{cid}: unsupported turn role {role!r} at turn {t_idx}"
                )

        if structural_errors:
            errors.extend(structural_errors)
        else:
            try:
                item = dataset[idx]
            except RuntimeError as exc:
                errors.append(f"{cid}: dataset contract failure: {exc}")
                item = None

            if item is not None:
                for target in item["evidence_turn_labels"]:
                    if target == 1:
                        turn_targets["positive"] += 1
                    elif target == 0:
                        turn_targets["negative"] += 1
                    else:
                        turn_targets["ignored"] += 1

        for t_idx, turn in enumerate(turns):
            text = str(turn.get("text", ""))
            enc = tokenizer(
                text,
                padding=False,
                truncation=False,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            length = len(enc["input_ids"])
            token_lengths.append(length)
            max_tokens_seen = max(max_tokens_seen, length)

            supervised_spans = []
            role = str(turn.get("role", "")).lower()
            for span in turn.get("span_annotations", []) or []:
                target = span_supervision_target(span)
                if target is None:
                    ignored_spans += 1
                    continue

                if role != "user":
                    errors.append(
                        f"{cid}: turn {t_idx} role={role!r} contains "
                        "target-bearing span supervision; causal span "
                        "localization is user-turn-only"
                    )
                    continue

                if target[0] == 1:
                    positive_spans += 1
                    supervised_spans.append(span)
                else:
                    negative_spans += 1

            if length <= max_tokens:
                continue

            over_token_cap += 1
            offsets = list(enc["offset_mapping"])[:max_tokens]
            covered_char = max(
                [int(end) for start, end in offsets if int(end) > int(start)]
                or [0]
            )
            for span in supervised_spans:
                if int(span.get("char_end", 0)) > covered_char:
                    positive_spans_beyond_cap += 1

            errors.append(
                f"{cid}: turn {t_idx} tokenizes to {length} tokens, "
                f"exceeding max_tokens={max_tokens}"
            )

    arr = np.asarray(token_lengths, dtype=np.float64)
    return {
        "split": name,
        "records": len(records),
        "turns": len(token_lengths),
        "max_realized_turns": max_turns_seen,
        "max_tokens_per_turn_seen": max_tokens_seen,
        "p95_tokens_per_turn": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99_tokens_per_turn": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "turns_over_token_cap": over_token_cap,
        "span_target_annotations": {
            "positive": positive_spans,
            "negative": negative_spans,
            "ignored": ignored_spans,
        },
        "turn_targets": turn_targets,
        "positive_spans_beyond_token_cap": positive_spans_beyond_cap,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--backbone", default="answerdotai/ModernBERT-large")
    parser.add_argument(
        "--backbone-revision",
        default="45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
    )
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.max_turns <= 0 or args.max_tokens <= 0:
        raise ValueError("max-turns and max-tokens must be positive")

    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone,
        revision=args.backbone_revision,
        use_fast=True,
    )
    backbone_config = AutoConfig.from_pretrained(
        args.backbone,
        revision=args.backbone_revision,
    )
    backbone_limit = getattr(backbone_config, "max_position_embeddings", None)
    if (
        isinstance(backbone_limit, int)
        and backbone_limit > 0
        and args.max_tokens > backbone_limit
    ):
        raise RuntimeError(
            f"max-tokens={args.max_tokens} exceeds backbone "
            f"max_position_embeddings={backbone_limit}; select a backbone with "
            "native context coverage rather than truncating the turn"
        )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "representation audit requires a fast tokenizer with offset mappings"
        )

    summaries = [
        audit_split(
            "train", args.train, tokenizer, args.max_turns, args.max_tokens
        ),
        audit_split(
            "dev", args.dev, tokenizer, args.max_turns, args.max_tokens
        ),
    ]
    all_errors = [
        error for summary in summaries for error in summary["errors"]
    ]
    payload = {
        "status": "failed" if all_errors else "passed",
        "backbone": args.backbone,
        "backbone_revision": args.backbone_revision,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "backbone_max_position_embeddings": backbone_limit,
        "held_out_test_accessed": False,
        "splits": {
            summary["split"]: {
                k: v for k, v in summary.items() if k != "errors"
            }
            for summary in summaries
        },
        "errors": all_errors,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    print("=== Representation coverage audit ===")
    for summary in summaries:
        print(
            f"{summary['split']}: records={summary['records']} "
            f"turns={summary['turns']} "
            f"max_turns={summary['max_realized_turns']} "
            f"token_p95={summary['p95_tokens_per_turn']:.1f} "
            f"token_p99={summary['p99_tokens_per_turn']:.1f} "
            f"token_max={summary['max_tokens_per_turn_seen']} "
            f"over_cap={summary['turns_over_token_cap']} "
            f"turn_targets={summary['turn_targets']} "
            f"span_targets={summary['span_target_annotations']} "
            f"positive_spans_beyond_cap="
            f"{summary['positive_spans_beyond_token_cap']}"
        )
    print("Held-out test accessed: NO")

    if all_errors:
        print("REPRESENTATION COVERAGE AUDIT FAILED", file=sys.stderr)
        for error in all_errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(all_errors) > 100:
            print(
                f"... {len(all_errors)-100} additional errors",
                file=sys.stderr,
            )
        sys.exit(2)

    print("REPRESENTATION COVERAGE AUDIT PASSED")


if __name__ == "__main__":
    main()
