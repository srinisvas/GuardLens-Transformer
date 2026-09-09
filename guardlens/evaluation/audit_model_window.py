#!/usr/bin/env python3
"""Fail-closed audit for GuardLens realized-turn/model-window compatibility.

The hierarchical collator indexes pivots directly by ``pivot_turn_id``. The
repaired NAACL corpora therefore require realized turn IDs to be exactly
0..N-1 and every primary record to fit wholly inside ``max_turns``. This audit
runs before training so conversation labels, attribution targets, and pivots
cannot be silently truncated.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def audit_split(name: str, path: str, max_turns: int):
    records = load_jsonl(path)
    errors = []
    lengths = Counter()
    sources = Counter()
    max_seen = 0
    over_window = 0
    supported_pivots = 0
    ignored_malicious_pivots = 0

    for record in records:
        cid = str(record.get("conversation_id", "")) or "<missing>"
        turns = list(record.get("turns", []))
        n_turns = len(turns)
        lengths[n_turns] += 1
        max_seen = max(max_seen, n_turns)
        sources[str(record.get("corpus_source", "unknown"))] += 1

        if n_turns == 0:
            errors.append(f"{cid}: empty realized trajectory")
            continue
        if n_turns > max_turns:
            over_window += 1
            errors.append(
                f"{cid}: realized turns={n_turns} exceed model max_turns={max_turns}"
            )

        ids = []
        for idx, turn in enumerate(turns):
            raw = turn.get("turn_id")
            if not isinstance(raw, int) or isinstance(raw, bool):
                errors.append(f"{cid}: non-integer turn_id at realized index {idx}: {raw!r}")
                continue
            ids.append(raw)
        if len(ids) == n_turns and ids != list(range(n_turns)):
            errors.append(
                f"{cid}: realized turn IDs must equal indices 0..{n_turns-1}; got {ids[:12]}"
            )

        label = record.get("label")
        pivot = record.get("pivot_turn_id")
        ignore = bool(record.get("pivot_supervision_ignore", False))
        if label == 1 and pivot is None and not ignore:
            errors.append(
                f"{cid}: malicious record with unknown pivot must set pivot_supervision_ignore=true"
            )
        if label == 1 and pivot is None and ignore:
            ignored_malicious_pivots += 1

        if pivot is not None:
            if ignore:
                errors.append(f"{cid}: non-null pivot is simultaneously marked ignored")
            if not isinstance(pivot, int) or isinstance(pivot, bool):
                errors.append(f"{cid}: pivot_turn_id is non-integer: {pivot!r}")
            elif pivot < 0 or pivot >= n_turns:
                errors.append(f"{cid}: pivot_turn_id={pivot} is outside realized trajectory length {n_turns}")
            elif pivot >= max_turns:
                errors.append(f"{cid}: pivot_turn_id={pivot} is outside model window {max_turns}")
            else:
                if ids and ids[pivot] != pivot:
                    errors.append(
                        f"{cid}: pivot index semantics broken; turns[{pivot}].turn_id={ids[pivot]}"
                    )
                supported_pivots += 1

    summary = {
        "split": name,
        "records": len(records),
        "max_realized_turns": max_seen,
        "over_window": over_window,
        "sources": dict(sources),
        "length_histogram": dict(sorted(lengths.items())),
        "supported_pivots": supported_pivots,
        "ignored_malicious_unknown_pivots": ignored_malicious_pivots,
        "errors": errors,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--max-turns", type=int, default=48)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.max_turns <= 0:
        raise ValueError("max-turns must be positive")

    summaries = [
        audit_split("train", args.train, args.max_turns),
        audit_split("dev", args.dev, args.max_turns),
        audit_split("test", args.test, args.max_turns),
    ]
    payload = {
        "max_turns": args.max_turns,
        "splits": {item["split"]: {k: v for k, v in item.items() if k != "errors"} for item in summaries},
        "status": "passed" if not any(item["errors"] for item in summaries) else "failed",
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    print("=== GuardLens model-window audit ===")
    for item in summaries:
        print(
            f"{item['split']}: n={item['records']} max_turns_seen={item['max_realized_turns']} "
            f"over_window={item['over_window']} supported_pivots={item['supported_pivots']}"
        )
    errors = [error for item in summaries for error in item["errors"]]
    if errors:
        print("MODEL WINDOW AUDIT FAILED", file=sys.stderr)
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"... {len(errors)-100} additional errors", file=sys.stderr)
        sys.exit(2)
    print("MODEL WINDOW AUDIT PASSED")


if __name__ == "__main__":
    main()
