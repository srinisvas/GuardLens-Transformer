"""Verify frozen DataGen artifacts by SHA-256 before model training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Tuple


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _required_field(report: Dict[str, Any], path: str) -> Any:
    value: Any = report
    traversed = []
    for key in path.split("."):
        traversed.append(key)
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(
                "freeze report missing required field: " + ".".join(traversed)
            )
        value = value[key]
    return value


def _required_sha256(report: Dict[str, Any], path: str) -> str:
    value = _required_field(report, path)
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise RuntimeError(
            f"freeze report field {path} must be a lowercase SHA-256 hex digest"
        )
    return value


def _required_count(report: Dict[str, Any], path: str) -> int:
    value = _required_field(report, path)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"freeze report field {path} must be a non-negative integer"
        )
    return value


def expected_artifacts(
    report: Dict[str, Any], variant: str
) -> Tuple[str, str, int, int]:
    """Read the exact DataGen freeze-report contract for a training variant."""
    if report.get("status") != "passed":
        raise RuntimeError("freeze report itself is not status=passed")

    if variant == "primary":
        train_hash_path = "artifact_sha256.primary_train"
        dev_hash_path = "artifact_sha256.primary_dev"
        train_count_path = "counts.primary_splits.train"
        dev_count_path = "counts.primary_splits.dev"
    elif variant == "primary_plus_auxiliary":
        # The restored-A builder deliberately retained these stable consumer
        # keys even though the training view now includes both A and B aux data.
        train_hash_path = "artifact_sha256.auxiliary_candidate_train"
        dev_hash_path = "artifact_sha256.auxiliary_candidate_dev"
        train_count_path = "auxiliary_candidate.train_records"
        dev_count_path = "auxiliary_candidate.dev_records"
    else:
        raise RuntimeError(f"unsupported freeze variant: {variant!r}")

    return (
        _required_sha256(report, train_hash_path),
        _required_sha256(report, dev_hash_path),
        _required_count(report, train_count_path),
        _required_count(report, dev_count_path),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument(
        "--variant",
        choices=["primary", "primary_plus_auxiliary"],
        default="primary_plus_auxiliary",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    for path in (args.report, args.train, args.dev):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    (
        expected_train,
        expected_dev,
        expected_train_n,
        expected_dev_n,
    ) = expected_artifacts(report, args.variant)

    actual_train = sha256(args.train)
    actual_dev = sha256(args.dev)
    train_n = count_jsonl(args.train)
    dev_n = count_jsonl(args.dev)

    errors = []
    if actual_train != expected_train:
        errors.append(
            f"train SHA mismatch expected={expected_train} actual={actual_train}"
        )
    if actual_dev != expected_dev:
        errors.append(
            f"dev SHA mismatch expected={expected_dev} actual={actual_dev}"
        )
    if train_n != expected_train_n:
        errors.append(
            f"train count mismatch expected={expected_train_n} actual={train_n}"
        )
    if dev_n != expected_dev_n:
        errors.append(
            f"dev count mismatch expected={expected_dev_n} actual={dev_n}"
        )

    payload = {
        "status": "failed" if errors else "passed",
        "variant": args.variant,
        "report": {
            "path": args.report,
            "expected_train_sha256": expected_train,
            "expected_dev_sha256": expected_dev,
            "expected_train_records": expected_train_n,
            "expected_dev_records": expected_dev_n,
        },
        "train": {
            "path": args.train,
            "sha256": actual_train,
            "records": train_n,
        },
        "dev": {
            "path": args.dev,
            "sha256": actual_dev,
            "records": dev_n,
        },
        "errors": errors,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    if errors:
        print("FROZEN DATA VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

    print("FROZEN DATA VERIFICATION PASSED")
    print(
        f"variant={args.variant} train={train_n} dev={dev_n} "
        f"train_sha={actual_train} dev_sha={actual_dev}"
    )


if __name__ == "__main__":
    main()
