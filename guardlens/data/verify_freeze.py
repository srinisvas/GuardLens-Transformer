"""Verify frozen DataGen artifacts by SHA-256 before model training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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

    report = json.load(open(args.report, encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("freeze report itself is not status=passed")

    hashes = report.get("artifact_sha256") or {}
    if args.variant == "primary":
        expected_train = hashes.get("primary_train")
        expected_dev = hashes.get("primary_dev")
        expected_train_n = (
            (report.get("counts") or {})
            .get("primary_splits", {})
            .get("train")
        )
        expected_dev_n = (
            (report.get("counts") or {})
            .get("primary_splits", {})
            .get("dev")
        )
    else:
        expected_train = hashes.get("auxiliary_candidate_train")
        expected_dev = hashes.get("auxiliary_candidate_dev")
        aux = report.get("auxiliary_candidate") or {}
        expected_train_n = aux.get("train_records")
        expected_dev_n = aux.get("dev_records")

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
    if expected_train_n is not None and train_n != int(expected_train_n):
        errors.append(
            f"train count mismatch expected={expected_train_n} actual={train_n}"
        )
    if expected_dev_n is not None and dev_n != int(expected_dev_n):
        errors.append(
            f"dev count mismatch expected={expected_dev_n} actual={dev_n}"
        )

    payload = {
        "status": "failed" if errors else "passed",
        "variant": args.variant,
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
