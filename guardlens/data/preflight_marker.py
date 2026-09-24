"""Create and verify reusable train/dev preflight attestations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def create(args):
    length = json.loads(Path(args.length_report).read_text(encoding="utf-8"))
    auc = float(length["dev"]["roc_auc"])
    if length.get("input_view") != args.input_view:
        raise RuntimeError("length report input view differs from preflight")
    if auc > args.length_auc_ceiling:
        raise RuntimeError(
            f"dev length-only AUC {auc:.6f} exceeds ceiling "
            f"{args.length_auc_ceiling:.6f}"
        )
    paths = {
        "train": args.train,
        "dev": args.dev,
        "freeze_report": args.freeze_report,
        "representation_report": args.representation_report,
        "length_report": args.length_report,
    }
    for name, path in paths.items():
        if not Path(path).is_file():
            raise RuntimeError(f"missing {name}: {path}")
    write_json(args.output, {
        "version": 1,
        "code_sha": args.code_sha,
        "variant": args.variant,
        "input_view": args.input_view,
        "backbone": args.backbone,
        "backbone_revision": args.backbone_revision,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "length_auc_ceiling": args.length_auc_ceiling,
        "dev_length_auc": auc,
        "files": {
            name: {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
            for name, path in paths.items()
        },
        "held_out_test_accessed": False,
    })


def verify(args):
    marker = json.loads(Path(args.marker).read_text(encoding="utf-8"))
    expected = {
        "version": 1,
        "code_sha": args.code_sha,
        "variant": args.variant,
        "input_view": args.input_view,
        "backbone": args.backbone,
        "backbone_revision": args.backbone_revision,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "held_out_test_accessed": False,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(
                f"preflight marker {key} mismatch: {marker.get(key)!r} != {value!r}"
            )
    for name, path in {"train": args.train, "dev": args.dev}.items():
        recorded = marker.get("files", {}).get(name, {})
        if recorded.get("sha256") != file_hash(path):
            raise RuntimeError(f"preflight marker {name} SHA256 mismatch")
    for name in ("freeze_report", "representation_report", "length_report"):
        recorded = marker.get("files", {}).get(name, {})
        path = recorded.get("path")
        if not path or not Path(path).is_file() or recorded.get("sha256") != file_hash(path):
            raise RuntimeError(f"preflight marker {name} is missing or changed")
    print(
        f"verified shared preflight variant={args.variant} "
        f"view={args.input_view} dev_length_auc={marker['dev_length_auc']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    verify_parser = subparsers.add_parser("verify")
    for current in (create_parser, verify_parser):
        current.add_argument("--variant", choices=["primary", "primary_plus_auxiliary"], required=True)
        current.add_argument("--train", required=True)
        current.add_argument("--dev", required=True)
        current.add_argument("--code-sha", required=True)
        current.add_argument("--input-view", choices=["pre_response", "retrospective"], required=True)
        current.add_argument("--backbone", required=True)
        current.add_argument("--backbone-revision", required=True)
        current.add_argument("--max-turns", type=int, required=True)
        current.add_argument("--max-tokens", type=int, required=True)
    create_parser.add_argument("--freeze-report", required=True)
    create_parser.add_argument("--representation-report", required=True)
    create_parser.add_argument("--length-report", required=True)
    create_parser.add_argument("--length-auc-ceiling", type=float, default=0.650)
    create_parser.add_argument("--output", required=True)
    verify_parser.add_argument("--marker", required=True)
    create_parser.set_defaults(func=create)
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
