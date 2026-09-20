"""Evaluation entry point for the causal-localization redesign.

The legacy evaluation implementation is intentionally disabled until the full
NAACL evaluator migration is complete. This avoids silently applying EMNLP-era
pivot/construction semantics to redesigned checkpoints.
"""

import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser(
        description="GuardLens evaluation migration gate"
    )
    parser.add_argument("--checkpoint", required=True)
    args, _ = parser.parse_known_args()

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    architecture = ckpt.get("architecture_version", "legacy")
    if architecture == "causal_localization_v1":
        raise SystemExit(
            "Evaluation for causal_localization_v1 is not yet migrated on this "
            "branch. Do not use legacy evaluators. Complete the NAACL evaluation "
            "migration first."
        )
    raise SystemExit(
        "This branch no longer supports the legacy evaluation entry point. "
        "Use the naacl-validity-repair branch for historical checkpoints."
    )


if __name__ == "__main__":
    main()
