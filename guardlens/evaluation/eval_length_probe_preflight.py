"""Dev-only length shortcut preflight for the NAACL repair.

This intentionally does not open the held-out test split. Fit the same simple
length-only logistic probe on train, tune/report on dev, and use it only as a
pre-training shortcut gate. The full eval_length_probe.py remains the final
held-out report after the pipeline is frozen.
"""

from __future__ import annotations

import argparse
import json
import os

from guardlens.evaluation.eval_length_probe import (
    FEATURE_NAMES,
    distribution_summary,
    fit_logistic,
    load_jsonl,
    make_xy,
    metrics,
    predict_proba,
    tune_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()

    train_records = load_jsonl(args.train)
    dev_records = load_jsonl(args.dev)
    x_train, y_train = make_xy(train_records)
    x_dev, y_dev = make_xy(dev_records)
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError("Training split must contain both labels")
    if len(set(y_dev.tolist())) < 2:
        raise RuntimeError("Dev split must contain both labels")

    weights, mean, std = fit_logistic(
        x_train, y_train,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    train_probs = predict_proba(x_train, weights, mean, std)
    dev_probs = predict_proba(x_dev, weights, mean, std)
    threshold = tune_threshold(y_dev, dev_probs)

    result = {
        "probe": "logistic_regression_length_only_preflight",
        "features": FEATURE_NAMES,
        "threshold_selection": "maximize dev F1",
        "train": metrics(y_train, train_probs, threshold),
        "dev": metrics(y_dev, dev_probs, threshold),
        "distribution": {
            "train": distribution_summary(train_records),
            "dev": distribution_summary(dev_records),
        },
        "held_out_test_accessed": False,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)

    print("=== Dev-only length shortcut preflight ===")
    print(
        f"Dev: AUC={result['dev']['roc_auc']:.3f} "
        f"balanced_acc={result['dev']['balanced_accuracy']:.3f} "
        f"F1={result['dev']['f1']:.3f}"
    )
    print("Held-out test accessed: NO")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
