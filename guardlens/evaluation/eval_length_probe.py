"""Length-only shortcut probe for the NAACL GuardLens validity repair.

The v11 audit found a large malicious/benign conversation-length mismatch. This
script asks a deliberately simple question: how well can a linear classifier
predict the label using only conversation-size features?

No scikit-learn dependency is required. The probe is logistic regression fit
with NumPy, the decision threshold is tuned on dev F1, and all reported test
metrics are held out.

Usage
-----
python -m guardlens.evaluation.eval_length_probe \
    --train splits/train.jsonl \
    --dev splits/dev.jsonl \
    --test splits/test.jsonl \
    --output results/length_probe.json

For the repaired dataset, performance should collapse toward chance. If this
probe remains strong, do not treat the repaired classification result as clean.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np


FEATURE_NAMES = [
    "n_user_turns",
    "n_assistant_turns",
    "n_total_turns",
    "total_user_chars",
    "mean_user_chars",
]


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_features(record: Dict) -> np.ndarray:
    turns = record.get("turns", [])
    user_texts = [
        str(t.get("text", ""))
        for t in turns
        if str(t.get("role", "")).lower() == "user"
    ]
    assistant_count = sum(
        1 for t in turns if str(t.get("role", "")).lower() == "assistant"
    )
    user_chars = sum(len(text) for text in user_texts)
    mean_user_chars = user_chars / max(1, len(user_texts))
    return np.asarray(
        [
            len(user_texts),
            assistant_count,
            len(turns),
            user_chars,
            mean_user_chars,
        ],
        dtype=np.float64,
    )


def make_xy(records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.stack([extract_features(r) for r in records], axis=0)
    y = np.asarray([int(r.get("label", 0)) for r in records], dtype=np.float64)
    return x, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 5000,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    design = np.concatenate([np.ones((len(xs), 1)), xs], axis=1)

    weights = np.zeros(design.shape[1], dtype=np.float64)
    n = max(1, len(y))

    for _ in range(steps):
        probs = sigmoid(design @ weights)
        grad = (design.T @ (probs - y)) / n
        grad[1:] += l2 * weights[1:]
        weights -= learning_rate * grad

    return weights, mean, std


def predict_proba(
    x: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    xs = (x - mean) / std
    design = np.concatenate([np.ones((len(xs), 1)), xs], axis=1)
    return sigmoid(design @ weights)


def confusion(y: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, int]:
    pred = (probs >= threshold).astype(np.int64)
    yt = y.astype(np.int64)
    return {
        "tp": int(np.sum((pred == 1) & (yt == 1))),
        "tn": int(np.sum((pred == 0) & (yt == 0))),
        "fp": int(np.sum((pred == 1) & (yt == 0))),
        "fn": int(np.sum((pred == 0) & (yt == 1))),
    }


def metrics(y: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    c = confusion(y, probs, threshold)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    balanced_accuracy = 0.5 * (
        tp / max(1, tp + fn) + tn / max(1, tn + fp)
    )
    return {
        **c,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc(y, probs)),
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
    }


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC AUC with average ranks for ties."""
    y = y.astype(np.int64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.zeros(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def tune_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        f1 = metrics(y, probs, float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def distribution_summary(records: Sequence[Dict]) -> Dict:
    x, y = make_xy(records)
    out: Dict = {"n": int(len(records))}
    for label_value, label_name in [(0, "benign"), (1, "malicious")]:
        mask = y == label_value
        if not np.any(mask):
            out[label_name] = {"n": 0}
            continue
        xm = x[mask]
        out[label_name] = {
            "n": int(np.sum(mask)),
            "mean": {
                name: float(xm[:, i].mean()) for i, name in enumerate(FEATURE_NAMES)
            },
            "median": {
                name: float(np.median(xm[:, i])) for i, name in enumerate(FEATURE_NAMES)
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()

    train_records = load_jsonl(args.train)
    dev_records = load_jsonl(args.dev)
    test_records = load_jsonl(args.test)

    x_train, y_train = make_xy(train_records)
    x_dev, y_dev = make_xy(dev_records)
    x_test, y_test = make_xy(test_records)

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training split must contain both labels")
    if len(np.unique(y_dev)) < 2:
        raise RuntimeError("Dev split must contain both labels")
    if len(np.unique(y_test)) < 2:
        raise RuntimeError("Test split must contain both labels")

    weights, mean, std = fit_logistic(
        x_train,
        y_train,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    train_probs = predict_proba(x_train, weights, mean, std)
    dev_probs = predict_proba(x_dev, weights, mean, std)
    test_probs = predict_proba(x_test, weights, mean, std)
    threshold = tune_threshold(y_dev, dev_probs)

    coefficients = {
        "intercept": float(weights[0]),
        **{
            name: float(weights[i + 1])
            for i, name in enumerate(FEATURE_NAMES)
        },
    }

    result = {
        "probe": "logistic_regression_length_only",
        "features": FEATURE_NAMES,
        "fit": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "coefficients_on_standardized_features": coefficients,
        },
        "threshold_selection": "maximize dev F1",
        "train": metrics(y_train, train_probs, threshold),
        "dev": metrics(y_dev, dev_probs, threshold),
        "test": metrics(y_test, test_probs, threshold),
        "distribution": {
            "train": distribution_summary(train_records),
            "dev": distribution_summary(dev_records),
            "test": distribution_summary(test_records),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)

    print("=== Length-only shortcut probe ===")
    print(f"Threshold (dev): {threshold:.3f}")
    print(
        f"Test: acc={result['test']['accuracy']:.3f} "
        f"bal_acc={result['test']['balanced_accuracy']:.3f} "
        f"F1={result['test']['f1']:.3f} "
        f"AUC={result['test']['roc_auc']:.3f}"
    )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
