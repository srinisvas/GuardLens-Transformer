"""Dev-only detector operating-point calibration."""
from __future__ import annotations

import math

from .metrics import binary


def _select(labels, probabilities, candidates, objective, fpr_cap=None):
    scored = []
    for threshold in candidates:
        metrics = binary(labels, probabilities, threshold)
        if fpr_cap is not None and metrics["fpr"] > fpr_cap:
            continue
        if objective == "max_f1":
            key = (metrics["f1"], -metrics["fpr"], metrics["recall"], threshold)
        elif objective == "max_recall_at_fpr":
            key = (metrics["recall"], metrics["precision"], -metrics["fpr"], threshold)
        else:
            raise ValueError("unknown calibration objective")
        scored.append((key, threshold, metrics))
    if not scored:
        raise ValueError("no feasible detector threshold")
    _, threshold, metrics = max(scored, key=lambda row: row[0])
    return {"threshold": threshold, "metrics": metrics}


def calibrate_operating_points(rows, fpr_caps=(0.01, 0.05, 0.10)):
    """Select operating points on canonical B dev and report all-dev metrics."""
    if not rows:
        raise ValueError("calibration requires nonempty dev predictions")
    if any(r.get("source_family") not in {"A", "B"} for r in rows):
        raise ValueError("calibration rows require frozen A/B source families")
    canonical = [r for r in rows if r["source_family"] == "B"]
    if {r["label"] for r in canonical} != {0, 1}:
        raise ValueError("canonical B dev must contain both classes")
    labels = [r["label"] for r in canonical]
    probabilities = [r["probability"] for r in canonical]
    candidates = sorted({float(p) for p in probabilities})
    candidates.append(math.nextafter(max(candidates), math.inf))
    policies = {"max_f1": _select(labels, probabilities, candidates, "max_f1")}
    for cap in fpr_caps:
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 <= cap <= 1:
            raise ValueError("FPR caps must be numeric fractions in [0,1]")
        name = f"max_recall_at_b_fpr_{100 * cap:g}pct"
        policies[name] = _select(labels, probabilities, candidates, "max_recall_at_fpr", float(cap))
    all_labels = [r["label"] for r in rows]
    all_probabilities = [r["probability"] for r in rows]
    for policy in policies.values():
        policy["all_dev_metrics"] = binary(all_labels, all_probabilities, policy["threshold"])
    return {
        "selection_population": "canonical_source_family_B_dev",
        "selection_rule": "dev_only_no_test_access",
        "canonical_n": len(canonical),
        "all_dev_n": len(rows),
        "policies": policies,
    }
