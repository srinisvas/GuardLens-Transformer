"""Authoritative causal-localization targets shared by training and evaluation.

Only intervention-backed evidence becomes positive localization supervision.
LLM-confirmed or construction-derived annotations without counterfactual support
remain ignored. Explicit negative controls are negative localization targets.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from guardlens.data.training_contract import (
    is_auxiliary_detection_record,
    localization_supervision_ignored,
    training_label,
)

SPAN_POSITIVE_TIERS = {"cf_strong": 1.0, "cf_weak": 0.70}
TURN_POSITIVE_STATUS = {"supported_strong": 1.0, "supported_weak": 0.70}
TURN_NEGATIVE_STATUS = {"not_supported": 1.0}
SPAN_NEGATIVE_STATUS = {"negative_control_supported"}


def span_supervision_target(span: Dict) -> Optional[Tuple[int, float]]:
    """Return (binary target, weight) or None when a span must be ignored."""
    if bool(span.get("semantic_token_supervision_ignore", False)):
        return None

    tier = str(span.get("supervision_tier", "ignore"))
    causal_type = str(span.get("causal_type", "unvalidated"))
    evidence_status = str(span.get("evidence_status", "unassessed"))

    if tier in SPAN_POSITIVE_TIERS and (
        causal_type == "causal"
        or evidence_status in TURN_POSITIVE_STATUS
    ):
        return 1, SPAN_POSITIVE_TIERS[tier]

    if tier == "incidental" and (
        causal_type == "incidental"
        or evidence_status in SPAN_NEGATIVE_STATUS
    ):
        return 0, 1.0

    return None


def _validate_turn_id(
    tid: int,
    turns: Sequence[Dict],
    *,
    conversation_id: str,
    source: str,
) -> None:
    if isinstance(tid, bool) or not isinstance(tid, int):
        raise RuntimeError(
            f"{conversation_id}: {source} contains non-integer turn id {tid!r}"
        )
    if tid < 0 or tid >= len(turns):
        raise RuntimeError(
            f"{conversation_id}: {source} turn id {tid} outside 0..{len(turns)-1}"
        )
    if str(turns[tid].get("role", "")).lower() != "user":
        raise RuntimeError(
            f"{conversation_id}: {source} turn id {tid} is not a user turn"
        )


def _positive_weight_from_spans(turn: Dict, default: float) -> float:
    weight = default
    for span in turn.get("span_annotations", []) or []:
        target = span_supervision_target(span)
        if target is not None and target[0] == 1:
            weight = max(weight, float(target[1]))
    return weight


def build_evidence_turn_targets(
    record: Dict,
    turns: Sequence[Dict],
) -> Tuple[List[int], List[float]]:
    """Build multi-label user-turn targets.

    Target values:
      1  intervention-supported evidence turn
      0  explicitly tested not-supported turn, or validated benign user turn
     -1  untested / not-assessable / auxiliary / otherwise unknown

    Positive evidence_turn_ids override a turn-level not_supported outcome when
    span-level intervention evidence independently supports that same turn.
    """
    n = len(turns)
    labels = [-1] * n
    weights = [0.0] * n

    if is_auxiliary_detection_record(record) or localization_supervision_ignored(record):
        return labels, weights

    cid = str(record.get("conversation_id", "")) or "<missing>"
    label = training_label(record)

    if label == 0:
        for idx, turn in enumerate(turns):
            if str(turn.get("role", "")).lower() == "user":
                labels[idx] = 0
                weights[idx] = 1.0
        return labels, weights

    analysis = record.get("frontier_evidence_analysis") or {}
    interventions = list(analysis.get("turn_interventions") or [])

    # Dataset A stored one tested anchor intervention under evidence_analysis.
    legacy_analysis = record.get("evidence_analysis") or {}
    legacy_anchor = legacy_analysis.get("anchor_turn_intervention")
    legacy_anchor_tid = legacy_analysis.get("fresh_anchor_turn_id")
    if isinstance(legacy_anchor, dict) and isinstance(legacy_anchor_tid, int):
        interventions.append({
            "turn_id": legacy_anchor_tid,
            "status": legacy_anchor.get("status", ""),
            "_source": "legacy_anchor_turn_intervention",
        })

    status_by_turn: Dict[int, str] = {}

    for intervention in interventions:
        tid = intervention.get("turn_id")
        status = str(intervention.get("status", ""))
        if not isinstance(tid, int) or isinstance(tid, bool):
            continue
        source = str(intervention.get("_source", "turn_interventions"))
        _validate_turn_id(
            tid, turns, conversation_id=cid, source=source
        )
        status_by_turn[tid] = status
        if status in TURN_POSITIVE_STATUS:
            labels[tid] = 1
            weights[tid] = TURN_POSITIVE_STATUS[status]
        elif status in TURN_NEGATIVE_STATUS:
            labels[tid] = 0
            weights[tid] = TURN_NEGATIVE_STATUS[status]

    evidence_ids = record.get("evidence_turn_ids")
    if evidence_ids is None:
        evidence_ids = []

    # Dataset A compatibility: only CF-tier legacy pivots may become positive.
    if not evidence_ids and str(record.get("supervision_tier", "")) in {
        "cf_strong", "cf_weak"
    }:
        pivot = record.get("pivot_turn_id")
        if isinstance(pivot, int) and not isinstance(pivot, bool):
            evidence_ids = [pivot]

    seen = set()
    for raw_tid in evidence_ids:
        tid = int(raw_tid) if isinstance(raw_tid, int) and not isinstance(raw_tid, bool) else raw_tid
        _validate_turn_id(tid, turns, conversation_id=cid, source="evidence_turn_ids")
        if tid in seen:
            continue
        seen.add(tid)

        status = status_by_turn.get(tid)
        if status in TURN_POSITIVE_STATUS:
            weight = TURN_POSITIVE_STATUS[status]
        else:
            record_tier = str(record.get("supervision_tier", ""))
            default = 1.0 if record_tier == "cf_strong" else 0.70
            weight = _positive_weight_from_spans(turns[tid], default)
        labels[tid] = 1
        weights[tid] = max(weights[tid], weight)

    return labels, weights
