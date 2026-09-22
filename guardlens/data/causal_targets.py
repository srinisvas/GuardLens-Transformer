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
SPAN_POSITIVE_STATUS_BY_TIER = {
    "cf_strong": "supported_strong",
    "cf_weak": "supported_weak",
}
TURN_POSITIVE_STATUS = {"supported_strong": 1.0, "supported_weak": 0.70}
TURN_NEGATIVE_STATUS = {"not_supported": 1.0}
SPAN_NEGATIVE_STATUS = {"negative_control_supported", "benign_negative"}


def span_supervision_target(span: Dict) -> Optional[Tuple[int, float]]:
    """Return (binary target, weight) or None when a span must be ignored."""
    if bool(span.get("semantic_token_supervision_ignore", False)):
        return None

    tier = str(span.get("supervision_tier", "ignore"))
    causal_type = str(span.get("causal_type", "unvalidated"))
    evidence_status = str(span.get("evidence_status", "unassessed"))

    if (
        tier in SPAN_POSITIVE_TIERS
        and causal_type == "causal"
        and evidence_status == SPAN_POSITIVE_STATUS_BY_TIER[tier]
    ):
        return 1, SPAN_POSITIVE_TIERS[tier]

    if (
        tier == "incidental"
        and causal_type == "incidental"
        and evidence_status in SPAN_NEGATIVE_STATUS
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


def _positive_weight_from_spans(turn: Dict) -> Optional[float]:
    """Return turn-level support from explicit span interventions.

    semantic_token_supervision_ignore masks token labels only. A span with a
    valid supported intervention can still establish that its containing turn
    is evidence for the independent turn-localization objective.
    """
    weights = []
    for span in turn.get("span_annotations", []) or []:
        tier = str(span.get("supervision_tier", "ignore"))
        if (
            tier in SPAN_POSITIVE_TIERS
            and str(span.get("causal_type", "unvalidated")) == "causal"
            and str(span.get("evidence_status", "unassessed"))
            == SPAN_POSITIVE_STATUS_BY_TIER[tier]
        ):
            weights.append(float(SPAN_POSITIVE_TIERS[tier]))
    return max(weights) if weights else None


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
    if (
        isinstance(legacy_anchor, dict)
        and isinstance(legacy_anchor_tid, int)
        and not isinstance(legacy_anchor_tid, bool)
    ):
        interventions.append({
            "turn_id": legacy_anchor_tid,
            "status": legacy_anchor.get("status", ""),
            "_source": "legacy_anchor_turn_intervention",
        })

    status_by_turn: Dict[int, str] = {}

    for intervention in interventions:
        tid = intervention.get("turn_id")
        status = str(intervention.get("status", ""))
        source = str(intervention.get("_source", "turn_interventions"))
        _validate_turn_id(
            tid, turns, conversation_id=cid, source=source
        )
        previous_status = status_by_turn.get(tid)
        if previous_status is not None and previous_status != status:
            raise RuntimeError(
                f"{cid}: conflicting turn intervention statuses for turn {tid}: "
                f"{previous_status!r} versus {status!r}"
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

    validated_evidence_ids = []
    seen = set()
    for tid in evidence_ids:
        _validate_turn_id(
            tid, turns, conversation_id=cid, source="evidence_turn_ids"
        )
        if tid in seen:
            raise RuntimeError(
                f"{cid}: evidence_turn_ids contains duplicate turn {tid}"
            )
        seen.add(tid)
        validated_evidence_ids.append(tid)

    span_weight_by_turn: Dict[int, float] = {}
    for tid, turn in enumerate(turns):
        # Assistant annotations never enter model targets. The representation
        # audit independently rejects target-bearing assistant spans so they
        # cannot be silently admitted to a canonical run.
        if str(turn.get("role", "")).lower() != "user":
            continue
        span_weight = _positive_weight_from_spans(turn)
        if span_weight is None:
            continue
        span_weight_by_turn[tid] = span_weight

    locally_supported_ids = {
        tid
        for tid, status in status_by_turn.items()
        if status in TURN_POSITIVE_STATUS
    } | set(span_weight_by_turn)
    declared_ids = set(validated_evidence_ids)
    unbacked_ids = sorted(declared_ids - locally_supported_ids)
    if unbacked_ids:
        if len(unbacked_ids) == 1:
            detail = f"turn {unbacked_ids[0]}"
        else:
            detail = f"turns {unbacked_ids}"
        raise RuntimeError(
            f"{cid}: evidence_turn_ids contains {detail} without an explicit "
            "supported turn intervention or supported span"
        )
    undeclared_ids = sorted(locally_supported_ids - declared_ids)
    if undeclared_ids:
        raise RuntimeError(
            f"{cid}: intervention-backed positive turns {undeclared_ids} are "
            "missing from evidence_turn_ids"
        )

    for tid in validated_evidence_ids:
        prev_label = labels[tid]
        status = status_by_turn.get(tid)
        span_weight = span_weight_by_turn.get(tid)
        if status in TURN_POSITIVE_STATUS:
            weight = TURN_POSITIVE_STATUS[status]
            if span_weight is not None:
                weight = max(weight, span_weight)
        elif status in TURN_NEGATIVE_STATUS:
            if span_weight is None:
                raise RuntimeError(
                    f"{cid}: evidence_turn_ids contains turn {tid} but its "
                    f"tested turn intervention is {status!r} and no independently "
                    "supported positive span exists on that turn"
                )
            # Independent span intervention evidence may override a negative
            # whole-turn intervention result.
            weight = span_weight
        elif span_weight is not None:
            weight = span_weight
        else:
            # Reconciliation above makes this branch unreachable. Keep the
            # explicit failure so future schema changes cannot turn provenance
            # or pivot metadata into positive supervision.
            raise RuntimeError(
                f"{cid}: evidence turn {tid} lacks local intervention support"
            )

        labels[tid] = 1
        # Only combine confidences when the turn was already positive. If this
        # evidence_turn_ids pass flips a previously negative/unset target to
        # positive, stale negative confidence must not leak into the positive
        # weight.
        weights[tid] = (
            max(weights[tid], weight)
            if prev_label == 1
            else weight
        )

    return labels, weights
