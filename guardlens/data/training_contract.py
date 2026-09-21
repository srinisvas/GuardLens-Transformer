"""Training-label, source, and loss-weight contract for frozen NAACL data."""
from __future__ import annotations

import math
from typing import Dict


PRIMARY_SOURCE_FAMILIES = {
    "legacy_restored_primary": "A",
    "frontier_authored_v3": "B",
}
AUXILIARY_SOURCE_FAMILIES = {
    "legacy_detection_aux": "A",
    "frontier_authored_v3_auxiliary": "B",
}
CANONICAL_DETECTION_SOURCE_FAMILY = "B"
PRIMARY_MALICIOUS_TIERS = {"cf_strong", "cf_weak", "llm_confirmed"}


def is_auxiliary_detection_record(record: Dict) -> bool:
    return bool(record.get("auxiliary_detection_only", False)) or (
        record.get("use_as") == "auxiliary_detection_only"
    )


def source_family(record: Dict) -> str:
    """Return the frozen A/B source family and reject unknown source aliases."""
    cid = str(record.get("conversation_id", "")) or "<missing>"
    source = str(record.get("corpus_source", ""))
    mapping = (
        AUXILIARY_SOURCE_FAMILIES
        if is_auxiliary_detection_record(record)
        else PRIMARY_SOURCE_FAMILIES
    )
    if source not in mapping:
        raise RuntimeError(
            f"{cid}: corpus_source={source!r} is not valid for "
            f"{'auxiliary' if is_auxiliary_detection_record(record) else 'primary'} "
            "training data"
        )
    return mapping[source]


def _binary_label(value, *, field: str, conversation_id: str) -> int:
    if isinstance(value, bool) or value not in (0, 1):
        raise RuntimeError(
            f"{conversation_id}: {field} must be binary 0/1, got {value!r}"
        )
    return int(value)


def _positive_finite_weight(value, *, field: str, conversation_id: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RuntimeError(
            f"{conversation_id}: {field} must be finite and positive, got {value!r}"
        )
    return float(value)


def validate_auxiliary_detection_contract(record: Dict) -> None:
    if not is_auxiliary_detection_record(record):
        return
    cid = str(record.get("conversation_id", "")) or "<missing>"
    family = source_family(record)
    if record.get("supervision_tier") != "auxiliary_detection":
        raise RuntimeError(
            f"{cid}: auxiliary supervision_tier must be 'auxiliary_detection'"
        )
    _binary_label(
        record.get("detection_label"),
        field="detection_label",
        conversation_id=cid,
    )
    if record.get("localization_supervision_ignore") is not True:
        raise RuntimeError(
            f"{cid}: auxiliary record must ignore localization supervision"
        )
    if record.get("pivot_supervision_ignore") is not True:
        raise RuntimeError(f"{cid}: auxiliary record must ignore pivot supervision")
    if record.get("pivot_loss_weight") != 0.0:
        raise RuntimeError(f"{cid}: auxiliary pivot_loss_weight must be 0.0")
    if record.get("span_loss_weight") != 0.0:
        raise RuntimeError(f"{cid}: auxiliary span_loss_weight must be 0.0")
    if family == "B" and record.get("pivot_turn_id") is not None:
        raise RuntimeError(f"{cid}: B auxiliary pivot_turn_id must be null")
    if family == "B" and record.get("evidence_turn_ids") not in (None, []):
        raise RuntimeError(f"{cid}: B auxiliary evidence_turn_ids must be empty")
    if (
        family == "A"
        and record.get("auxiliary_original_annotations_preserved") is not True
    ):
        raise RuntimeError(
            f"{cid}: A auxiliary must declare preserved original annotations"
        )
    detection_weight = _positive_finite_weight(
        record.get("detection_loss_weight", record.get("loss_weight")),
        field="detection_loss_weight",
        conversation_id=cid,
    )
    source = str(record.get("corpus_source", ""))
    expected_weight = 1.0 if family == "A" else 0.25
    if detection_weight != expected_weight:
        raise RuntimeError(
            f"{cid}: {source} detection_loss_weight must be {expected_weight}, "
            f"got {detection_weight}"
        )
    if family == "A" and int(record.get("detection_label")) != 0:
        raise RuntimeError(f"{cid}: restored-A auxiliary must be benign")


def validate_training_record(record: Dict) -> None:
    """Validate supervision-bearing fields before constructing any targets.

    Metadata that is useful only for provenance or evaluation slicing may remain
    in the record, but it cannot silently become a training target here.
    """
    cid = str(record.get("conversation_id", ""))
    if not cid:
        raise RuntimeError("training record is missing conversation_id")

    if is_auxiliary_detection_record(record):
        validate_auxiliary_detection_contract(record)
        return

    label = _binary_label(record.get("label"), field="label", conversation_id=cid)
    family = source_family(record)
    tier = str(record.get("supervision_tier", ""))
    if label == 0 and tier != "benign_validated":
        raise RuntimeError(
            f"{cid}: benign primary record must use benign_validated, got {tier!r}"
        )
    if label == 1 and tier not in PRIMARY_MALICIOUS_TIERS:
        raise RuntimeError(
            f"{cid}: malicious primary supervision_tier={tier!r} is unsupported"
        )

    evidence_ids = record.get("evidence_turn_ids")
    if evidence_ids is not None:
        if not isinstance(evidence_ids, list):
            raise RuntimeError(f"{cid}: evidence_turn_ids must be a JSON list")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in evidence_ids):
            raise RuntimeError(f"{cid}: evidence_turn_ids must contain integers only")
        if evidence_ids != sorted(set(evidence_ids)):
            raise RuntimeError(
                f"{cid}: evidence_turn_ids must be sorted and duplicate-free"
            )
        if evidence_ids and tier not in {"cf_strong", "cf_weak"}:
            raise RuntimeError(
                f"{cid}: supervision_tier={tier!r} cannot carry positive "
                "evidence_turn_ids"
            )

    # Dataset identity is consumed by source-aware dev selection. Referencing
    # the value here also makes an accidental A/B source remap fail before GPU
    # initialization.
    if family not in {"A", "B"}:
        raise RuntimeError(f"{cid}: unsupported source family {family!r}")


def training_label(record: Dict) -> int:
    cid = str(record.get("conversation_id", "")) or "<missing>"
    if is_auxiliary_detection_record(record):
        validate_auxiliary_detection_contract(record)
        return _binary_label(
            record.get("detection_label"),
            field="detection_label",
            conversation_id=cid,
        )
    return _binary_label(record.get("label"), field="label", conversation_id=cid)


def classification_loss_weight(record: Dict) -> float:
    """Primary labels are behaviorally validated and always receive weight 1.

    Localization confidence must never down-weight trajectory detection.
    Detection-only auxiliary examples retain their audited detection weight.
    """
    cid = str(record.get("conversation_id", "")) or "<missing>"
    if not is_auxiliary_detection_record(record):
        return 1.0
    validate_auxiliary_detection_contract(record)
    return _positive_finite_weight(
        record.get("detection_loss_weight", record.get("loss_weight")),
        field="detection_loss_weight",
        conversation_id=cid,
    )


def localization_supervision_ignored(record: Dict) -> bool:
    if is_auxiliary_detection_record(record):
        validate_auxiliary_detection_contract(record)
        return True
    return bool(record.get("localization_supervision_ignore", False))
