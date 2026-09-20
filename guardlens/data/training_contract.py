"""Training-label and loss-weight contract for frozen NAACL data."""
from __future__ import annotations

import math
from typing import Dict


def is_auxiliary_detection_record(record: Dict) -> bool:
    return bool(record.get("auxiliary_detection_only", False)) or (
        record.get("use_as") == "auxiliary_detection_only"
    )


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
    _positive_finite_weight(
        record.get("detection_loss_weight", record.get("loss_weight")),
        field="detection_loss_weight",
        conversation_id=cid,
    )


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
