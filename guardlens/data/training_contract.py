"""Shared training-label contract for primary and optional auxiliary examples."""
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


def validate_auxiliary_detection_contract(record: Dict) -> None:
    if not is_auxiliary_detection_record(record):
        return
    cid = str(record.get("conversation_id", "")) or "<missing>"
    _binary_label(record.get("detection_label"), field="detection_label", conversation_id=cid)
    if record.get("localization_supervision_ignore") is not True:
        raise RuntimeError(f"{cid}: auxiliary record must ignore localization supervision")
    if record.get("pivot_supervision_ignore") is not True:
        raise RuntimeError(f"{cid}: auxiliary record must ignore pivot supervision")
    if record.get("pivot_loss_weight") != 0.0:
        raise RuntimeError(f"{cid}: auxiliary pivot_loss_weight must be 0.0")
    if record.get("span_loss_weight") != 0.0:
        raise RuntimeError(f"{cid}: auxiliary span_loss_weight must be 0.0")
    weight = record.get("detection_loss_weight", record.get("loss_weight"))
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) <= 0
    ):
        raise RuntimeError(
            f"{cid}: auxiliary detection_loss_weight must be finite and positive"
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
    cid = str(record.get("conversation_id", "")) or "<missing>"
    if is_auxiliary_detection_record(record):
        validate_auxiliary_detection_contract(record)
        value = record.get("detection_loss_weight", record.get("loss_weight"))
    else:
        value = record.get("loss_weight", 0.5)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RuntimeError(
            f"{cid}: classification loss weight must be finite and positive"
        )
    return float(value)


def localization_supervision_ignored(record: Dict) -> bool:
    if is_auxiliary_detection_record(record):
        validate_auxiliary_detection_contract(record)
    return bool(record.get("localization_supervision_ignore", False))


def counterfactual_loss_eligible(record: Dict) -> bool:
    return not is_auxiliary_detection_record(record)
