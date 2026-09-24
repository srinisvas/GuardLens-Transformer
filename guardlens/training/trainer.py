"""Training loop for the NAACL causal-localization redesign.

The trainer reads frozen train/dev partitions only. Held-out test evaluation is
intentionally a separate pipeline and cannot be triggered from this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensCollator, GuardLensDataset
from guardlens.data.causal_targets import span_supervision_target
from guardlens.data.training_contract import (
    CANONICAL_DETECTION_SOURCE_FAMILY,
    classification_loss_weight,
    is_auxiliary_detection_record,
    source_family,
    training_label,
    validate_training_record,
)
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.schedule import get_current_phase, get_lambda_schedule


ARCHITECTURE_VERSION = "causal_localization_v3"
TRAINING_CONTRACT_VERSION = "restored_a_pre_response_v3"


def load_records(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
    if not records:
        raise RuntimeError(f"empty dataset: {path}")
    return records


def _code_revision() -> str:
    env_sha = os.environ.get("GUARDLENS_CODE_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _runtime_versions() -> Dict[str, str]:
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_train_dev_disjoint(train_records: Sequence[Dict], dev_records: Sequence[Dict]):
    train_ids = {
        str(r.get("conversation_id", ""))
        for r in train_records if r.get("conversation_id")
    }
    dev_ids = {
        str(r.get("conversation_id", ""))
        for r in dev_records if r.get("conversation_id")
    }
    overlap = train_ids & dev_ids
    if overlap:
        raise RuntimeError(
            f"train/dev conversation leakage: {len(overlap)} shared ids"
        )
    if any(is_auxiliary_detection_record(r) for r in dev_records):
        raise RuntimeError("dev must remain primary-only; auxiliary records are train-only")


def _validate_frozen_schema(
    train_records: Sequence[Dict],
    dev_records: Sequence[Dict],
    train_variant: str,
) -> None:
    if train_variant not in {"primary", "primary_plus_auxiliary"}:
        raise RuntimeError(f"unsupported train_variant={train_variant!r}")

    for record in [*train_records, *dev_records]:
        validate_training_record(record)

    train_aux = [r for r in train_records if is_auxiliary_detection_record(r)]
    if train_variant == "primary" and train_aux:
        raise RuntimeError("primary train variant must not contain auxiliary records")
    if train_variant == "primary_plus_auxiliary" and not train_aux:
        raise RuntimeError(
            "primary_plus_auxiliary train variant contains no auxiliary records"
        )

    dev_families = {source_family(r) for r in dev_records}
    if dev_families != {"A", "B"}:
        raise RuntimeError(
            f"dev must contain primary source families A and B, got {sorted(dev_families)}"
        )
    b_dev_labels = {
        training_label(r)
        for r in dev_records
        if source_family(r) == CANONICAL_DETECTION_SOURCE_FAMILY
    }
    if b_dev_labels != {0, 1}:
        raise RuntimeError(
            "B-dev must contain both detection classes for canonical selection, "
            f"got {sorted(b_dev_labels)}"
        )
    if train_variant == "primary_plus_auxiliary":
        aux_families = {source_family(r) for r in train_aux}
        if aux_families != {"A", "B"}:
            raise RuntimeError(
                "canonical auxiliary training must contain A and B auxiliaries, "
                f"got {sorted(aux_families)}"
            )


def _resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip()
    if not requested:
        raise RuntimeError("device must be non-empty")
    if requested.lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested but CUDA is unavailable"
        )
    device = torch.device(requested)
    if device.type == "cuda" and device.index is not None:
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested CUDA device index {device.index} but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
    return device


def _prepare_output_dir(output_dir: str) -> None:
    if not output_dir:
        raise RuntimeError("output directory must be non-empty")
    if os.path.exists(output_dir):
        raise RuntimeError(
            f"output directory already exists: {output_dir}; use a fresh run directory"
        )
    os.makedirs(output_dir, exist_ok=False)


def _weighted_detection_balance(records: Sequence[Dict]) -> Tuple[int, int, float, float]:
    n_pos = n_neg = 0
    pos_mass = neg_mass = 0.0
    for record in records:
        label = training_label(record)
        weight = classification_loss_weight(record)
        if label == 1:
            n_pos += 1
            pos_mass += weight
        else:
            n_neg += 1
            neg_mass += weight
    return n_pos, n_neg, pos_mass, neg_mass


def _span_annotation_balance(records: Sequence[Dict]) -> Tuple[int, int]:
    positive = negative = 0
    for record in records:
        if is_auxiliary_detection_record(record):
            continue
        for turn in record.get("turns", []) or []:
            if str(turn.get("role", "")).lower() != "user":
                continue
            for span in turn.get("span_annotations", []) or []:
                target = span_supervision_target(span)
                if target is None:
                    continue
                if target[0] == 1:
                    positive += 1
                else:
                    negative += 1
    return positive, negative


def _turn_supervision_balance(dataset: GuardLensDataset) -> Tuple[int, int, float, float]:
    n_pos = n_neg = 0
    pos_mass = neg_mass = 0.0
    for idx in range(len(dataset)):
        item = dataset[idx]
        for label, weight in zip(
            item["evidence_turn_labels"], item["evidence_turn_weights"]
        ):
            if label == 1 and weight > 0:
                n_pos += 1
                pos_mass += float(weight)
            elif label == 0 and weight > 0:
                n_neg += 1
                neg_mass += float(weight)
    return n_pos, n_neg, pos_mass, neg_mass


def find_best_threshold(probs: Sequence[float], labels: Sequence[int]) -> float:
    best_f1 = -1.0
    best_thresh = 0.5
    for thresh in np.linspace(0.05, 0.95, 181):
        metrics = _binary_metrics(probs, labels, float(thresh))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = float(thresh)
    return best_thresh


def _binary_metrics(
    probs: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for prob, label in zip(probs, labels):
        pred = 1 if float(prob) >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
    }


def _average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if not scores:
        return None
    positives = sum(int(x) == 1 for x in labels)
    negatives = sum(int(x) == 0 for x in labels)
    if positives == 0 or negatives == 0:
        return None
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), i),
    )
    tp = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, 1):
        if int(labels[idx]) == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def _roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if not scores:
        return None
    positives = sum(int(x) == 1 for x in labels)
    negatives = sum(int(x) == 0 for x in labels)
    if positives == 0 or negatives == 0:
        return None

    # Mann-Whitney U with average ranks for tied scores.
    ordered = sorted(
        ((float(score), int(label)) for score, label in zip(scores, labels)),
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _, label in ordered[start:end]
        )
        start = end
    u_stat = positive_rank_sum - positives * (positives + 1) / 2.0
    return float(u_stat / (positives * negatives))


def _detection_metrics(
    probs: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> Dict[str, object]:
    result: Dict[str, object] = _binary_metrics(probs, labels, threshold)
    result["auprc"] = _average_precision(probs, labels)
    result["auroc"] = _roc_auc(probs, labels)
    result["records"] = len(labels)
    result["positives"] = sum(int(x) == 1 for x in labels)
    result["negatives"] = sum(int(x) == 0 for x in labels)
    return result


def _source_detection_metrics(
    probs: Sequence[float],
    labels: Sequence[int],
    families: Sequence[str],
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for family in sorted(set(families)):
        indices = [i for i, value in enumerate(families) if value == family]
        result[family] = _detection_metrics(
            [probs[i] for i in indices],
            [labels[i] for i in indices],
            threshold,
        )
    return result


def _canonical_detection_score(dev_metrics: Dict[str, object]) -> float:
    by_source = dev_metrics.get("detection_by_source") or {}
    source_metrics = by_source.get(CANONICAL_DETECTION_SOURCE_FAMILY) or {}
    score = source_metrics.get("auprc")
    if score is None:
        raise RuntimeError(
            "canonical checkpoint selection requires B-dev detection AUPRC"
        )
    return float(score)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: GuardLensLoss,
    config: GuardLensConfig,
    epoch: int,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    if getattr(model, "backbone_fully_frozen", False) and getattr(model, "backbone", None) is not None:
        model.backbone.eval()

    phase = get_current_phase(epoch, config)
    lambda_detection, lambda_turn, lambda_span = get_lambda_schedule(
        epoch, config
    )

    totals = Counter()
    correct = 0
    seen = 0
    n_batches = 0
    accumulation = max(1, int(config.gradient_accumulation))
    loader_steps = len(loader)
    if loader_steps <= 0:
        raise RuntimeError("training DataLoader is empty")

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        role_ids = batch["role_ids"].to(device)
        token_labels = batch["token_labels"].to(device)
        labels = batch["labels"].to(device)

        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, loader_steps - group_start)

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                turn_mask=turn_mask,
                role_ids=role_ids,
                compute_localization=(phase >= 2),
            )
            losses = loss_fn(
                outputs,
                labels,
                token_labels,
                span_weights=batch["span_weights"].to(device),
                detection_weights=batch["detection_weights"].to(device),
                turn_labels=batch["turn_labels"].to(device),
                turn_weights=batch["turn_weights"].to(device),
                phase=phase,
                lambda_detection=lambda_detection,
                lambda_turn=lambda_turn,
                lambda_span=lambda_span,
            )
            loss = losses["total"]
            (loss / group_size).backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM at epoch={epoch} step={step} with "
                    f"batch_size={config.batch_size}, max_turns={config.max_turns}, "
                    f"max_tokens_per_turn={config.max_tokens_per_turn}. "
                    "Aborting instead of silently skipping a batch."
                ) from exc
            raise

        if (
            (step + 1) % accumulation == 0
            or (step + 1) == loader_steps
        ):
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += float(losses["total"].detach())
        totals["detection_loss"] += float(losses["detection"].detach())
        if "turn" in losses:
            totals["turn_loss"] += float(losses["turn"].detach())
        if "span" in losses:
            totals["span_loss"] += float(losses["span"].detach())

        preds = (torch.sigmoid(outputs["cls_logits"]) >= 0.5).long()
        correct += int((preds == labels).sum().item())
        seen += int(labels.numel())
        n_batches += 1

    return {
        "loss": totals["loss"] / max(1, n_batches),
        "detection_loss": totals["detection_loss"] / max(1, n_batches),
        "turn_loss": totals["turn_loss"] / max(1, n_batches),
        "span_loss": totals["span_loss"] / max(1, n_batches),
        "accuracy": correct / max(1, seen),
        "phase": phase,
        "lambda_detection": lambda_detection,
        "lambda_turn": lambda_turn,
        "lambda_span": lambda_span,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.no_grad()
def _collect_detection_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[float], List[int], List[str]]:
    probs: List[float] = []
    labels: List[int] = []
    families: List[str] = []
    model.eval()
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            turn_mask=batch["turn_mask"].to(device),
            role_ids=batch["role_ids"].to(device),
            compute_localization=False,
        )
        probs.extend(torch.sigmoid(outputs["cls_logits"]).cpu().tolist())
        labels.extend(batch["labels"].tolist())
        families.extend(str(row["source_family"]) for row in batch["metadata"])
    return probs, labels, families


@torch.no_grad()
def evaluate_dev(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: GuardLensLoss,
    config: GuardLensConfig,
    device: torch.device,
    threshold: float,
) -> Dict[str, object]:
    """Dev-only metrics for checkpoint selection. Not the paper evaluation."""
    model.eval()

    det_probs: List[float] = []
    det_labels: List[int] = []
    det_families: List[str] = []
    span_probs: List[float] = []
    span_labels: List[int] = []
    turn_probs: List[float] = []
    turn_labels: List[int] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        token_labels = batch["token_labels"].to(device)
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            turn_mask=batch["turn_mask"].to(device),
            role_ids=batch["role_ids"].to(device),
            compute_localization=True,
        )
        losses = loss_fn(
            outputs,
            labels,
            token_labels,
            span_weights=batch["span_weights"].to(device),
            detection_weights=batch["detection_weights"].to(device),
            turn_labels=batch["turn_labels"].to(device),
            turn_weights=batch["turn_weights"].to(device),
            phase=2,
            lambda_detection=config.lambda_detection,
            lambda_turn=config.lambda_turn,
            lambda_span=config.lambda_span,
        )
        total_loss += float(losses["total"].item())
        n_batches += 1

        det_probs.extend(torch.sigmoid(outputs["cls_logits"]).cpu().tolist())
        det_labels.extend(labels.cpu().tolist())
        det_families.extend(
            str(row["source_family"]) for row in batch["metadata"]
        )

        if outputs.get("attr_probs") is not None:
            span_valid = (
                (batch["token_labels"] >= 0)
                & (batch["span_weights"] > 0)
            )
            if span_valid.any():
                span_probs.extend(
                    outputs["attr_probs"].cpu()[span_valid].tolist()
                )
                span_labels.extend(batch["token_labels"][span_valid].tolist())

        if outputs.get("turn_probs") is not None:
            turn_valid = (
                (batch["turn_labels"] >= 0)
                & (batch["turn_weights"] > 0)
            )
            if turn_valid.any():
                turn_probs.extend(
                    outputs["turn_probs"].cpu()[turn_valid].tolist()
                )
                turn_labels.extend(batch["turn_labels"][turn_valid].tolist())

    detection = _detection_metrics(det_probs, det_labels, threshold)
    detection_by_source = _source_detection_metrics(
        det_probs, det_labels, det_families, threshold
    )
    source_auprcs = [
        float(metrics["auprc"])
        for metrics in detection_by_source.values()
        if metrics.get("auprc") is not None
    ]
    span = _binary_metrics(span_probs, span_labels, 0.5) if span_probs else None
    turn = _binary_metrics(turn_probs, turn_labels, 0.5) if turn_probs else None
    span_ap = _average_precision(span_probs, span_labels)
    turn_ap = _average_precision(turn_probs, turn_labels)
    aps = [x for x in (span_ap, turn_ap) if x is not None]
    localization_score = float(np.mean(aps)) if aps else None

    return {
        "loss": total_loss / max(1, n_batches),
        "detection": detection,
        "detection_by_source": detection_by_source,
        "detection_macro_source_auprc": (
            float(np.mean(source_auprcs)) if source_auprcs else None
        ),
        "canonical_detection_source_family": (
            CANONICAL_DETECTION_SOURCE_FAMILY
        ),
        "span": span,
        "turn": turn,
        "span_auprc": span_ap,
        "turn_auprc": turn_ap,
        "localization_score": localization_score,
        "n_span_targets": len(span_labels),
        "n_turn_targets": len(turn_labels),
    }


def _checkpoint_payload(
    *,
    model,
    config,
    model_name,
    epoch,
    phase,
    threshold,
    dev_metrics,
    data_sha256,
    code_sha,
    score,
    score_name,
    runtime_versions,
):
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "epoch": epoch,
        "phase": phase,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "config": config,
        "dev_metrics": dev_metrics,
        "threshold": threshold,
        "score": score,
        "score_name": score_name,
        "data_sha256": data_sha256,
        "code_sha": code_sha,
        "runtime_versions": runtime_versions,
    }


def train(
    config: GuardLensConfig,
    output_dir: str,
    model_name: str = "guardlens",
):
    if model_name != "guardlens":
        raise RuntimeError(
            "baseline training is intentionally disabled until the fair-context "
            "baseline migration is complete"
        )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if not config.train_path or not config.dev_path:
        raise RuntimeError(
            "frozen --train-path and --dev-path are required; "
            "internal re-splitting is disabled"
        )
    for path in (config.train_path, config.dev_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    train_records = load_records(config.train_path)
    dev_records = load_records(config.dev_path)
    _assert_train_dev_disjoint(train_records, dev_records)
    _validate_frozen_schema(
        train_records, dev_records, config.train_variant
    )

    device = _resolve_device(config.device)
    print(f"Device: {device}")
    print(f"Train: {len(train_records)}  Dev: {len(dev_records)}")
    print(f"Train variant: {config.train_variant}")
    print("Held-out test: NOT LOADED")

    data_sha256 = {
        "train": _sha256(config.train_path),
        "dev": _sha256(config.dev_path),
    }
    code_sha = _code_revision()
    runtime_versions = _runtime_versions()
    print(f"Train SHA256: {data_sha256['train']}")
    print(f"Dev SHA256:   {data_sha256['dev']}")
    print(f"Code SHA:     {code_sha}")
    print(f"Runtime:      {runtime_versions}")
    _prepare_output_dir(output_dir)

    n_pos, n_neg, pos_mass, neg_mass = _weighted_detection_balance(
        train_records
    )
    dev_pos = sum(training_label(r) == 1 for r in dev_records)
    dev_neg = len(dev_records) - dev_pos
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"train split must contain both detection classes, got pos={n_pos} neg={n_neg}"
        )
    if dev_pos == 0 or dev_neg == 0:
        raise RuntimeError(
            f"dev split must contain both detection classes, got pos={dev_pos} neg={dev_neg}"
        )
    if config.pos_weight <= 0:
        config.pos_weight = neg_mass / max(1e-8, pos_mass)
    print(
        f"Detection balance raw={n_pos} pos/{n_neg} neg; "
        f"weighted={pos_mass:.2f} pos/{neg_mass:.2f} neg; "
        f"pos_weight={config.pos_weight:.4f}"
    )

    tier_dist = Counter(
        str(r.get("supervision_tier", "?")) for r in train_records
    )
    aux_count = sum(
        is_auxiliary_detection_record(r) for r in train_records
    )
    print(f"Supervision tiers: {dict(tier_dist.most_common())}")
    print(f"Detection-only auxiliary records: {aux_count}")

    train_dataset = GuardLensDataset(train_records, config)
    dev_dataset = GuardLensDataset(dev_records, config)

    turn_pos, turn_neg, turn_pos_mass, turn_neg_mass = (
        _turn_supervision_balance(train_dataset)
    )
    dev_turn_pos, dev_turn_neg, _, _ = _turn_supervision_balance(dev_dataset)
    train_span_pos, train_span_neg = _span_annotation_balance(train_records)
    dev_span_pos, dev_span_neg = _span_annotation_balance(dev_records)

    missing = []
    if turn_pos == 0 or turn_neg == 0:
        missing.append(
            f"train turn targets pos={turn_pos} neg={turn_neg}"
        )
    if dev_turn_pos == 0 or dev_turn_neg == 0:
        missing.append(
            f"dev turn targets pos={dev_turn_pos} neg={dev_turn_neg}"
        )
    if train_span_pos == 0 or train_span_neg == 0:
        missing.append(
            f"train span targets pos={train_span_pos} neg={train_span_neg}"
        )
    if dev_span_pos == 0 or dev_span_neg == 0:
        missing.append(
            f"dev span targets pos={dev_span_pos} neg={dev_span_neg}"
        )
    if missing:
        raise RuntimeError(
            "causal-localization supervision coverage is incomplete: "
            + "; ".join(missing)
        )

    turn_pos_weight = turn_neg_mass / max(1e-8, turn_pos_mass)
    print(
        f"Turn supervision train={turn_pos} pos/{turn_neg} neg; "
        f"dev={dev_turn_pos} pos/{dev_turn_neg} neg; "
        f"weighted train={turn_pos_mass:.2f}/{turn_neg_mass:.2f}; "
        f"pos_weight={turn_pos_weight:.4f}"
    )
    print(
        f"Span annotation targets train={train_span_pos} pos/{train_span_neg} neg; "
        f"dev={dev_span_pos} pos/{dev_span_neg} neg"
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.backbone_name,
        revision=config.backbone_revision,
        use_fast=True,
    )
    collator = GuardLensCollator(tokenizer, config)

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=max(1, config.batch_size * 2),
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if model_name not in MODEL_REGISTRY:
        raise RuntimeError(
            f"unknown model {model_name!r}; available={sorted(MODEL_REGISTRY)}"
        )
    model = MODEL_REGISTRY[model_name](config)
    model.setup_backbone()
    model = model.to(device)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model={model_name} total={total_params:,} "
        f"trainable={trainable:,} frozen={total_params-trainable:,}"
    )

    backbone_param_ids = {
        id(param)
        for param in model.backbone.parameters()
        if param.requires_grad
    }
    head_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in backbone_param_ids
    ]
    backbone_params = [
        param for param in model.backbone.parameters() if param.requires_grad
    ]
    optimizer_groups = [{"params": head_params, "lr": config.learning_rate}]
    max_learning_rates = [config.learning_rate]
    if backbone_params:
        optimizer_groups.append({
            "params": backbone_params,
            "lr": config.backbone_learning_rate,
        })
        max_learning_rates.append(config.backbone_learning_rate)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=config.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / max(1, int(config.gradient_accumulation))
    )
    total_steps = max(1, optimizer_steps_per_epoch * config.max_epochs)
    pct_start = min(
        0.99,
        max(
            1.0 / total_steps,
            config.warmup_steps / total_steps,
        ),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rates,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
    )

    loss_fn = GuardLensLoss(config)
    loss_fn.set_pos_weight(config.pos_weight)
    if turn_pos > 0 and turn_neg > 0:
        loss_fn.set_turn_pos_weight(turn_pos_weight)

    best_detection = -1.0
    best_localization = -1.0
    best_joint = -1.0
    best_threshold = config.default_threshold
    patience_counter = 0
    last_phase = 1
    last_dev_metrics = None

    print(
        f"Training {config.max_epochs} epochs: "
        f"phase1 detection-only 0-{config.phase1_epochs-1}; "
        f"phase2 joint {config.phase1_epochs}-{config.max_epochs-1}"
    )

    for epoch in range(config.max_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            config,
            epoch,
            device,
        )

        if (epoch + 1) % config.eval_every != 0:
            continue

        if config.tune_threshold:
            probs, labels, families = _collect_detection_probs(
                model, dev_loader, device
            )
            canonical_indices = [
                i for i, family in enumerate(families)
                if family == CANONICAL_DETECTION_SOURCE_FAMILY
            ]
            if not canonical_indices:
                raise RuntimeError(
                    "cannot tune threshold: dev has no B-source records"
                )
            best_threshold = find_best_threshold(
                [probs[i] for i in canonical_indices],
                [labels[i] for i in canonical_indices],
            )

        dev_metrics = evaluate_dev(
            model,
            dev_loader,
            loss_fn,
            config,
            device,
            best_threshold,
        )
        last_dev_metrics = dev_metrics
        phase = train_metrics["phase"]
        span_f1 = (
            dev_metrics["span"]["f1"]
            if dev_metrics["span"] is not None else float("nan")
        )
        turn_f1 = (
            dev_metrics["turn"]["f1"]
            if dev_metrics["turn"] is not None else float("nan")
        )
        print(
            f"Ep {epoch:02d} P{phase} "
            f"loss={train_metrics['loss']:.4f} "
            f"detF1={dev_metrics['detection']['f1']:.3f} "
            f"BdetAP={_canonical_detection_score(dev_metrics):.3f} "
            f"spanF1={span_f1:.3f} turnF1={turn_f1:.3f} "
            f"locAP={dev_metrics['localization_score']} "
            f"lr={train_metrics['learning_rate']:.3e} "
            f"lamLoc={train_metrics['lambda_span']:.3f} "
            f"thr={best_threshold:.3f}"
        )

        if phase != last_phase:
            patience_counter = 0
            last_phase = phase

        det_score = _canonical_detection_score(dev_metrics)
        if det_score > best_detection:
            best_detection = det_score
            payload = _checkpoint_payload(
                model=model,
                config=config,
                model_name=model_name,
                epoch=epoch,
                phase=phase,
                threshold=best_threshold,
                dev_metrics=dev_metrics,
                data_sha256=data_sha256,
                code_sha=code_sha,
                score=det_score,
                score_name="b_dev_detection_auprc",
                runtime_versions=runtime_versions,
            )
            torch.save(
                payload, os.path.join(output_dir, "best_detection.pt")
            )

        loc_score = dev_metrics["localization_score"]
        if phase >= 2 and loc_score is not None:
            loc_score = float(loc_score)
            if loc_score > best_localization:
                best_localization = loc_score
                payload = _checkpoint_payload(
                    model=model,
                    config=config,
                    model_name=model_name,
                    epoch=epoch,
                    phase=phase,
                    threshold=best_threshold,
                    dev_metrics=dev_metrics,
                    data_sha256=data_sha256,
                    code_sha=code_sha,
                    score=loc_score,
                    score_name="mean_dev_turn_span_auprc",
                    runtime_versions=runtime_versions,
                )
                torch.save(
                    payload, os.path.join(output_dir, "best_localization.pt")
                )

            # Canonical checkpoint selection is joint, not localization-only.
            # B-source detection AUPRC, turn AUPRC and span AUPRC are all bounded [0,1],
            # so an equal-weight mean is transparent and avoids selecting a
            # localization peak that materially sacrifices detection.
            turn_ap = dev_metrics.get("turn_auprc")
            span_ap = dev_metrics.get("span_auprc")
            if turn_ap is not None and span_ap is not None:
                joint_score = float(np.mean([
                    det_score, float(turn_ap), float(span_ap)
                ]))
                dev_metrics["joint_selection_score"] = joint_score
                if joint_score > best_joint:
                    best_joint = joint_score
                    patience_counter = 0
                    payload = _checkpoint_payload(
                        model=model,
                        config=config,
                        model_name=model_name,
                        epoch=epoch,
                        phase=phase,
                        threshold=best_threshold,
                        dev_metrics=dev_metrics,
                        data_sha256=data_sha256,
                        code_sha=code_sha,
                        score=joint_score,
                        score_name="mean_b_dev_detection_turn_span_auprc",
                        runtime_versions=runtime_versions,
                    )
                    torch.save(
                        payload, os.path.join(output_dir, "best_joint.pt")
                    )
                elif config.patience > 0:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(
                            f"Early stop after {config.patience} "
                            "joint-phase evaluations without joint-score improvement"
                        )
                        break

    joint_ckpt = os.path.join(output_dir, "best_joint.pt")
    localization_ckpt = os.path.join(
        output_dir, "best_localization.pt"
    )
    detection_ckpt = os.path.join(output_dir, "best_detection.pt")
    if os.path.exists(joint_ckpt):
        chosen = joint_ckpt
    elif os.path.exists(localization_ckpt):
        chosen = localization_ckpt
    elif os.path.exists(detection_ckpt):
        chosen = detection_ckpt
    else:
        raise RuntimeError("training produced no checkpoint")

    shutil.copy2(chosen, os.path.join(output_dir, "best.pt"))
    summary = {
        "status": "completed",
        "architecture_version": ARCHITECTURE_VERSION,
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "model_name": model_name,
        "best_checkpoint": os.path.basename(chosen),
        "best_b_dev_detection_auprc": best_detection,
        "best_localization_auprc": (
            best_localization if best_localization >= 0 else None
        ),
        "best_joint_score": best_joint if best_joint >= 0 else None,
        "data_sha256": data_sha256,
        "code_sha": code_sha,
        "runtime_versions": runtime_versions,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "held_out_test_accessed": False,
        "last_dev_metrics": last_dev_metrics,
    }
    with open(
        os.path.join(output_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Best checkpoint: {chosen}")
    print("Held-out test accessed: NO")
    return summary
