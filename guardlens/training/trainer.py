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
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import (
    FlatConversationCollator,
    GuardLensCollator,
    GuardLensDataset,
)
from guardlens.data.training_contract import (
    classification_loss_weight,
    is_auxiliary_detection_record,
    training_label,
)
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.schedule import get_current_phase, get_lambda_schedule


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
    if config.freeze_backbone and getattr(model, "backbone", None) is not None:
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
    }


@torch.no_grad()
def _collect_detection_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[float], List[int]]:
    probs: List[float] = []
    labels: List[int] = []
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
    return probs, labels


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

    detection = _binary_metrics(det_probs, det_labels, threshold)
    span = _binary_metrics(span_probs, span_labels, 0.5) if span_probs else None
    turn = _binary_metrics(turn_probs, turn_labels, 0.5) if turn_probs else None
    span_ap = _average_precision(span_probs, span_labels)
    turn_ap = _average_precision(turn_probs, turn_labels)
    aps = [x for x in (span_ap, turn_ap) if x is not None]
    localization_score = float(np.mean(aps)) if aps else None

    return {
        "loss": total_loss / max(1, n_batches),
        "detection": detection,
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
    score,
    score_name,
):
    return {
        "architecture_version": "causal_localization_v1",
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
    }


def train(
    config: GuardLensConfig,
    output_dir: str,
    model_name: str = "guardlens",
):
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

    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Train: {len(train_records)}  Dev: {len(dev_records)}")
    print("Held-out test: NOT LOADED")

    data_sha256 = {
        "train": _sha256(config.train_path),
        "dev": _sha256(config.dev_path),
    }
    print(f"Train SHA256: {data_sha256['train']}")
    print(f"Dev SHA256:   {data_sha256['dev']}")

    n_pos, n_neg, pos_mass, neg_mass = _weighted_detection_balance(
        train_records
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
    turn_pos_weight = turn_neg_mass / max(1e-8, turn_pos_mass)
    print(
        f"Turn supervision={turn_pos} pos/{turn_neg} neg; "
        f"weighted={turn_pos_mass:.2f}/{turn_neg_mass:.2f}; "
        f"pos_weight={turn_pos_weight:.4f}"
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = (
        FlatConversationCollator(tokenizer, config)
        if model_name == "conversation_deberta"
        else GuardLensCollator(tokenizer, config)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
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
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
    )

    loss_fn = GuardLensLoss(config)
    loss_fn.set_pos_weight(config.pos_weight)
    if turn_pos > 0 and turn_neg > 0:
        loss_fn.set_turn_pos_weight(turn_pos_weight)

    os.makedirs(output_dir, exist_ok=True)
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
            probs, labels = _collect_detection_probs(
                model, dev_loader, device
            )
            best_threshold = find_best_threshold(probs, labels)

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
            f"spanF1={span_f1:.3f} turnF1={turn_f1:.3f} "
            f"locAP={dev_metrics['localization_score']} "
            f"thr={best_threshold:.3f}"
        )

        if phase != last_phase:
            patience_counter = 0
            last_phase = phase

        det_score = float(dev_metrics["detection"]["f1"])
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
                score=det_score,
                score_name="dev_detection_f1",
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
                    score=loc_score,
                    score_name="mean_dev_turn_span_auprc",
                )
                torch.save(
                    payload, os.path.join(output_dir, "best_localization.pt")
                )

            # Canonical checkpoint selection is joint, not localization-only.
            # Detection F1, turn AUPRC and span AUPRC are all bounded [0,1],
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
                        score=joint_score,
                        score_name="mean_dev_detection_f1_turn_auprc_span_auprc",
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
        "architecture_version": "causal_localization_v1",
        "model_name": model_name,
        "best_checkpoint": os.path.basename(chosen),
        "best_detection_f1": best_detection,
        "best_localization_auprc": (
            best_localization if best_localization >= 0 else None
        ),
        "best_joint_score": best_joint if best_joint >= 0 else None,
        "data_sha256": data_sha256,
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
