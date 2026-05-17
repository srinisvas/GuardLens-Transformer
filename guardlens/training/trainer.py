"""Training and evaluation loops — v11 dataset compatible."""

import json
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import (
    GuardLensDataset, GuardLensCollator, FlatConversationCollator,
    build_weighted_sampler,
)
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.schedule import get_current_phase, get_lambda_schedule


# =========================================================
# Training epoch
# =========================================================

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
    if config.freeze_backbone and hasattr(model, "backbone") and model.backbone is not None:
        model.backbone.eval()

    phase = get_current_phase(epoch, config)
    lambda_cls, lambda_attr, lambda_cf = get_lambda_schedule(epoch, config)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_attr_loss = 0.0
    total_cf_loss = 0.0
    total_pivot_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        role_ids = batch["role_ids"].to(device)
        token_labels = batch["token_labels"].to(device)
        span_weights = batch["span_weights"].to(device)
        labels = batch["labels"].to(device)
        sample_weights = batch["sample_weights"].to(device)
        pivot_labels = batch["pivot_labels"].to(device)
        pivot_kind_labels = batch["pivot_kind_labels"].to(device)

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                turn_mask=turn_mask,
                role_ids=role_ids,
                compute_attribution=(phase >= 2),
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"  OOM at step {step}, skipping batch")
                continue
            raise

        losses = loss_fn(
            outputs, labels, token_labels,
            span_weights=span_weights,
            sample_weights=sample_weights,
            pivot_labels=pivot_labels,
            pivot_kind_labels=pivot_kind_labels,
            phase=phase,
            lambda_cls=lambda_cls,
            lambda_attr=lambda_attr,
            lambda_cf=lambda_cf,
            lambda_pivot=config.lambda_pivot,
        )
        loss = losses["total"]

        # Counterfactual loss (phase 3, only for models with attribution)
        if (phase >= 3 and lambda_cf > 0
                and outputs.get("attr_probs") is not None
                and hasattr(model, "forward_cf")):
            phase3_start = config.phase1_epochs + config.phase2_epochs
            cf_progress = min(1.0, (epoch - phase3_start) / max(1, config.phase3_epochs))

            l_cf = loss_fn.counterfactual_loss(model, {
                "attention_mask": attention_mask,
                "turn_mask": turn_mask,
                "role_ids": role_ids,
                "labels": labels,
            }, outputs, cf_progress=cf_progress)
            loss = loss + lambda_cf * l_cf
            total_cf_loss += l_cf.item()

        loss = loss / config.gradient_accumulation
        loss.backward()

        if (step + 1) % config.gradient_accumulation == 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        # Log the actual total loss including CF
        actual_total = loss.detach().item() * config.gradient_accumulation
        total_loss += actual_total
        total_cls_loss += losses["cls"].item()
        if "attr" in losses:
            total_attr_loss += losses["attr"].item()
        if "pivot" in losses:
            total_pivot_loss += losses["pivot"].item()

        preds = (torch.sigmoid(outputs["cls_logits"]) > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "cls_loss": total_cls_loss / max(1, n_batches),
        "attr_loss": total_attr_loss / max(1, n_batches),
        "cf_loss": total_cf_loss / max(1, n_batches),
        "pivot_loss": total_pivot_loss / max(1, n_batches),
        "accuracy": correct / max(1, total),
        "phase": phase,
        "lambda_cls": lambda_cls,
        "lambda_attr": lambda_attr,
        "lambda_cf": lambda_cf,
    }


# =========================================================
# Evaluation
# =========================================================

def find_best_threshold(probs: List[float], labels: List[int]) -> float:
    """Find classification threshold that maximizes F1 on dev set."""
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in [i / 100.0 for i in range(20, 80)]:
        preds = [1 if p > thresh else 0 for p in probs]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: GuardLensLoss,
    config: GuardLensConfig,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_meta = []
    total_loss = 0.0
    n_batches = 0

    attr_tp = attr_fp = attr_fn = attr_tn = 0
    # Per-tier attribution stats
    tier_attr_stats = {}

    pivot_correct = 0
    pivot_total = 0
    pivot_mal_correct = 0
    pivot_mal_total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        role_ids = batch["role_ids"].to(device)
        token_labels = batch["token_labels"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            turn_mask=turn_mask,
            role_ids=role_ids,
            compute_attribution=True,
        )

        losses = loss_fn(
            outputs, labels, token_labels,
            span_weights=batch["span_weights"].to(device),
            sample_weights=batch["sample_weights"].to(device),
            pivot_labels=batch["pivot_labels"].to(device),
            pivot_kind_labels=batch["pivot_kind_labels"].to(device),
            phase=3,
        )
        total_loss += losses["total"].item()
        n_batches += 1

        probs = torch.sigmoid(outputs["cls_logits"])
        preds = (probs > threshold).long()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_meta.extend(batch["metadata"])

        # Attribution metrics
        if outputs["attr_probs"] is not None:
            ap = (outputs["attr_probs"] > 0.5).long()
            valid = token_labels >= 0
            if valid.any():
                a_pred = ap[valid]
                a_true = token_labels[valid]
                attr_tp += ((a_pred == 1) & (a_true == 1)).sum().item()
                attr_fp += ((a_pred == 1) & (a_true == 0)).sum().item()
                attr_fn += ((a_pred == 0) & (a_true == 1)).sum().item()
                attr_tn += ((a_pred == 0) & (a_true == 0)).sum().item()

        # Pivot accuracy
        if outputs.get("pivot_logits") is not None:
            pivot_preds = outputs["pivot_logits"].argmax(dim=1)
            pivot_gt = batch["pivot_labels"].to(device)
            valid_pivot = pivot_gt >= 0  # Exclude truncated (-1)
            if valid_pivot.any():
                pivot_correct += (pivot_preds[valid_pivot] == pivot_gt[valid_pivot]).sum().item()
                pivot_total += valid_pivot.sum().item()
                # Malicious-only
                mal_mask = valid_pivot & (labels == 1)
                if mal_mask.any():
                    pivot_mal_correct += (pivot_preds[mal_mask] == pivot_gt[mal_mask]).sum().item()
                    pivot_mal_total += mal_mask.sum().item()

    # Classification metrics
    preds_t = torch.tensor(all_preds)
    labels_t = torch.tensor(all_labels)
    tp = ((preds_t == 1) & (labels_t == 1)).sum().item()
    fp = ((preds_t == 1) & (labels_t == 0)).sum().item()
    fn = ((preds_t == 0) & (labels_t == 1)).sum().item()
    tn = ((preds_t == 0) & (labels_t == 0)).sum().item()

    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    # Attribution F1
    attr_precision = attr_tp / max(1, attr_tp + attr_fp)
    attr_recall = attr_tp / max(1, attr_tp + attr_fn)
    attr_f1 = 2 * attr_precision * attr_recall / max(1e-8, attr_precision + attr_recall)

    # Per-difficulty accuracy
    diff_acc = {}
    for diff in ["easy", "medium", "hard"]:
        idx = [i for i, m in enumerate(all_meta) if m["difficulty"] == diff]
        if idx:
            c = sum(1 for i in idx if all_preds[i] == all_labels[i])
            diff_acc[diff] = c / len(idx)

    # Per-family accuracy
    fam_acc = {}
    for fam in set(m["family"] for m in all_meta):
        idx = [i for i, m in enumerate(all_meta) if m["family"] == fam]
        if len(idx) >= 5:
            c = sum(1 for i in idx if all_preds[i] == all_labels[i])
            fam_acc[fam] = round(c / len(idx), 3)

    # Per-transfer-tier accuracy (v11)
    tier_acc = {}
    for tier in ["transfer_success", "target_only", "cross_only", "no_jailbreak", "benign"]:
        idx = [i for i, m in enumerate(all_meta) if m.get("transfer_tier") == tier]
        if idx:
            c = sum(1 for i in idx if all_preds[i] == all_labels[i])
            tier_acc[tier] = round(c / len(idx), 3)

    # Per-benign-status accuracy (v11)
    benign_acc = {}
    for status in ["clean_benign", "validated_benign_twin"]:
        idx = [i for i, m in enumerate(all_meta) if m.get("benign_status") == status]
        if idx:
            c = sum(1 for i in idx if all_preds[i] == all_labels[i])
            benign_acc[status] = round(c / len(idx), 3)

    # Per-supervision-tier accuracy (v11)
    sup_acc = {}
    for tier in ["cf_strong", "cf_weak", "llm_confirmed", "construction", "benign_validated"]:
        idx = [i for i, m in enumerate(all_meta) if m.get("supervision_tier") == tier]
        if idx:
            c = sum(1 for i in idx if all_preds[i] == all_labels[i])
            sup_acc[tier] = round(c / len(idx), 3)

    return {
        "loss": total_loss / max(1, n_batches),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold,
        "difficulty_accuracy": diff_acc,
        "family_accuracy": fam_acc,
        "transfer_tier_accuracy": tier_acc,
        "benign_accuracy": benign_acc,
        "supervision_tier_accuracy": sup_acc,
        "attr_precision": attr_precision,
        "attr_recall": attr_recall,
        "attr_f1": attr_f1,
        "pivot_accuracy": pivot_correct / max(1, pivot_total),
        "pivot_accuracy_malicious": pivot_mal_correct / max(1, pivot_mal_total),
    }


# =========================================================
# Main training function
# =========================================================

def load_records(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def train(
    config: GuardLensConfig,
    data_path: str,
    output_dir: str,
    model_name: str = "guardlens",
):
    """Full training pipeline — v11 dataset compatible."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load pre-split data ----
    if config.train_path and config.dev_path and config.test_path:
        print("Loading pre-split data...")
        train_records = load_records(config.train_path)
        val_records = load_records(config.dev_path)
        test_records = load_records(config.test_path)
        print(f"  Train: {len(train_records)}, Dev: {len(val_records)}, Test: {len(test_records)}")
    else:
        # Fallback: load single file and split
        print(f"Loading data from {data_path}...")
        records = load_records(data_path)
        print(f"  {len(records)} records")
        from guardlens.data.splits import pair_aware_split
        train_idx, val_idx, test_idx = pair_aware_split(records, seed=config.seed)
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        test_records = [records[i] for i in test_idx]
        print(f"  Train: {len(train_records)}, Dev: {len(val_records)}, Test: {len(test_records)}")

    # ---- Compute class balance ----
    n_pos = sum(1 for r in train_records if r.get("label") == 1)
    n_neg = len(train_records) - n_pos
    if config.pos_weight <= 0:
        config.pos_weight = n_neg / max(1, n_pos)
    print(f"  Class balance: {n_pos} pos, {n_neg} neg (pos_weight={config.pos_weight:.2f})")

    # ---- Print tier distribution ----
    tier_dist = Counter(r.get("supervision_tier", "?") for r in train_records)
    print(f"  Supervision tiers: {dict(tier_dist.most_common())}")

    # ---- Datasets ----
    train_dataset = GuardLensDataset(train_records, config)
    val_dataset = GuardLensDataset(val_records, config)
    test_dataset = GuardLensDataset(test_records, config)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    if model_name == "conversation_deberta":
        collator = FlatConversationCollator(tokenizer, config)
    else:
        collator = GuardLensCollator(tokenizer, config)

    # ---- Sampler (oversample CF records in phase 2+) ----
    if config.oversample_cf:
        sampler = build_weighted_sampler(train_records, config)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collator, num_workers=config.num_workers,
            pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator, num_workers=config.num_workers,
            pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2,
        collate_fn=collator, num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2,
        collate_fn=collator, num_workers=config.num_workers,
    )

    # ---- Model-specific config overrides ----
    if model_name == "guardlens_no_cf":
        config.phase3_epochs = 0
        config.max_epochs = config.phase1_epochs + config.phase2_epochs

    if model_name in ("turn_level", "conversation_deberta"):
        # Baselines don't have attribution, pivot, or CF heads
        config.phase3_epochs = 0
        config.max_epochs = config.phase1_epochs + config.phase2_epochs
        config.lambda_attr = 0.0
        config.lambda_cf = 0.0
        config.lambda_pivot = 0.0
        config.use_pivot_head = False

    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    print(f"\nBuilding model: {model_name} ({model_cls.__name__})...")
    model = model_cls(config)
    model.setup_backbone()
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total_params:,}  Trainable: {trainable:,}  Frozen: {total_params - trainable:,}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) // config.gradient_accumulation * config.max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=max(1, total_steps),
        pct_start=config.warmup_steps / max(1, total_steps),
        anneal_strategy="cos",
    )

    loss_fn = GuardLensLoss(config)
    loss_fn.set_pos_weight(config.pos_weight)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Training loop ----
    best_det_score = 0.0  # Best detection (classification F1)
    best_attr_score = 0.0  # Best attribution (attr F1)
    best_threshold = config.default_threshold
    patience_counter = 0
    last_phase = 1

    print(f"\nTraining {config.max_epochs} epochs:")
    print(f"  Phase 1 (cls):   0-{config.phase1_epochs - 1}")
    print(f"  Phase 2 (+attr): {config.phase1_epochs}-{config.phase1_epochs + config.phase2_epochs - 1}")
    if config.phase3_epochs > 0:
        print(f"  Phase 3 (+cf):   {config.phase1_epochs + config.phase2_epochs}-{config.max_epochs - 1}")

    for epoch in range(config.max_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, config, epoch, device,
        )

        if (epoch + 1) % config.eval_every == 0:
            # Tune threshold on dev set
            if config.tune_threshold:
                # Quick pass to get probs
                model.eval()
                dev_probs, dev_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        out = model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            turn_mask=batch["turn_mask"].to(device),
                            role_ids=batch["role_ids"].to(device),
                            compute_attribution=False,
                        )
                        dev_probs.extend(torch.sigmoid(out["cls_logits"]).cpu().tolist())
                        dev_labels.extend(batch["labels"].tolist())
                best_threshold = find_best_threshold(dev_probs, dev_labels)

            val_metrics = evaluate(
                model, val_loader, loss_fn, config, device,
                threshold=best_threshold,
            )

            phase = train_metrics["phase"]
            print(
                f"Ep {epoch:3d} P{phase} | "
                f"loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f} | "
                f"val F1 {val_metrics['f1']:.3f} acc {val_metrics['accuracy']:.3f} | "
                f"attr F1 {val_metrics['attr_f1']:.3f} pivot {val_metrics['pivot_accuracy']:.3f} | "
                f"thr {best_threshold:.2f}"
            )

            if val_metrics.get("transfer_tier_accuracy"):
                parts = [f"{k}={v:.3f}" for k, v in val_metrics["transfer_tier_accuracy"].items()]
                print(f"       tiers: {' '.join(parts)}")

            # Reset patience on phase change
            if phase != last_phase:
                patience_counter = 0
                last_phase = phase

            # Save best detection checkpoint
            det_score = val_metrics["f1"]
            if det_score > best_det_score:
                best_det_score = det_score
                torch.save({
                    "epoch": epoch, "phase": phase,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                    "threshold": best_threshold,
                    "score": det_score,
                }, os.path.join(output_dir, "best_detection.pt"))
                print(f"       saved best_detection.pt (F1={det_score:.4f})")

            # Save best attribution checkpoint (phase 2+)
            if phase >= 2:
                attr_score = val_metrics["attr_f1"]
                if attr_score > best_attr_score:
                    best_attr_score = attr_score
                    patience_counter = 0
                    torch.save({
                        "epoch": epoch, "phase": phase,
                        "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "val_metrics": val_metrics,
                        "threshold": best_threshold,
                        "score": attr_score,
                    }, os.path.join(output_dir, "best_attribution.pt"))
                    print(f"       saved best_attribution.pt (attrF1={attr_score:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"       early stop (patience={config.patience})")
                        break

    # ---- Final test evaluation ----
    # Use attribution checkpoint (primary contribution)
    best_ckpt = os.path.join(output_dir, "best_attribution.pt")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(output_dir, "best_detection.pt")
    if not os.path.exists(best_ckpt):
        print("ERROR: No checkpoint found!")
        return {}

    import shutil
    shutil.copy2(best_ckpt, os.path.join(output_dir, "best.pt"))

    print("\n" + "=" * 60)
    print(f"  Test evaluation ({model_name})")
    print(f"  Checkpoint: {os.path.basename(best_ckpt)}")
    print("=" * 60)

    ckpt = torch.load(best_ckpt, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_threshold = ckpt.get("threshold", 0.5)
    print(f"  Loaded from epoch {ckpt['epoch']}, phase {ckpt['phase']}, threshold {test_threshold:.2f}")

    test_metrics = evaluate(model, test_loader, loss_fn, config, device, threshold=test_threshold)

    print(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  Precision:      {test_metrics['precision']:.4f}")
    print(f"  Recall:         {test_metrics['recall']:.4f}")
    print(f"  F1:             {test_metrics['f1']:.4f}")
    print(f"  Attr F1:        {test_metrics['attr_f1']:.4f}")
    print(f"  Pivot Accuracy: {test_metrics['pivot_accuracy']:.4f}")

    if test_metrics.get("transfer_tier_accuracy"):
        print("  By transfer tier:")
        for k, v in test_metrics["transfer_tier_accuracy"].items():
            print(f"    {k}: {v:.4f}")

    if test_metrics.get("benign_accuracy"):
        print("  By benign status:")
        for k, v in test_metrics["benign_accuracy"].items():
            print(f"    {k}: {v:.4f}")

    if test_metrics.get("supervision_tier_accuracy"):
        print("  By supervision tier:")
        for k, v in test_metrics["supervision_tier_accuracy"].items():
            print(f"    {k}: {v:.4f}")

    if test_metrics.get("family_accuracy"):
        print("  By family:")
        for k, v in sorted(test_metrics["family_accuracy"].items(), key=lambda x: x[1]):
            print(f"    {k}: {v:.4f}")

    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(
            {k: v for k, v in test_metrics.items() if not isinstance(v, torch.Tensor)},
            f, indent=2,
        )

    return test_metrics