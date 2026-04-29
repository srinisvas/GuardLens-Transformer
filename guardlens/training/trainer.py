"""Training and evaluation loops."""

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.data.splits import pair_aware_split
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.schedule import get_current_phase, get_lambda_schedule


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
    lambda_attr, lambda_cf = get_lambda_schedule(epoch, config)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_attr_loss = 0.0
    total_cf_loss = 0.0
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
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            turn_mask=turn_mask,
            role_ids=role_ids,
            compute_attribution=(phase >= 2),
        )

        losses = loss_fn(
            outputs, labels, token_labels,
            phase=phase, lambda_attr=lambda_attr, lambda_cf=lambda_cf,
        )
        loss = losses["total"]

        # Counterfactual loss (phase 3)
        if phase >= 3 and lambda_cf > 0:
            # Compute progress through phase 3 for curriculum threshold
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

        total_loss += losses["total"].item()
        total_cls_loss += losses["cls"].item()
        if "attr" in losses:
            total_attr_loss += losses["attr"].item()

        preds = (torch.sigmoid(outputs["cls_logits"]) > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "cls_loss": total_cls_loss / max(1, n_batches),
        "attr_loss": total_attr_loss / max(1, n_batches),
        "cf_loss": total_cf_loss / max(1, n_batches),
        "accuracy": correct / max(1, total),
        "phase": phase,
        "lambda_attr": lambda_attr,
        "lambda_cf": lambda_cf,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: GuardLensLoss,
    config: GuardLensConfig,
    device: torch.device,
) -> Dict:
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_difficulties = []
    all_families = []
    total_loss = 0.0
    n_batches = 0

    attr_tp = attr_fp = attr_fn = attr_tn = 0

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

        losses = loss_fn(outputs, labels, token_labels, phase=3)
        total_loss += losses["total"].item()
        n_batches += 1

        probs = torch.sigmoid(outputs["cls_logits"])
        preds = (probs > 0.5).long()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        for m in batch["metadata"]:
            all_difficulties.append(m["difficulty"])
            all_families.append(m["family"])

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

    # Per-difficulty accuracy
    diff_acc = {}
    for diff in ["easy", "medium", "hard"]:
        idx = [i for i, d in enumerate(all_difficulties) if d == diff]
        if idx:
            c = sum(1 for i in idx if all_preds[i] == all_labels[i])
            diff_acc[diff] = c / len(idx)

    # Per-family accuracy
    fam_acc = {}
    for fam in set(all_families):
        idx = [i for i, f in enumerate(all_families) if f == fam]
        if len(idx) >= 5:
            c = sum(1 for i in idx if all_preds[i] == all_labels[i])
            fam_acc[fam] = round(c / len(idx), 3)

    # Attribution metrics
    attr_precision = attr_tp / max(1, attr_tp + attr_fp)
    attr_recall = attr_tp / max(1, attr_tp + attr_fn)
    attr_f1 = 2 * attr_precision * attr_recall / max(1e-8, attr_precision + attr_recall)

    return {
        "loss": total_loss / max(1, n_batches),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "difficulty_accuracy": diff_acc,
        "family_accuracy": fam_acc,
        "attr_precision": attr_precision,
        "attr_recall": attr_recall,
        "attr_f1": attr_f1,
    }


def train(
    config: GuardLensConfig,
    data_path: str,
    output_dir: str,
    model_name: str = "guardlens",
):
    """Full training pipeline."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {data_path}...")
    records = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"  {len(records)} records")

    # Split
    train_idx, val_idx, test_idx = pair_aware_split(records, seed=config.seed)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Dataset + loaders
    full_dataset = GuardLensDataset(records, config)

    from transformers import AutoTokenizer
    from guardlens.data.dataset import FlatConversationCollator
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    # Select collator based on model type
    if model_name == "conversation_deberta":
        collator = FlatConversationCollator(tokenizer, config)
    else:
        collator = GuardLensCollator(tokenizer, config)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=config.batch_size, shuffle=True,
        collate_fn=collator, num_workers=config.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=config.batch_size * 2,
        collate_fn=collator, num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=config.batch_size * 2,
        collate_fn=collator, num_workers=config.num_workers,
    )

    # Model
    if model_name == "guardlens_no_cf":
        config.phase3_epochs = 0
        config.max_epochs = config.phase1_epochs + config.phase2_epochs

    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    print(f"Building model: {model_name} ({model_cls.__name__})...")
    model = model_cls(config)
    model.setup_backbone()
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total_params:,}  Trainable: {trainable:,}  Frozen: {total_params - trainable:,}")

    # Optimizer + scheduler
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
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    # Track best checkpoint per phase. Phase 1 optimizes classification only.
    # Phase 2+ optimizes a composite of classification F1 and attribution F1,
    # because the paper's contribution is attribution, not just classification.
    best_scores = {1: 0.0, 2: 0.0, 3: 0.0}
    patience_counter = 0
    last_phase = 1

    print(f"\nTraining {config.max_epochs} epochs:")
    print(f"  Phase 1 (cls):  0-{config.phase1_epochs-1}")
    print(f"  Phase 2 (+attr): {config.phase1_epochs}-{config.phase1_epochs+config.phase2_epochs-1}")
    if config.phase3_epochs > 0:
        print(f"  Phase 3 (+cf):  {config.phase1_epochs+config.phase2_epochs}-{config.max_epochs-1}")

    for epoch in range(config.max_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, config, epoch, device,
        )

        if (epoch + 1) % config.eval_every == 0:
            val_metrics = evaluate(model, val_loader, loss_fn, config, device)

            phase = train_metrics["phase"]
            print(
                f"Ep {epoch:3d} P{phase} | "
                f"loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f} | "
                f"val F1 {val_metrics['f1']:.3f} acc {val_metrics['accuracy']:.3f} | "
                f"attr F1 {val_metrics['attr_f1']:.3f}"
            )

            if val_metrics.get("difficulty_accuracy"):
                parts = [f"{k}={v:.3f}" for k, v in val_metrics["difficulty_accuracy"].items()]
                print(f"       diff: {' '.join(parts)}")

            # Reset patience when entering a new phase
            if phase != last_phase:
                patience_counter = 0
                last_phase = phase

            # Compute composite score based on phase
            if phase == 1:
                # Phase 1: pure classification
                composite = val_metrics["f1"]
            else:
                # Phase 2+: 60% classification F1 + 40% attribution F1
                # Attribution is the novel contribution and must be optimized
                composite = 0.6 * val_metrics["f1"] + 0.4 * val_metrics["attr_f1"]

            # Save best per phase
            ckpt_name = f"best_phase{phase}.pt"
            if composite > best_scores[phase]:
                best_scores[phase] = composite
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "phase": phase,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                    "composite_score": composite,
                }, os.path.join(output_dir, ckpt_name))
                print(f"       saved {ckpt_name} (composite={composite:.4f}, "
                      f"F1={val_metrics['f1']:.4f}, attrF1={val_metrics['attr_f1']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.patience and phase >= 2:
                    print(f"       early stop (patience={config.patience})")
                    break

    # Select the best checkpoint for final evaluation:
    # Prefer the highest phase checkpoint that exists (phase 3 > 2 > 1)
    # because later phases have trained attribution heads.
    best_ckpt_path = None
    for p in [3, 2, 1]:
        candidate = os.path.join(output_dir, f"best_phase{p}.pt")
        if os.path.exists(candidate):
            best_ckpt_path = candidate
            break

    if best_ckpt_path is None:
        print("ERROR: No checkpoint found!")
        return {}

    # Also save as best.pt for convenience
    import shutil
    shutil.copy2(best_ckpt_path, os.path.join(output_dir, "best.pt"))

    # Final test with the attribution-trained checkpoint
    print("\n" + "=" * 60)
    print(f"  Test evaluation ({model_name})")
    print(f"  Checkpoint: {os.path.basename(best_ckpt_path)}")
    print("=" * 60)

    ckpt = torch.load(best_ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded from epoch {ckpt['epoch']}, phase {ckpt['phase']}")
    test_metrics = evaluate(model, test_loader, loss_fn, config, device)

    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Attr F1:   {test_metrics['attr_f1']:.4f}")

    if test_metrics.get("difficulty_accuracy"):
        print("  By difficulty:")
        for k, v in test_metrics["difficulty_accuracy"].items():
            print(f"    {k}: {v:.4f}")

    if test_metrics.get("family_accuracy"):
        print("  By family:")
        for k, v in sorted(test_metrics["family_accuracy"].items(), key=lambda x: x[1]):
            print(f"    {k}: {v:.4f}")

    # Clean vs augmented evaluation (paper requirement)
    test_records = [records[i] for i in test_idx]
    clean_test_idx = [
        i for i, r in enumerate(test_records)
        if r.get("metadata", {}).get("clean_holdout")
        or not r.get("metadata", {}).get("augmentation_applied")
    ]
    aug_test_idx = [
        i for i, r in enumerate(test_records)
        if r.get("metadata", {}).get("augmentation_applied")
    ]

    test_dataset_full = Subset(full_dataset, test_idx)
    if clean_test_idx and aug_test_idx:
        print("\n  Clean vs augmented split:")

        clean_loader = DataLoader(
            Subset(test_dataset_full, clean_test_idx),
            batch_size=config.batch_size * 2,
            collate_fn=collator, num_workers=config.num_workers,
        )
        aug_loader = DataLoader(
            Subset(test_dataset_full, aug_test_idx),
            batch_size=config.batch_size * 2,
            collate_fn=collator, num_workers=config.num_workers,
        )

        clean_metrics = evaluate(model, clean_loader, loss_fn, config, device)
        aug_metrics = evaluate(model, aug_loader, loss_fn, config, device)

        print(f"    Clean ({len(clean_test_idx)} samples):     "
              f"F1={clean_metrics['f1']:.4f}  Acc={clean_metrics['accuracy']:.4f}  "
              f"AttrF1={clean_metrics['attr_f1']:.4f}")
        print(f"    Augmented ({len(aug_test_idx)} samples):  "
              f"F1={aug_metrics['f1']:.4f}  Acc={aug_metrics['accuracy']:.4f}  "
              f"AttrF1={aug_metrics['attr_f1']:.4f}")

        test_metrics["clean_split"] = {
            "n": len(clean_test_idx),
            "f1": clean_metrics["f1"],
            "accuracy": clean_metrics["accuracy"],
            "attr_f1": clean_metrics["attr_f1"],
        }
        test_metrics["augmented_split"] = {
            "n": len(aug_test_idx),
            "f1": aug_metrics["f1"],
            "accuracy": aug_metrics["accuracy"],
            "attr_f1": aug_metrics["attr_f1"],
        }

    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump({k: v for k, v in test_metrics.items() if not isinstance(v, torch.Tensor)}, f, indent=2)

    return test_metrics
