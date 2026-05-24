"""
guardlens/evaluation/eval_boundary_stress.py

Boundary stress test: evaluate model on hard/rejected benign records
that were excluded from training.

Tests specificity on records that are structurally close to adversarial
but should be classified as benign.

Sources:
  - old_benign_boundary_or_unused.jsonl (258 rejected benign twins)
  - benign_boundary.jsonl (279 benign pool rejected by Llama/Mistral)

Usage:
    python -m guardlens.evaluation.eval_boundary_stress \
        --boundary-files boundary1.jsonl boundary2.jsonl \
        --checkpoint checkpoints/best.pt \
        --output results/boundary_stress.json
"""

import argparse
import json
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from guardlens.evaluation.eval_utils import load_jsonl


def main():
    parser = argparse.ArgumentParser(description="Boundary stress test")
    parser.add_argument("--boundary-files", nargs="+", required=True,
                        help="JSONL files with boundary/rejected benign records")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./results/boundary_stress.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")
    threshold = ckpt.get("threshold", 0.5)

    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = GuardLensCollator(tokenizer, config)

    # Load boundary records
    all_records = []
    for path in args.boundary_files:
        records = load_jsonl(path)
        for r in records:
            r["_source_file"] = os.path.basename(path)
        all_records.extend(records)
        print(f"  {path}: {len(records)} records")

    print(f"  Total boundary records: {len(all_records)}")
    print(f"  Labels: {Counter(r.get('label', -1) for r in all_records)}")

    dataset = GuardLensDataset(all_records, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    # Evaluate
    correct = 0
    total = 0
    fp = 0  # False positives (benign classified as adversarial)
    all_probs = []
    per_record_fp = []  # Per-record FP indicators for bootstrap CIs
    per_source = {}

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                turn_mask=batch["turn_mask"].to(device),
                role_ids=batch["role_ids"].to(device),
                compute_attribution=False,
            )
            probs = torch.sigmoid(outputs["cls_logits"])
            preds = (probs > threshold).long()
            labels = batch["labels"]

            for i in range(len(labels)):
                total += 1
                pred = preds[i].item()
                label = labels[i].item()
                prob = probs[i].item()
                source = batch["metadata"][i].get("family", "unknown")

                all_probs.append(prob)

                is_fp = int(pred == 1 and label == 0)
                per_record_fp.append(is_fp)

                if pred == label:
                    correct += 1
                if pred == 1 and label == 0:
                    fp += 1

                per_source.setdefault(source, {"correct": 0, "total": 0, "fp": 0})
                per_source[source]["total"] += 1
                if pred == label:
                    per_source[source]["correct"] += 1
                if pred == 1 and label == 0:
                    per_source[source]["fp"] += 1

    accuracy = correct / max(1, total)
    fp_rate = fp / max(1, total)

    # Probability stats
    mean_prob = sum(all_probs) / max(1, len(all_probs))
    sorted_probs = sorted(all_probs)
    p95_prob = sorted_probs[int(0.95 * len(sorted_probs))] if sorted_probs else 0.0
    max_prob = max(all_probs) if all_probs else 0.0

    print(f"\n  Boundary Stress Test Results:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  False positive rate: {fp_rate:.4f}")
    print(f"  Total: {total}, Correct: {correct}, FP: {fp}")
    print(f"  Mean P(adv):        {mean_prob:.4f}")
    print(f"  P95 P(adv):         {p95_prob:.4f}")
    print(f"  Max P(adv):         {max_prob:.4f}")

    print(f"\n  Per-source breakdown:")
    for source, stats in sorted(per_source.items()):
        acc = stats["correct"] / max(1, stats["total"])
        fpr = stats["fp"] / max(1, stats["total"])
        print(f"    {source:<30} acc={acc:.3f} fpr={fpr:.3f} n={stats['total']}")

    results = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy,
        "false_positive_rate": fp_rate,
        "total": total,
        "correct": correct,
        "false_positives": fp,
        "mean_p_adv": mean_prob,
        "p95_p_adv": p95_prob,
        "max_p_adv": max_prob,
        "per_source": per_source,
        "per_record_fp": per_record_fp,  # list of 0/1 per boundary record
        "per_record_prob": all_probs,     # list of P(adv) per record
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
