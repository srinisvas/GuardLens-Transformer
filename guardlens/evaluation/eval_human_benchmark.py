#!/usr/bin/env python3
"""
GuardLens Human Benchmark Evaluation: Multi-Turn Causal Hit Metrics

Runs GuardLens inference on the 100 human-annotated benchmark conversations
and computes multi-turn causal hit metrics against human annotations.

Metrics:
  1. Top-1/3/5 turn hit rate against human-marked causal turns
  2. Top-3/5 coverage of human causal turns
  3. Causal turn mass ratio (attribution on human turns / total)
  4. First causal region distance
  5. Human turn coverage (above-median attribution)
  6. Classification agreement (F1, kappa)
  7. All metrics also computed for surface-risk baseline

Usage:
  python eval_human_benchmark.py \
    --checkpoint checkpoints/guardlens/best_attribution.pt \
    --dataset /path/to/final_dataset.jsonl \
    --human-annotations /path/to/human_benchmark_annotated.jsonl \
    --output results/human_benchmark_eval.json
"""

import argparse
import json
import sys
import os
import numpy as np
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ─── project imports (PYTHONPATH must include GuardLens-Transformer root) ───
from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models.guardlens import GuardLens
from transformers import AutoTokenizer

# ─── surface-risk keywords (try canonical import, fallback to inline) ───
try:
    from guardlens.evaluation.causal_eval import RISK_KEYWORDS
except ImportError:
    RISK_KEYWORDS = [
        "hack", "exploit", "vulnerability", "attack", "malware", "inject",
        "bypass", "credential", "phishing", "brute force", "backdoor",
        "rootkit", "payload", "shell", "trojan", "ransomware", "keylogger",
        "spoof", "exfiltrate", "privilege escalation", "zero-day",
        "bomb", "weapon", "poison", "kill", "murder", "drug", "cocaine",
        "heroin", "methamphetamine", "fentanyl", "explosive", "detonate",
        "terroris", "bioweapon", "nerve agent", "ricin", "anthrax",
        "manipulat", "deceiv", "scam", "fraud", "launder", "counterfeit",
        "blackmail", "extort", "stalk", "harass", "threaten", "intimidat",
        "suicide", "self-harm", "harm", "abuse", "traffick", "smuggl",
    ]


# ═══════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════

def load_human_annotations(path):
    """Load human benchmark annotations → {conversation_id: annotation_dict}."""
    annotations = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            annotations[r["conversation_id"]] = {
                "is_adversarial": r["is_adversarial"],
                "human_pivot_turns": r.get("human_pivot_turns", []),
                "causal_trigger_spans": r.get("causal_trigger_spans", []),
                "confidence": r.get("confidence", "unknown"),
            }
    return annotations


def load_dataset_records(dataset_path, target_ids):
    """Load full conversation records from final_dataset.jsonl for target IDs."""
    records = {}
    with open(dataset_path) as f:
        for line in f:
            r = json.loads(line)
            cid = r.get("conversation_id")
            if cid in target_ids:
                records[cid] = r
    return records


# ═══════════════════════════════════════════════════════════════════
#  Turn-level attribution extraction
# ═══════════════════════════════════════════════════════════════════

def extract_turn_attribution(model, record, config, tokenizer, collator, device):
    """
    Run GuardLens on a single record, return turn-level attribution scores.

    Goes through GuardLensDataset → GuardLensCollator → model forward,
    matching the exact pipeline used in training and evaluation.

    Returns dict:
        cls_prob:    float — P(adversarial)
        turn_scores: {user_turn_index: mean_token_attribution}  (only user turns)
        pivot_pred:  int or None
    """
    # Wrap in dataset to get the __getitem__ preprocessing
    ds = GuardLensDataset([record], config)
    item = ds[0]

    # Collate single item → batch tensors
    batch = collator([item])

    # Move tensors to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            compute_attribution=True,
        )

    # ── Classification probability ──
    cls_prob = torch.sigmoid(outputs["cls_logits"]).item()

    # ── Token attribution → turn-level aggregation ──
    # attr_probs: [1, T, S] — per-token attribution probability
    attr_probs = outputs["attr_probs"][0]   # [T, S]
    attn_mask = batch["attention_mask"][0]   # [T, S]

    turns = record.get("turns", [])
    n_turns = min(attr_probs.shape[0], len(turns), config.max_turns)

    turn_scores = {}
    for t_idx in range(n_turns):
        role = turns[t_idx].get("role", turns[t_idx].get("speaker", ""))
        if role != "user":
            continue

        # Only score valid (non-padding) tokens
        mask = attn_mask[t_idx].bool()
        scores = attr_probs[t_idx][mask].cpu().numpy()

        if len(scores) == 0:
            turn_scores[t_idx] = 0.0
        else:
            turn_scores[t_idx] = float(np.mean(scores))

    # ── Pivot prediction ──
    pivot_pred = None
    if "pivot_logits" in outputs:
        pivot_logits = outputs["pivot_logits"][0]  # [T+1]
        pred = torch.argmax(pivot_logits).item()
        if pred < n_turns:
            pivot_pred = pred

    return {
        "cls_prob": cls_prob,
        "turn_scores": turn_scores,
        "pivot_pred": pivot_pred,
    }


def compute_surface_risk_turn_scores(record):
    """Keyword-based surface-risk score per user turn."""
    turns = record.get("turns", [])
    turn_scores = {}

    for i, turn in enumerate(turns):
        role = turn.get("role", turn.get("speaker", ""))
        if role != "user":
            continue

        text = turn.get("text", "").lower()
        hits = sum(1 for kw in RISK_KEYWORDS if kw in text)
        n_words = max(len(text.split()), 1)
        turn_scores[i] = hits / n_words

    return turn_scores


# ═══════════════════════════════════════════════════════════════════
#  Multi-turn causal hit metrics
# ═══════════════════════════════════════════════════════════════════

def compute_multiturn_metrics(turn_scores, human_causal_turns):
    """
    Compute multi-turn causal hit metrics for a single adversarial conversation.

    Args:
        turn_scores:        {user_turn_index: score}
        human_causal_turns: [turn_index, ...] from human annotation

    Returns dict of per-record metrics, or None if inputs are empty.
    """
    if not human_causal_turns or not turn_scores:
        return None

    human_set = set(human_causal_turns)

    # Rank user turns by attribution (descending)
    ranked = sorted(turn_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_indices = [idx for idx, _ in ranked]

    # ── Top-k hit: does any top-k predicted turn match any human turn? ──
    top1_hit = int(ranked_indices[0] in human_set) if ranked_indices else 0
    top3_hit = int(any(idx in human_set for idx in ranked_indices[:3]))
    top5_hit = int(any(idx in human_set for idx in ranked_indices[:5]))

    # ── Top-k coverage: fraction of human turns captured in top-k ──
    top3_coverage = len(set(ranked_indices[:3]) & human_set) / len(human_set)
    top5_coverage = len(set(ranked_indices[:5]) & human_set) / len(human_set)

    # ── Causal mass ratio ──
    total_mass = sum(turn_scores.values())
    causal_mass = sum(turn_scores.get(t, 0) for t in human_causal_turns)
    mass_ratio = causal_mass / total_mass if total_mass > 0 else 0.0

    # ── First causal region distance ──
    if ranked_indices:
        min_dist = min(abs(ranked_indices[0] - ht) for ht in human_causal_turns)
    else:
        min_dist = float("inf")

    # ── Human turn coverage: fraction with above-median attribution ──
    if turn_scores:
        median_score = float(np.median(list(turn_scores.values())))
        above_median = sum(
            1 for t in human_causal_turns if turn_scores.get(t, 0) > median_score
        )
        human_coverage = above_median / len(human_causal_turns)
    else:
        human_coverage = 0.0

    return {
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "top5_hit": top5_hit,
        "top3_coverage": top3_coverage,
        "top5_coverage": top5_coverage,
        "causal_mass_ratio": mass_ratio,
        "first_causal_distance": min_dist,
        "human_coverage": human_coverage,
        "n_human_causal_turns": len(human_causal_turns),
        "n_user_turns": len(turn_scores),
    }


# ═══════════════════════════════════════════════════════════════════
#  Aggregation and reporting
# ═══════════════════════════════════════════════════════════════════

def aggregate_metrics(metrics_list, name, file=sys.stderr):
    """Aggregate per-record metrics into means, print summary."""
    if not metrics_list:
        print(f"\n  {name}: No valid records", file=file)
        return {}

    n = len(metrics_list)
    agg = {}
    for key in [
        "top1_hit", "top3_hit", "top5_hit",
        "top3_coverage", "top5_coverage",
        "causal_mass_ratio", "first_causal_distance", "human_coverage",
    ]:
        vals = [m[key] for m in metrics_list]
        agg[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }

    agg["n"] = n
    agg["mean_human_causal_turns"] = float(
        np.mean([m["n_human_causal_turns"] for m in metrics_list])
    )

    print(f"\n  {name} (n={n}):", file=file)
    print(f"    Top-1 turn hit:         {agg['top1_hit']['mean']:.3f}", file=file)
    print(f"    Top-3 turn hit:         {agg['top3_hit']['mean']:.3f}", file=file)
    print(f"    Top-5 turn hit:         {agg['top5_hit']['mean']:.3f}", file=file)
    print(f"    Top-3 coverage:         {agg['top3_coverage']['mean']:.3f}", file=file)
    print(f"    Top-5 coverage:         {agg['top5_coverage']['mean']:.3f}", file=file)
    print(f"    Causal mass ratio:      {agg['causal_mass_ratio']['mean']:.3f}", file=file)
    print(
        f"    First causal distance:  {agg['first_causal_distance']['mean']:.1f} "
        f"(median {agg['first_causal_distance']['median']:.1f})",
        file=file,
    )
    print(f"    Human turn coverage:    {agg['human_coverage']['mean']:.3f}", file=file)
    print(
        f"    Mean human causal turns: {agg['mean_human_causal_turns']:.1f}",
        file=file,
    )
    return agg


def compute_classification_agreement(classification_results, file=sys.stderr):
    """Compute F1, kappa, confusion matrix from classification predictions."""
    tp = sum(1 for r in classification_results.values() if r["pred"] and r["human"])
    tn = sum(1 for r in classification_results.values() if not r["pred"] and not r["human"])
    fp = sum(1 for r in classification_results.values() if r["pred"] and not r["human"])
    fn = sum(1 for r in classification_results.values() if not r["pred"] and r["human"])
    n = len(classification_results)

    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    # Cohen's kappa
    p_pred = (tp + fp) / n if n else 0
    p_human = (tp + fn) / n if n else 0
    pe = p_pred * p_human + (1 - p_pred) * (1 - p_human)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 1.0

    results = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "kappa": kappa,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
    }

    print(f"\n  Classification Agreement:", file=file)
    print(f"    Accuracy:  {acc:.3f}", file=file)
    print(f"    Precision: {prec:.3f}", file=file)
    print(f"    Recall:    {rec:.3f}", file=file)
    print(f"    F1:        {f1:.3f}", file=file)
    print(f"    Kappa:     {kappa:.3f}", file=file)
    print(f"    TP={tp} TN={tn} FP={fp} FN={fn}", file=file)
    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Human Benchmark Multi-Turn Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, help="final_dataset.jsonl")
    parser.add_argument("--human-annotations", required=True,
                        help="human_benchmark_annotated.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override classification threshold (default: from checkpoint)")
    args = parser.parse_args()

    P = sys.stderr  # all progress to stderr, stdout stays clean

    print("=" * 60, file=P)
    print("  GuardLens Human Benchmark Evaluation", file=P)
    print("=" * 60, file=P)

    # ── 1. Load human annotations ──
    print(f"\nLoading annotations from {args.human_annotations}...", file=P)
    annotations = load_human_annotations(args.human_annotations)
    adv_ann = {k: v for k, v in annotations.items() if v["is_adversarial"]}
    ben_ann = {k: v for k, v in annotations.items() if not v["is_adversarial"]}
    print(f"  {len(annotations)} total: {len(adv_ann)} adversarial, {len(ben_ann)} benign", file=P)

    # ── 2. Load matching records from dataset ──
    print(f"Loading records from {args.dataset}...", file=P)
    records = load_dataset_records(args.dataset, set(annotations.keys()))
    print(f"  Matched {len(records)}/{len(annotations)} conversations", file=P)

    missing = set(annotations.keys()) - set(records.keys())
    if missing:
        print(f"  WARNING: {len(missing)} IDs not found in dataset:", file=P)
        for mid in sorted(missing)[:10]:
            print(f"    {mid}", file=P)

    # ── 3. Load model ──
    print(f"\nLoading checkpoint {args.checkpoint}...", file=P)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    threshold = args.threshold or ckpt.get("threshold", 0.5)
    epoch = ckpt.get("epoch", "?")
    phase = ckpt.get("phase", "?")
    print(f"  Epoch {epoch}, phase {phase}, threshold {threshold}", file=P)

    config = GuardLensConfig()
    # Restore any saved config overrides
    saved_config = ckpt.get("config", {})
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)

    model = GuardLens(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    backbone = getattr(config, "backbone", "microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    collator = GuardLensCollator(tokenizer=tokenizer, config=config)
    print(f"  Model loaded on {device}", file=P)

    # ── 4. Run inference ──
    print(f"\nRunning inference on {len(records)} conversations...", file=P)

    gl_results = {}      # cid → {cls_prob, turn_scores, pivot_pred}
    sr_results = {}      # cid → {turn_idx: score}
    cls_results = {}     # cid → {pred, human, prob}

    for i, (cid, record) in enumerate(sorted(records.items())):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}...", file=P)

        try:
            # GuardLens
            gl = extract_turn_attribution(model, record, config, tokenizer, collator, device)
            gl_results[cid] = gl

            # Classification
            cls_results[cid] = {
                "pred": gl["cls_prob"] >= threshold,
                "human": annotations[cid]["is_adversarial"],
                "prob": gl["cls_prob"],
            }

            # Surface-risk
            sr_results[cid] = compute_surface_risk_turn_scores(record)

        except Exception as e:
            print(f"  ERROR {cid[:12]}: {e}", file=P)
            import traceback; traceback.print_exc(file=P)

    print(f"  Done: {len(gl_results)} successful", file=P)

    # ── 5. Compute multi-turn metrics (adversarial only) ──
    print(f"\n{'='*60}", file=P)
    print(f"  RESULTS", file=P)
    print(f"{'='*60}", file=P)

    gl_metrics = []
    sr_metrics = []

    for cid in sorted(adv_ann.keys()):
        if cid not in gl_results:
            continue

        human_turns = annotations[cid]["human_pivot_turns"]
        if not human_turns:
            continue

        gl_m = compute_multiturn_metrics(gl_results[cid]["turn_scores"], human_turns)
        if gl_m:
            gl_m["conversation_id"] = cid
            gl_metrics.append(gl_m)

        sr_m = compute_multiturn_metrics(sr_results.get(cid, {}), human_turns)
        if sr_m:
            sr_m["conversation_id"] = cid
            sr_metrics.append(sr_m)

    gl_agg = aggregate_metrics(gl_metrics, "GuardLens", file=P)
    sr_agg = aggregate_metrics(sr_metrics, "Surface-Risk", file=P)

    # ── 6. Classification agreement ──
    cls_agg = compute_classification_agreement(cls_results, file=P)
    cls_agg["threshold"] = threshold

    # ── 7. Human annotation stats ──
    pivot_counts = [len(a["human_pivot_turns"]) for a in adv_ann.values()]
    multi_pivot = sum(1 for c in pivot_counts if c > 1)
    human_stats = {
        "n_total": len(annotations),
        "n_adversarial": len(adv_ann),
        "n_benign": len(ben_ann),
        "multi_pivot_rate": multi_pivot / len(adv_ann) if adv_ann else 0,
        "mean_causal_turns": float(np.mean(pivot_counts)) if pivot_counts else 0,
        "median_causal_turns": float(np.median(pivot_counts)) if pivot_counts else 0,
        "pivot_count_distribution": dict(Counter(pivot_counts)),
    }

    print(f"\n  Human Annotation Stats:", file=P)
    print(f"    Multi-pivot rate:  {human_stats['multi_pivot_rate']:.1%}", file=P)
    print(f"    Mean causal turns: {human_stats['mean_causal_turns']:.1f}", file=P)
    print(f"    Distribution:      {human_stats['pivot_count_distribution']}", file=P)

    # ── 8. Save ──
    output = {
        "guardlens_multiturn": gl_agg,
        "surface_risk_multiturn": sr_agg,
        "classification": cls_agg,
        "human_annotation_stats": human_stats,
        "per_record_guardlens": gl_metrics,
        "per_record_surface_risk": sr_metrics,
        "per_record_classification": {k: v for k, v in cls_results.items()},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved → {args.output}", file=P)
    print("=" * 60, file=P)


if __name__ == "__main__":
    main()