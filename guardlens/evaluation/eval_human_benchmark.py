#!/usr/bin/env python3
"""
GuardLens Human Benchmark Evaluation: Multi-Turn Causal Hit Metrics

Runs GuardLens on human-annotated benchmark conversations and computes
multi-turn causal hit metrics. Supports multiple annotator files to
compute inter-annotator agreement and annotator-stratified metrics.

Usage:
    python eval_human_benchmark.py \
        --checkpoint checkpoints/guardlens/best_attribution.pt \
        --dataset final_dataset.jsonl \
        --human-annotations ann_A.jsonl ann_B.jsonl \
        --annotator-names "Annotator_A" "Annotator_B" \
        --output results/human_benchmark_eval.json
"""

import argparse
import json
import sys
import os
import numpy as np
from collections import Counter

import torch

from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from transformers import AutoTokenizer

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
    records = {}
    with open(dataset_path) as f:
        for line in f:
            r = json.loads(line)
            cid = r.get("conversation_id")
            if cid in target_ids:
                records[cid] = r
    return records


# ═══════════════════════════════════════════════════════════════════
#  Turn-level attribution
# ═══════════════════════════════════════════════════════════════════

def extract_turn_attribution(model, record, config, tokenizer, collator, device):
    ds = GuardLensDataset([record], config)
    item = ds[0]
    batch = collator([item])

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

    cls_prob = torch.sigmoid(outputs["cls_logits"]).item()

    attr_probs = outputs["attr_probs"][0]
    attn_mask = batch["attention_mask"][0]

    turns = record.get("turns", [])
    n_turns = min(attr_probs.shape[0], len(turns), config.max_turns)

    turn_scores = {}
    for t_idx in range(n_turns):
        role = turns[t_idx].get("role", turns[t_idx].get("speaker", ""))
        if role != "user":
            continue
        mask = attn_mask[t_idx].bool()
        scores = attr_probs[t_idx][mask].cpu().numpy()
        turn_scores[t_idx] = float(np.mean(scores)) if len(scores) > 0 else 0.0

    pivot_pred = None
    if "pivot_logits" in outputs:
        pred = torch.argmax(outputs["pivot_logits"][0]).item()
        if pred < n_turns:
            pivot_pred = pred

    return {"cls_prob": cls_prob, "turn_scores": turn_scores, "pivot_pred": pivot_pred}


def compute_surface_risk_turn_scores(record):
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
#  Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_multiturn_metrics(turn_scores, human_causal_turns):
    if not human_causal_turns or not turn_scores:
        return None

    human_set = set(human_causal_turns)
    ranked = sorted(turn_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_indices = [idx for idx, _ in ranked]

    top1_hit = int(ranked_indices[0] in human_set) if ranked_indices else 0
    top3_hit = int(any(idx in human_set for idx in ranked_indices[:3]))
    top5_hit = int(any(idx in human_set for idx in ranked_indices[:5]))

    top3_coverage = len(set(ranked_indices[:3]) & human_set) / len(human_set)
    top5_coverage = len(set(ranked_indices[:5]) & human_set) / len(human_set)

    total_mass = sum(turn_scores.values())
    causal_mass = sum(turn_scores.get(t, 0) for t in human_causal_turns)
    mass_ratio = causal_mass / total_mass if total_mass > 0 else 0.0

    min_dist = min(abs(ranked_indices[0] - ht) for ht in human_causal_turns) if ranked_indices else float("inf")

    if turn_scores:
        median_score = float(np.median(list(turn_scores.values())))
        above_median = sum(1 for t in human_causal_turns if turn_scores.get(t, 0) > median_score)
        human_coverage = above_median / len(human_causal_turns)
    else:
        human_coverage = 0.0

    return {
        "top1_hit": top1_hit, "top3_hit": top3_hit, "top5_hit": top5_hit,
        "top3_coverage": top3_coverage, "top5_coverage": top5_coverage,
        "causal_mass_ratio": mass_ratio, "first_causal_distance": min_dist,
        "human_coverage": human_coverage,
        "n_human_causal_turns": len(human_causal_turns),
        "n_user_turns": len(turn_scores),
    }


def aggregate_metrics(metrics_list, name, file=sys.stderr):
    if not metrics_list:
        print(f"\n  {name}: No valid records", file=file)
        return {}

    n = len(metrics_list)
    agg = {}
    for key in ["top1_hit", "top3_hit", "top5_hit", "top3_coverage", "top5_coverage",
                 "causal_mass_ratio", "first_causal_distance", "human_coverage"]:
        vals = [m[key] for m in metrics_list]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                     "median": float(np.median(vals))}

    agg["n"] = n
    agg["mean_human_causal_turns"] = float(np.mean([m["n_human_causal_turns"] for m in metrics_list]))

    print(f"\n  {name} (n={n}):", file=file)
    print(f"    Top-1 turn hit:         {agg['top1_hit']['mean']:.3f}", file=file)
    print(f"    Top-3 turn hit:         {agg['top3_hit']['mean']:.3f}", file=file)
    print(f"    Top-5 turn hit:         {agg['top5_hit']['mean']:.3f}", file=file)
    print(f"    Top-3 coverage:         {agg['top3_coverage']['mean']:.3f}", file=file)
    print(f"    Top-5 coverage:         {agg['top5_coverage']['mean']:.3f}", file=file)
    print(f"    Causal mass ratio:      {agg['causal_mass_ratio']['mean']:.3f}", file=file)
    print(f"    First causal distance:  {agg['first_causal_distance']['mean']:.1f} "
          f"(median {agg['first_causal_distance']['median']:.1f})", file=file)
    print(f"    Human turn coverage:    {agg['human_coverage']['mean']:.3f}", file=file)
    print(f"    Mean human causal turns: {agg['mean_human_causal_turns']:.1f}", file=file)
    return agg


# ═══════════════════════════════════════════════════════════════════
#  Inter-annotator agreement
# ═══════════════════════════════════════════════════════════════════

def compute_inter_annotator(ann_a, ann_b, name_a, name_b, file=sys.stderr):
    """Compute agreement between two annotators on shared conversations."""
    shared = sorted(set(ann_a.keys()) & set(ann_b.keys()))
    if not shared:
        return {}

    tp = sum(1 for c in shared if ann_a[c]["is_adversarial"] and ann_b[c]["is_adversarial"])
    tn = sum(1 for c in shared if not ann_a[c]["is_adversarial"] and not ann_b[c]["is_adversarial"])
    fp = sum(1 for c in shared if not ann_a[c]["is_adversarial"] and ann_b[c]["is_adversarial"])
    fn = sum(1 for c in shared if ann_a[c]["is_adversarial"] and not ann_b[c]["is_adversarial"])

    po = (tp + tn) / len(shared)
    pa = (tp + fn) / len(shared)
    pb = (tp + fp) / len(shared)
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    # Pivot agreement on agreed-adversarial
    agreed_adv = [c for c in shared if ann_a[c]["is_adversarial"] and ann_b[c]["is_adversarial"]]
    jaccards = []
    recall_a_from_b = []
    recall_b_from_a = []
    for c in agreed_adv:
        pa_set = set(ann_a[c].get("human_pivot_turns", []))
        pb_set = set(ann_b[c].get("human_pivot_turns", []))
        if pa_set or pb_set:
            jaccards.append(len(pa_set & pb_set) / len(pa_set | pb_set))
        if pa_set:
            recall_a_from_b.append(len(pa_set & pb_set) / len(pa_set))
        if pb_set:
            recall_b_from_a.append(len(pa_set & pb_set) / len(pb_set))

    result = {
        "n_shared": len(shared),
        "classification_agreement": po,
        "kappa": kappa,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_agreed_adversarial": len(agreed_adv),
        "pivot_jaccard_mean": float(np.mean(jaccards)) if jaccards else 0,
        "pivot_jaccard_median": float(np.median(jaccards)) if jaccards else 0,
        f"recall_{name_a}_in_{name_b}": float(np.mean(recall_a_from_b)) if recall_a_from_b else 0,
        f"recall_{name_b}_in_{name_a}": float(np.mean(recall_b_from_a)) if recall_b_from_a else 0,
    }

    print(f"\n  Inter-Annotator: {name_a} vs {name_b} (n={len(shared)}):", file=file)
    print(f"    Classification: agreement={po:.3f} kappa={kappa:.3f}", file=file)
    print(f"    {name_a} adv: {tp+fn}, {name_b} adv: {tp+fp}, agreed adv: {len(agreed_adv)}", file=file)
    print(f"    Pivot Jaccard:  mean={result['pivot_jaccard_mean']:.3f} median={result['pivot_jaccard_median']:.3f}", file=file)
    print(f"    Recall of {name_a} pivots in {name_b}: {result[f'recall_{name_a}_in_{name_b}']:.3f}", file=file)
    print(f"    Recall of {name_b} pivots in {name_a}: {result[f'recall_{name_b}_in_{name_a}']:.3f}", file=file)
    return result


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Human Benchmark Multi-Turn Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, help="final_dataset.jsonl")
    parser.add_argument("--human-annotations", nargs="+", required=True,
                        help="One or more annotator JSONL files")
    parser.add_argument("--annotator-names", nargs="+", default=None,
                        help="Names for annotators (default: Ann_1, Ann_2, ...)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    P = sys.stderr
    ann_names = args.annotator_names or [f"Ann_{i+1}" for i in range(len(args.human_annotations))]
    assert len(ann_names) == len(args.human_annotations), "Mismatch between annotation files and names"

    print("=" * 60, file=P)
    print("  GuardLens Human Benchmark Evaluation", file=P)
    print("=" * 60, file=P)

    # ── 1. Load all annotator files ──
    all_annotations = {}
    all_ids = set()
    for name, path in zip(ann_names, args.human_annotations):
        ann = load_human_annotations(path)
        all_annotations[name] = ann
        all_ids |= set(ann.keys())
        adv = sum(1 for v in ann.values() if v["is_adversarial"])
        print(f"  {name}: {len(ann)} records ({adv} adv, {len(ann)-adv} ben)", file=P)

    # ── 2. Load dataset records (union of all annotated IDs) ──
    records = load_dataset_records(args.dataset, all_ids)
    print(f"  Matched: {len(records)}/{len(all_ids)} conversations", file=P)

    missing = all_ids - set(records.keys())
    if missing:
        print(f"  WARNING: {len(missing)} IDs not found in dataset", file=P)

    # ── 3. Load model ──
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)

    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")
    threshold = args.threshold or ckpt.get("threshold", 0.5)

    print(f"  Model: {model_name}, threshold: {threshold:.2f}", file=P)

    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = GuardLensCollator(tokenizer, config)

    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Loaded on {device}", file=P)

    # ── 4. Single inference pass over all unique conversations ──
    print(f"\n  Inference on {len(records)} conversations...", file=P)

    gl_results = {}   # cid → {cls_prob, turn_scores, pivot_pred}
    sr_results = {}   # cid → {turn_idx: score}

    for i, (cid, record) in enumerate(sorted(records.items())):
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(records)}...", file=P)
        try:
            gl_results[cid] = extract_turn_attribution(
                model, record, config, tokenizer, collator, device
            )
            sr_results[cid] = compute_surface_risk_turn_scores(record)
        except Exception as e:
            print(f"    ERROR {cid[:12]}: {e}", file=P)
            import traceback; traceback.print_exc(file=P)

    print(f"  Done: {len(gl_results)} successful", file=P)

    # ── 5. Compute metrics per annotator ──
    output = {"model_inference": {"n_records": len(gl_results), "threshold": threshold}}

    for ann_name, annotations in all_annotations.items():
        print(f"\n{'='*60}", file=P)
        print(f"  RESULTS vs {ann_name}", file=P)
        print(f"{'='*60}", file=P)

        adv_ann = {k: v for k, v in annotations.items() if v["is_adversarial"]}

        # Multi-turn metrics
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

        gl_agg = aggregate_metrics(gl_metrics, f"GuardLens vs {ann_name}", file=P)
        sr_agg = aggregate_metrics(sr_metrics, f"Surface-Risk vs {ann_name}", file=P)

        # Classification agreement
        cls_results = {}
        for cid in sorted(annotations.keys()):
            if cid not in gl_results:
                continue
            cls_results[cid] = {
                "pred": gl_results[cid]["cls_prob"] >= threshold,
                "human": annotations[cid]["is_adversarial"],
                "prob": gl_results[cid]["cls_prob"],
            }

        tp = sum(1 for r in cls_results.values() if r["pred"] and r["human"])
        tn = sum(1 for r in cls_results.values() if not r["pred"] and not r["human"])
        fp = sum(1 for r in cls_results.values() if r["pred"] and not r["human"])
        fn = sum(1 for r in cls_results.values() if not r["pred"] and r["human"])
        n = len(cls_results)

        acc = (tp + tn) / n if n else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        p_p = (tp + fp) / n if n else 0
        p_h = (tp + fn) / n if n else 0
        pe = p_p * p_h + (1 - p_p) * (1 - p_h)
        kappa = (acc - pe) / (1 - pe) if pe < 1 else 1.0

        cls_agg = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                    "kappa": kappa, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n}

        print(f"\n  Classification vs {ann_name}:", file=P)
        print(f"    F1={f1:.3f} Acc={acc:.3f} Kappa={kappa:.3f} TP={tp} TN={tn} FP={fp} FN={fn}", file=P)

        # Human stats for this annotator
        pivot_counts = [len(a["human_pivot_turns"]) for a in adv_ann.values()]
        multi = sum(1 for c in pivot_counts if c > 1)
        human_stats = {
            "n_adversarial": len(adv_ann),
            "multi_pivot_rate": multi / len(adv_ann) if adv_ann else 0,
            "mean_causal_turns": float(np.mean(pivot_counts)) if pivot_counts else 0,
            "pivot_count_distribution": dict(Counter(pivot_counts)),
        }

        output[ann_name] = {
            "guardlens_multiturn": gl_agg,
            "surface_risk_multiturn": sr_agg,
            "classification": cls_agg,
            "human_stats": human_stats,
            "per_record_guardlens": gl_metrics,
            "per_record_surface_risk": sr_metrics,
        }

    # ── 6. Inter-annotator agreement (all pairs) ──
    if len(all_annotations) > 1:
        print(f"\n{'='*60}", file=P)
        print(f"  INTER-ANNOTATOR AGREEMENT", file=P)
        print(f"{'='*60}", file=P)

        inter_ann = {}
        names = list(all_annotations.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{names[i]}_vs_{names[j]}"
                inter_ann[key] = compute_inter_annotator(
                    all_annotations[names[i]], all_annotations[names[j]],
                    names[i], names[j], file=P,
                )
        output["inter_annotator"] = inter_ann

    # ── 7. Save ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved → {args.output}", file=P)
    print("=" * 60, file=P)


if __name__ == "__main__":
    main()