"""
guardlens/evaluation/eval_utility_grid.py

Attribution Utility sensitivity analysis over k and lambda.

Reviewer i7qz: "report Utility over a small grid of k and lambda
values, or at least provide the range of lambda for which the
proposed method remains better than surface-risk."

Reports:
  - Utility(k, lambda) for each method at each (k, lambda) pair
  - The crossover lambda where surface-risk overtakes GuardLens
  - Winner matrix showing which method wins at each setting

Usage:
    python -m guardlens.evaluation.eval_utility_grid \
        --test-path splits/test.jsonl \
        --checkpoint checkpoints/guardlens/best_attribution.pt \
        --boundary-files benign_boundary.jsonl \
        --output results/utility_grid.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from guardlens.evaluation.causal_eval import (
    run_causal_evaluation,
)
from guardlens.evaluation.eval_utils import load_jsonl


def compute_utility(
    dd: float,
    boundary_fpr: float,
    lam: float,
) -> float:
    """Utility = DD - lambda * boundary_FPR."""
    return dd - lam * boundary_fpr


def run_boundary_fpr(
    model,
    boundary_records: List[Dict],
    config,
    tokenizer,
    device,
    threshold: float,
    batch_size: int = 8,
) -> float:
    """Compute FPR on boundary/high-risk benign records."""
    collator = GuardLensCollator(tokenizer, config)
    dataset = GuardLensDataset(boundary_records, config)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    fp = 0
    total = 0
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
                if preds[i].item() == 1 and labels[i].item() == 0:
                    fp += 1

    return fp / max(1, total)


def main():
    parser = argparse.ArgumentParser(
        description="Attribution Utility sensitivity grid",
    )
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--boundary-files", nargs="+", default=[],
                        help="Boundary/high-risk benign JSONL files for FPR")
    parser.add_argument("--output", type=str,
                        default="./results/utility_grid.json")
    parser.add_argument("--methods", nargs="+",
                        default=["guardlens", "surface_risk", "grad_x_input", "attention", "random"])
    parser.add_argument("--k-values", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--lambda-values", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--batch-size", type=int, default=4)
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
    model = model.to(device).eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = GuardLensCollator(tokenizer, config)

    # Load test data for DD computation
    test_records = load_jsonl(args.test_path)
    print(f"Test records: {len(test_records)}")

    dataset = GuardLensDataset(test_records, config)
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=collator, num_workers=4,
    )

    # Load boundary data for FPR computation
    boundary_records = []
    for path in args.boundary_files:
        if os.path.exists(path):
            recs = load_jsonl(path)
            boundary_records.extend(recs)
            print(f"Boundary: {len(recs)} from {os.path.basename(path)}")

    # Also use test set benign records with high-risk vocabulary
    test_benign_hr = [
        r for r in test_records
        if r.get("label") == 0
        and r.get("family") in ("false_lead_benign", "hard_benign",
                                 "research_technical", "topic_matched_safe")
    ]
    if test_benign_hr:
        print(f"Test high-risk benign: {len(test_benign_hr)}")
        boundary_records.extend(test_benign_hr)

    print(f"Total boundary records for FPR: {len(boundary_records)}")

    # Compute FPR on boundary records (model-level, not per-method)
    if boundary_records:
        boundary_fpr = run_boundary_fpr(
            model, boundary_records, config, tokenizer, device, threshold,
        )
        print(f"Boundary FPR: {boundary_fpr:.4f}")
    else:
        boundary_fpr = 0.0
        print("WARNING: No boundary records. FPR=0 (Utility = DD).")

    # Compute DD at each k for each method
    print(f"\nRunning causal evaluation for DD...")
    causal_results = run_causal_evaluation(
        model, test_loader, device,
        methods=args.methods,
        top_k_fractions=args.k_values,
        tokenizer=tokenizer,
    )

    # For surface_risk, compute its own FPR
    # Surface risk FPR is how often SR score >= threshold on boundary records
    from guardlens.evaluation.eval_surface_risk_fpr import (
        compute_conversation_surface_risk,
    )
    sr_boundary_fpr = 0.0
    if boundary_records:
        sr_fps = sum(
            1 for r in boundary_records
            if compute_conversation_surface_risk(r) >= 0.3
        )
        sr_boundary_fpr = sr_fps / len(boundary_records)
        print(f"Surface risk boundary FPR (threshold=0.3): {sr_boundary_fpr:.4f}")

    # Build utility grid
    print(f"\n{'='*80}")
    print(f"  ATTRIBUTION UTILITY GRID")
    print(f"  Utility(k, lambda) = DD@k - lambda * boundary_FPR")
    print(f"{'='*80}")

    grid = {}
    for k_frac in args.k_values:
        k_str = f"{int(k_frac * 100)}%"
        grid[k_str] = {}

        for lam in args.lambda_values:
            lam_str = f"λ={lam}"
            grid[k_str][lam_str] = {}

            for method in args.methods:
                method_data = causal_results.get(method, {})
                dd = method_data.get("deviation_drops", {}).get(k_str, 0)

                # Method-specific FPR
                if method == "surface_risk":
                    fpr = sr_boundary_fpr
                elif method in ("attention", "random"):
                    fpr = 0.0  # These methods don't produce classification FPR
                else:
                    fpr = boundary_fpr

                util = compute_utility(dd, fpr, lam)
                grid[k_str][lam_str][method] = {
                    "utility": util,
                    "dd": dd,
                    "fpr": fpr,
                }

    # Print grid
    for k_str in grid:
        print(f"\n  k = {k_str}:")
        print(f"  {'lambda':<10}", end="")
        for method in args.methods:
            print(f" {method:>14}", end="")
        print(f" {'winner':>14}")
        print(f"  {'-'*10}", end="")
        for _ in args.methods:
            print(f" {'-'*14}", end="")
        print(f" {'-'*14}")

        for lam in args.lambda_values:
            lam_str = f"λ={lam}"
            print(f"  {lam:<10}", end="")
            best_method = ""
            best_util = -999
            for method in args.methods:
                u = grid[k_str][lam_str][method]["utility"]
                print(f" {u:>14.3f}", end="")
                if u > best_util:
                    best_util = u
                    best_method = method
            print(f" {best_method:>14}")

    # Find crossover lambdas
    print(f"\n  CROSSOVER ANALYSIS:")
    print(f"  Lambda value where surface_risk overtakes guardlens:")
    for k_str in grid:
        gl_dd = causal_results.get("guardlens", {}).get("deviation_drops", {}).get(k_str, 0)
        sr_dd = causal_results.get("surface_risk", {}).get("deviation_drops", {}).get(k_str, 0)
        gl_fpr = boundary_fpr
        sr_fpr = sr_boundary_fpr

        # Utility_GL > Utility_SR when:
        # GL_DD - lam * GL_FPR > SR_DD - lam * SR_FPR
        # lam * (SR_FPR - GL_FPR) > SR_DD - GL_DD
        fpr_diff = sr_fpr - gl_fpr
        dd_diff = sr_dd - gl_dd

        if fpr_diff > 0:
            crossover = dd_diff / fpr_diff
            print(f"    k={k_str}: lambda < {crossover:.2f} → SR wins; "
                  f"lambda > {crossover:.2f} → GL wins")
        elif dd_diff > 0:
            print(f"    k={k_str}: SR wins at all lambda (higher DD, same or lower FPR)")
        else:
            print(f"    k={k_str}: GL wins at all lambda")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "grid": grid,
        "boundary_fpr": {
            "guardlens": boundary_fpr,
            "surface_risk": sr_boundary_fpr,
        },
        "methods": args.methods,
        "k_values": [f"{int(k*100)}%" for k in args.k_values],
        "lambda_values": args.lambda_values,
        "crossover_analysis": {},
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
