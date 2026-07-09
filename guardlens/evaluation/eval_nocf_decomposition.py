"""
guardlens/evaluation/eval_nocf_decomposition.py

Specificity-controlled ablation: decompose NoCF's DD@k advantage
into surface-risk-driven vs non-surface-risk components.

Reviewer aeLx's concern: NoCF outperforms the full model on DD@k.
Does this mean the CF loss hurts? Or does NoCF achieve higher DD
by over-attributing surface-risk tokens (less specific)?

Method:
  1. For each model (full GuardLens, NoCF):
     a. Get attribution scores on adversarial test conversations
     b. Get surface risk scores for the same tokens
     c. Select top-k% attributed tokens
     d. Partition those tokens into surface-risk-positive
        (surface_risk >= 0.3) and non-surface-risk
        (surface_risk < 0.3)
     e. Compute DD separately for each partition:
        - DD_sr:  zero only the SR-positive attributed tokens
        - DD_non: zero only the non-SR attributed tokens
  2. Compare:
     - If NoCF's advantage concentrates in DD_sr: the CF loss is
       teaching specificity, not hurting attribution
     - If NoCF wins on DD_non too: the CF loss may be genuinely
       harmful to attribution quality

Output table:
  | Model    | DD all | DD non-surface-risk | DD surface-risk | SR fraction |
  |----------|--------|---------------------|-----------------|-------------|
  | GuardLens| ...    | ...                 | ...             | ...         |
  | NoCF     | ...    | ...                 | ...             | ...         |

Usage:
    python -m guardlens.evaluation.eval_nocf_decomposition \\
        --test-path splits/test.jsonl \\
        --gl-checkpoint checkpoints/guardlens/best.pt \\
        --nocf-checkpoint checkpoints/guardlens_no_cf/best.pt \\
        --output results/nocf_decomposition.json \\
        --device cuda
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

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
    guardlens_attribution,
    surface_risk_attribution,
    _single_batch,
    _get_prob,
)


def _build_partitioned_masks(
    attr_scores: torch.Tensor,
    sr_scores: torch.Tensor,
    valid: torch.Tensor,
    k_frac: float,
    sr_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Build three masks from top-k attributed tokens, partitioned
    by surface risk.

    Returns:
        mask_all: standard top-k mask (1=keep, 0=remove). Removes all top-k.
        mask_sr_only: removes only top-k tokens that are SR-positive
        mask_non_sr_only: removes only top-k tokens that are SR-negative
        stats: partition statistics
    """
    valid_bool = valid.bool()
    flat_attr = attr_scores[valid_bool].flatten()
    n_tokens = flat_attr.numel()
    if n_tokens == 0:
        ones = torch.ones_like(attr_scores)
        return ones, ones, ones, {"n_topk": 0, "n_sr": 0, "n_non_sr": 0}

    k = max(1, int(n_tokens * k_frac))

    # Use topk indices to guarantee exactly k tokens are selected,
    # avoiding over-selection when multiple tokens tie at the threshold.
    topk_idx = torch.topk(flat_attr, k).indices

    flat_topk = torch.zeros(n_tokens, dtype=torch.bool)
    flat_topk[topk_idx] = True

    # Map back to full [T, S] shape
    topk_mask = torch.zeros_like(valid_bool)
    topk_mask[valid_bool] = flat_topk

    # SR partition within top-k (also gated by valid)
    sr_positive = (sr_scores >= sr_threshold) & valid_bool

    topk_and_sr = topk_mask & sr_positive
    topk_and_non_sr = topk_mask & ~sr_positive

    # Masks: 1=keep, 0=remove
    mask_all = torch.ones_like(attr_scores)
    mask_sr_only = torch.ones_like(attr_scores)
    mask_non_sr_only = torch.ones_like(attr_scores)

    mask_all[topk_mask] = 0.0
    mask_sr_only[topk_and_sr] = 0.0
    mask_non_sr_only[topk_and_non_sr] = 0.0

    stats = {
        "n_topk": topk_mask.sum().item(),
        "n_sr": topk_and_sr.sum().item(),
        "n_non_sr": topk_and_non_sr.sum().item(),
        "sr_fraction": topk_and_sr.sum().item() / max(1, topk_mask.sum().item()),
    }

    return mask_all, mask_sr_only, mask_non_sr_only, stats


def decomposed_deviation_drop(
    model,
    batch: Dict,
    attr_scores: torch.Tensor,
    sr_scores: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.15,
    sr_threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Compute DD separately for:
      - All top-k tokens (standard DD)
      - Surface-risk-positive top-k tokens only
      - Non-surface-risk top-k tokens only
    """
    labels = batch["labels"]
    adv_idx = (labels == 1).nonzero(as_tuple=True)[0]
    if len(adv_idx) == 0:
        return {"dd_all": 0, "dd_sr": 0, "dd_non_sr": 0,
                "n_tested": 0, "sr_fraction": 0}

    drops_all = []
    drops_sr = []
    drops_non_sr = []
    sr_fracs = []

    for i in adv_idx:
        valid_i = (batch["attention_mask"][i] * batch["turn_mask"][i].unsqueeze(-1))
        orig_prob = _get_prob(model, _single_batch(batch, i), device)[0].item()
        if orig_prob < 0.5:
            continue

        mask_all, mask_sr, mask_non_sr, stats = _build_partitioned_masks(
            attr_scores[i], sr_scores[i], valid_i, k_frac, sr_threshold,
        )

        # DD on all top-k
        p_all = _get_prob(
            model, _single_batch(batch, i), device,
            attribution_mask=mask_all.unsqueeze(0),
        )[0].item()
        drops_all.append(orig_prob - p_all)

        # DD on SR-positive top-k only
        if stats["n_sr"] > 0:
            p_sr = _get_prob(
                model, _single_batch(batch, i), device,
                attribution_mask=mask_sr.unsqueeze(0),
            )[0].item()
            drops_sr.append(orig_prob - p_sr)
        else:
            drops_sr.append(0.0)

        # DD on SR-negative top-k only
        if stats["n_non_sr"] > 0:
            p_non = _get_prob(
                model, _single_batch(batch, i), device,
                attribution_mask=mask_non_sr.unsqueeze(0),
            )[0].item()
            drops_non_sr.append(orig_prob - p_non)
        else:
            drops_non_sr.append(0.0)

        sr_fracs.append(stats["sr_fraction"])

    if not drops_all:
        return {"dd_all": 0, "dd_sr": 0, "dd_non_sr": 0,
                "n_tested": 0, "sr_fraction": 0}

    return {
        "dd_all": float(np.mean(drops_all)),
        "dd_sr": float(np.mean(drops_sr)),
        "dd_non_sr": float(np.mean(drops_non_sr)),
        "n_tested": len(drops_all),
        "sr_fraction": float(np.mean(sr_fracs)),
        "per_record_dd_all": [float(d) for d in drops_all],
        "per_record_dd_sr": [float(d) for d in drops_sr],
        "per_record_dd_non_sr": [float(d) for d in drops_non_sr],
    }


def run_decomposition(
    model,
    loader: DataLoader,
    device: torch.device,
    k_fracs: List[float],
    tokenizer,
    model_label: str,
) -> Dict:
    """Run decomposed DD for a single model across all k fractions."""
    model.eval()
    results = {}

    for k_frac in k_fracs:
        k_str = f"{int(k_frac * 100)}%"
        all_dd = []

        for batch in loader:
            # GuardLens attribution scores
            attr_scores = guardlens_attribution(model, batch, device)
            # Surface risk scores (token-level)
            sr_scores = surface_risk_attribution(
                model, batch, device, tokenizer=tokenizer,
            )

            dd = decomposed_deviation_drop(
                model, batch, attr_scores, sr_scores, device, k_frac,
            )
            if dd["n_tested"] > 0:
                all_dd.append(dd)

        if all_dd:
            # Aggregate per-record values across batches
            all_records_all = []
            all_records_sr = []
            all_records_non = []
            all_sr_fracs = []
            for d in all_dd:
                all_records_all.extend(d["per_record_dd_all"])
                all_records_sr.extend(d["per_record_dd_sr"])
                all_records_non.extend(d["per_record_dd_non_sr"])
                all_sr_fracs.append(d["sr_fraction"])

            results[k_str] = {
                "dd_all": float(np.mean(all_records_all)),
                "dd_sr": float(np.mean(all_records_sr)),
                "dd_non_sr": float(np.mean(all_records_non)),
                "sr_fraction": float(np.mean(all_sr_fracs)),
                "n_tested": len(all_records_all),
            }
        else:
            results[k_str] = {
                "dd_all": 0, "dd_sr": 0, "dd_non_sr": 0,
                "sr_fraction": 0, "n_tested": 0,
            }

        print(f"  {model_label} @ {k_str}: "
              f"DD_all={results[k_str]['dd_all']:.4f}  "
              f"DD_nonSR={results[k_str]['dd_non_sr']:.4f}  "
              f"DD_SR={results[k_str]['dd_sr']:.4f}  "
              f"SR_frac={results[k_str]['sr_fraction']:.3f}  "
              f"n={results[k_str]['n_tested']}")

    return results


def load_model(ckpt_path, device):
    """Load a model from checkpoint."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")
    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config, model_name


def main():
    parser = argparse.ArgumentParser(
        description="NoCF specificity decomposition",
    )
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--gl-checkpoint", type=str, required=True,
                        help="Full GuardLens checkpoint")
    parser.add_argument("--nocf-checkpoint", type=str, required=True,
                        help="GuardLens NoCF checkpoint")
    parser.add_argument("--output", type=str,
                        default="./results/nocf_decomposition.json")
    parser.add_argument("--top-k", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("\nLoading full GuardLens...")
    gl_model, gl_config, gl_name = load_model(args.gl_checkpoint, device)
    print(f"  Loaded: {gl_name}")

    print("Loading NoCF...")
    nocf_model, nocf_config, nocf_name = load_model(args.nocf_checkpoint, device)
    print(f"  Loaded: {nocf_name}")

    # Tokenizer and data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(gl_config.backbone_name)
    collator = GuardLensCollator(tokenizer, gl_config)

    from guardlens.evaluation.eval_utils import load_jsonl
    records = load_jsonl(args.test_path)
    print(f"Test records: {len(records)}")

    dataset = GuardLensDataset(records, gl_config)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=collator, num_workers=args.workers,
    )

    # Run decomposition for each model
    print(f"\n{'='*70}")
    print(f"  Full GuardLens Decomposition")
    print(f"{'='*70}")
    gl_results = run_decomposition(
        gl_model, loader, device, args.top_k, tokenizer, "GuardLens",
    )

    print(f"\n{'='*70}")
    print(f"  NoCF Decomposition")
    print(f"{'='*70}")
    nocf_results = run_decomposition(
        nocf_model, loader, device, args.top_k, tokenizer, "NoCF",
    )

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  SPECIFICITY DECOMPOSITION: GuardLens vs NoCF")
    print(f"{'='*80}")
    print(f"  {'k':<6} {'Model':<12} {'DD all':>8} {'DD non-SR':>10} "
          f"{'DD SR':>8} {'SR frac':>9} {'n':>5}")
    print(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*9} {'-'*5}")

    for k_str in [f"{int(k*100)}%" for k in args.top_k]:
        gl = gl_results.get(k_str, {})
        nc = nocf_results.get(k_str, {})

        print(f"  {k_str:<6} {'GuardLens':<12} "
              f"{gl.get('dd_all',0):>8.4f} {gl.get('dd_non_sr',0):>10.4f} "
              f"{gl.get('dd_sr',0):>8.4f} {gl.get('sr_fraction',0):>9.3f} "
              f"{gl.get('n_tested',0):>5}")
        print(f"  {'':6} {'NoCF':<12} "
              f"{nc.get('dd_all',0):>8.4f} {nc.get('dd_non_sr',0):>10.4f} "
              f"{nc.get('dd_sr',0):>8.4f} {nc.get('sr_fraction',0):>9.3f} "
              f"{nc.get('n_tested',0):>5}")

        # Delta row
        delta_all = nc.get("dd_all", 0) - gl.get("dd_all", 0)
        delta_non = nc.get("dd_non_sr", 0) - gl.get("dd_non_sr", 0)
        delta_sr = nc.get("dd_sr", 0) - gl.get("dd_sr", 0)
        delta_frac = nc.get("sr_fraction", 0) - gl.get("sr_fraction", 0)
        print(f"  {'':6} {'Delta':<12} "
              f"{delta_all:>+8.4f} {delta_non:>+10.4f} "
              f"{delta_sr:>+8.4f} {delta_frac:>+9.3f}")
        print()

    # Key diagnostic
    k_focus = "15%"
    gl_15 = gl_results.get(k_focus, {})
    nc_15 = nocf_results.get(k_focus, {})
    delta_non = nc_15.get("dd_non_sr", 0) - gl_15.get("dd_non_sr", 0)
    delta_sr = nc_15.get("dd_sr", 0) - gl_15.get("dd_sr", 0)

    print(f"  DIAGNOSTIC @ {k_focus}:")
    if delta_non <= 0 and delta_sr > 0:
        print(f"  CF LOSS VALIDATED: NoCF advantage concentrates in surface-risk tokens.")
        print(f"  NoCF DD non-SR delta: {delta_non:+.4f} (no advantage on non-surface-risk tokens)")
        print(f"  NoCF DD SR delta:     {delta_sr:+.4f} (over-attributes surface-risk tokens)")
        print(f"  NoCF SR fraction:     {nc_15.get('sr_fraction',0):.3f} "
              f"vs GuardLens {gl_15.get('sr_fraction',0):.3f}")
    elif delta_non > 0:
        print(f"  WARNING: NoCF also wins on non-surface-risk tokens ({delta_non:+.4f}).")
        print(f"  CF loss may be hurting attribution quality, not just specificity.")
        print(f"  Consider demoting CF from core contribution to ablation finding.")
    else:
        print(f"  MIXED: requires manual interpretation.")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "guardlens": gl_results,
        "nocf": nocf_results,
        "top_k_fractions": args.top_k,
        "sr_threshold": 0.3,
        "gl_checkpoint": args.gl_checkpoint,
        "nocf_checkpoint": args.nocf_checkpoint,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
