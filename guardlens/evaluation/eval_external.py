"""
guardlens/evaluation/eval_external.py

Self-eval vs external-eval comparison for causal attribution metrics.

The circularity concern:
  GuardLens produces attribution scores *and* is used to measure whether
  removing those tokens changes the classification. The model is grading
  its own homework. Deviation Drop and Flip Rate could be inflated simply
  because the model is internally consistent, not because the attribution
  is causally correct.

Fix:
  Use ConversationDeBERTa (flat DeBERTa, no attribution head, trained
  independently) as the external evaluator. GuardLens still *produces*
  the attribution scores. But the *evaluation* -- does removing these
  tokens change the unsafe classification? -- is answered by a separate
  model that has never seen GuardLens's attention patterns.

Paper table this produces:
  ┌─────────────────────┬──────────┬───────────┬─────────────┬──────────────┐
  │ Evaluator           │DD@15%    │Flip@15%   │Necessity@15%│Sufficiency@15│
  ├─────────────────────┼──────────┼───────────┼─────────────┼──────────────┤
  │ GuardLens (self)    │ 0.872    │ 0.868     │ 0.882       │ 0.787        │
  │ DeBERTa-flat (ext.) │   ?      │   ?       │   ?         │   ?          │
  │ TurnLevel (ext.)    │   ?      │   ?       │   ?         │   ?          │
  └─────────────────────┴──────────┴───────────┴─────────────┴──────────────┘

Expected: external eval drops modestly (~5-15pp). A large drop (>20pp)
indicates the attribution is exploiting model-specific features rather
than semantic causality.

Usage:
    python -m guardlens.evaluation.eval_external \\
        --data ~/work/results/dataset_gen/semantic_multiturn_v10_augmented.jsonl \\
        --attribution-checkpoint ~/work/results/dataset_gen/checkpoints/guardlens/best.pt \\
        --external-checkpoints \\
            ~/work/results/dataset_gen/checkpoints/conversation_deberta/best.pt \\
            ~/work/results/dataset_gen/checkpoints/turn_level/best.pt \\
        --output ~/work/results/dataset_gen/results/external_eval.json \\
        --top-k 0.05 0.10 0.15 0.20 \\
        --device cuda
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add project root to path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.data.splits import pair_aware_split
from guardlens.models import MODEL_REGISTRY


# -------------------------------------------------------
# External evaluator wrapper
#
# ConversationDeBERTa and TurnLevelClassifier have different
# forward signatures from GuardLens. This wrapper normalises
# them so the causal metric functions can call any evaluator
# identically.
# -------------------------------------------------------

class ExternalEvaluatorWrapper:
    """
    Wraps any baseline model so it can be called as an evaluator
    by the causal metric functions.

    The wrapper adapts the batch format. ConversationDeBERTa expects
    flattened [B, L] input; TurnLevelClassifier expects [B, T, S].
    GuardLens expects [B, T, S] with cross-turn tensors.

    The key invariant: when an attribution_mask is provided (shape
    [B, T, S], values 0/1), the wrapper zeros out the corresponding
    token embeddings before the backbone sees them. This is implemented
    by replacing masked token positions with the [PAD] token id rather
    than at the embedding level, since we don't have access to the
    external model's embedding layer to inject an attribution_mask.

    Why token replacement instead of embedding zeroing:
      Embedding zeroing requires access to the forward internals of the
      external model. Token replacement (masked tokens -> [PAD]) is a
      coarser approximation but is model-agnostic and avoids modifying
      the external model's architecture. The approximation is valid for
      measuring whether the classifier's *decision* changes -- the
      question is just whether removing those tokens flips the label.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        tokenizer,
        config: GuardLensConfig,
        device: torch.device,
    ):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.model.eval()

    @torch.no_grad()
    def get_prob(
        self,
        batch: Dict[str, torch.Tensor],
        attribution_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns P(adversarial) for each item in the batch.

        If attribution_mask is [B, T, S], token positions where mask=0
        are replaced with [PAD] before the forward pass.
        """
        input_ids = batch["input_ids"].to(self.device)          # [B, T, S]
        attention_mask = batch["attention_mask"].to(self.device) # [B, T, S]
        turn_mask = batch.get("turn_mask", torch.ones(
            input_ids.shape[:2], device=self.device
        )).to(self.device)
        role_ids = batch.get("role_ids", torch.zeros_like(
            batch["input_ids"]
        )).to(self.device)

        # Apply attribution mask: zero-out masked tokens by replacing with PAD
        if attribution_mask is not None:
            attr_mask = attribution_mask.to(self.device)  # [B, T, S]
            pad_ids = torch.full_like(input_ids, self.pad_token_id)
            input_ids = torch.where(attr_mask.bool(), input_ids, pad_ids)
            attention_mask = attention_mask * attr_mask

        if self.model_name == "conversation_deberta":
            # Flatten [B, T, S] -> [B, T*S] and trim to max non-pad length
            B, T, S = input_ids.shape
            flat_ids = input_ids.reshape(B, T * S)
            flat_mask = attention_mask.reshape(B, T * S)
            max_len = flat_mask.sum(dim=1).max().item()
            max_len = max(1, int(max_len))
            flat_ids = flat_ids[:, :max_len]
            flat_mask = flat_mask[:, :max_len]
            out = self.model(input_ids=flat_ids, attention_mask=flat_mask)
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                turn_mask=turn_mask,
                role_ids=role_ids,
            )

        return torch.sigmoid(out["cls_logits"])


# -------------------------------------------------------
# Causal metrics operating on ExternalEvaluatorWrapper
# -------------------------------------------------------

def _build_topk_mask(
    scores: torch.Tensor,
    valid: torch.Tensor,
    k_frac: float,
) -> torch.Tensor:
    """1=keep, 0=remove. Removes top-k% scored tokens."""
    flat = scores[valid.bool()]
    n = flat.numel()
    if n == 0:
        return torch.ones_like(scores)
    k = max(1, int(n * k_frac))
    threshold = flat.topk(k).values[-1]
    return (scores < threshold).float()


def _build_sufficiency_mask(
    scores: torch.Tensor,
    valid: torch.Tensor,
    k_frac: float,
    context_window: int = 2,
) -> torch.Tensor:
    """1=keep, 0=remove. Keeps ONLY top-k% tokens plus context window."""
    flat = scores[valid.bool()]
    n = flat.numel()
    if n == 0:
        return torch.zeros_like(scores)
    k = max(1, int(n * k_frac))
    threshold = flat.topk(k).values[-1]
    core = (scores >= threshold).float()
    if context_window > 0 and core.dim() >= 2:
        expanded = core.clone()
        T, S = core.shape
        for t in range(T):
            for s in range(S):
                if core[t, s] == 1:
                    lo = max(0, s - context_window)
                    hi = min(S, s + context_window + 1)
                    expanded[t, lo:hi] = 1.0
        return expanded * valid.float()
    return core * valid.float()


def _single_batch(batch: Dict, idx: int) -> Dict:
    return {k: v[idx:idx+1] if isinstance(v, torch.Tensor) else
            ([v[idx]] if isinstance(v, list) else v)
            for k, v in batch.items()}


def compute_deviation_drop(
    attr_scores: torch.Tensor,
    batch: Dict,
    evaluator: ExternalEvaluatorWrapper,
    k_frac: float,
) -> Tuple[List[float], int]:
    """
    Deviation Drop: mean P(adv) reduction when top-k tokens removed.
    Returns (list of per-sample drops, n_tested).
    """
    labels = batch["labels"]
    adv_idx = (labels == 1).nonzero(as_tuple=True)[0]
    if len(adv_idx) == 0:
        return [], 0

    drops = []
    for i in adv_idx:
        i = i.item()
        single = _single_batch(batch, i)
        orig_prob = evaluator.get_prob(single)[0].item()
        if orig_prob < 0.5:
            continue  # External model doesn't classify as adversarial; skip

        valid = (
            batch["attention_mask"][i] *
            batch.get("turn_mask", torch.ones(
                batch["attention_mask"].shape[:2]
            ))[i].unsqueeze(-1)
        )
        topk_mask = _build_topk_mask(attr_scores[i], valid, k_frac)
        cf_prob = evaluator.get_prob(single, attribution_mask=topk_mask.unsqueeze(0))[0].item()
        drops.append(max(0.0, orig_prob - cf_prob))

    return drops, len(drops)


def compute_flip_rate(
    attr_scores: torch.Tensor,
    batch: Dict,
    evaluator: ExternalEvaluatorWrapper,
    k_frac: float,
) -> Tuple[int, int]:
    """Flip Rate: fraction of adversarial samples where removal flips prediction."""
    labels = batch["labels"]
    adv_idx = (labels == 1).nonzero(as_tuple=True)[0]
    if len(adv_idx) == 0:
        return 0, 0

    flips = 0
    tested = 0
    for i in adv_idx:
        i = i.item()
        single = _single_batch(batch, i)
        orig_prob = evaluator.get_prob(single)[0].item()
        if orig_prob < 0.5:
            continue

        valid = (
            batch["attention_mask"][i] *
            batch.get("turn_mask", torch.ones(
                batch["attention_mask"].shape[:2]
            ))[i].unsqueeze(-1)
        )
        topk_mask = _build_topk_mask(attr_scores[i], valid, k_frac)
        cf_prob = evaluator.get_prob(single, attribution_mask=topk_mask.unsqueeze(0))[0].item()
        if cf_prob < 0.5:
            flips += 1
        tested += 1

    return flips, tested


def compute_necessity(
    attr_scores: torch.Tensor,
    batch: Dict,
    evaluator: ExternalEvaluatorWrapper,
    k_frac: float,
) -> Tuple[int, int]:
    """Necessity: removing attributed tokens eliminates adversarial classification."""
    return compute_flip_rate(attr_scores, batch, evaluator, k_frac)


def compute_sufficiency(
    attr_scores: torch.Tensor,
    batch: Dict,
    evaluator: ExternalEvaluatorWrapper,
    k_frac: float,
) -> Tuple[int, int]:
    """Sufficiency: attributed tokens alone preserve adversarial classification."""
    labels = batch["labels"]
    adv_idx = (labels == 1).nonzero(as_tuple=True)[0]
    if len(adv_idx) == 0:
        return 0, 0

    sufficient = 0
    tested = 0
    for i in adv_idx:
        i = i.item()
        single = _single_batch(batch, i)
        orig_prob = evaluator.get_prob(single)[0].item()
        if orig_prob < 0.5:
            continue

        valid = (
            batch["attention_mask"][i] *
            batch.get("turn_mask", torch.ones(
                batch["attention_mask"].shape[:2]
            ))[i].unsqueeze(-1)
        )
        suf_mask = _build_sufficiency_mask(attr_scores[i], valid, k_frac)
        cf_prob = evaluator.get_prob(single, attribution_mask=suf_mask.unsqueeze(0))[0].item()
        if cf_prob >= 0.5:
            sufficient += 1
        tested += 1

    return sufficient, tested


# -------------------------------------------------------
# Attribution score extraction (from GuardLens checkpoint)
# -------------------------------------------------------

@torch.no_grad()
def extract_attribution_scores(
    guardlens_model: torch.nn.Module,
    batch: Dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Run GuardLens forward pass and return attr_probs [B, T, S].
    GuardLens is used ONLY for attribution; the evaluator is external.
    """
    guardlens_model.eval()
    out = guardlens_model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        turn_mask=batch["turn_mask"].to(device),
        role_ids=batch["role_ids"].to(device),
        compute_attribution=True,
    )
    # attr_probs: [B, T, S], already sigmoided
    return out["attr_probs"].cpu()


# -------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------

def run_external_evaluation(
    guardlens_model: torch.nn.Module,
    external_evaluators: Dict[str, ExternalEvaluatorWrapper],
    test_loader: DataLoader,
    device: torch.device,
    top_k_fractions: List[float],
) -> Dict:
    """
    For each batch:
      1. Extract GuardLens attribution scores (attr_probs)
      2. For each external evaluator, compute causal metrics
         using those attribution scores

    This tests: are GuardLens's attributed tokens actually causal
    according to an independently trained model?
    """
    results = {
        name: {
            "deviation_drops": {f"{int(k*100)}%": [] for k in top_k_fractions},
            "flip_counts": {f"{int(k*100)}%": [0, 0] for k in top_k_fractions},
            "necessity_counts": {f"{int(k*100)}%": [0, 0] for k in top_k_fractions},
            "sufficiency_counts": {f"{int(k*100)}%": [0, 0] for k in top_k_fractions},
            "n_adversarial_tested": 0,
        }
        for name in external_evaluators
    }

    n_batches = len(test_loader)
    for batch_idx, batch in enumerate(test_loader):
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{n_batches}...")

        # Step 1: GuardLens attribution
        attr_scores = extract_attribution_scores(guardlens_model, batch, device)
        # attr_scores: [B, T, S]

        # Step 2: Evaluate with each external evaluator
        for name, evaluator in external_evaluators.items():
            for k_frac in top_k_fractions:
                k_str = f"{int(k_frac*100)}%"

                drops, n_tested = compute_deviation_drop(
                    attr_scores, batch, evaluator, k_frac
                )
                results[name]["deviation_drops"][k_str].extend(drops)

                flips, f_tested = compute_flip_rate(
                    attr_scores, batch, evaluator, k_frac
                )
                results[name]["flip_counts"][k_str][0] += flips
                results[name]["flip_counts"][k_str][1] += f_tested

                nec, n_tested2 = compute_necessity(
                    attr_scores, batch, evaluator, k_frac
                )
                results[name]["necessity_counts"][k_str][0] += nec
                results[name]["necessity_counts"][k_str][1] += n_tested2

                suf, s_tested = compute_sufficiency(
                    attr_scores, batch, evaluator, k_frac
                )
                results[name]["sufficiency_counts"][k_str][0] += suf
                results[name]["sufficiency_counts"][k_str][1] += s_tested

    # Aggregate
    final = {}
    for name, data in results.items():
        final[name] = {}
        for k_frac in top_k_fractions:
            k_str = f"{int(k_frac*100)}%"
            drops = data["deviation_drops"][k_str]
            flips, f_n = data["flip_counts"][k_str]
            nec, nec_n = data["necessity_counts"][k_str]
            suf, suf_n = data["sufficiency_counts"][k_str]
            final[name][k_str] = {
                "deviation_drop": sum(drops) / max(1, len(drops)),
                "flip_rate": flips / max(1, f_n),
                "necessity": nec / max(1, nec_n),
                "sufficiency": suf / max(1, suf_n),
                "n_adversarial": f_n,
            }

    return final


def print_comparison_table(
    self_eval_results: Dict,
    external_results: Dict,
    top_k_fractions: List[float],
):
    """Print self-eval vs external-eval side by side."""
    print("\n" + "=" * 80)
    print("  Self-Eval vs External-Eval Comparison")
    print("  Attribution: GuardLens | Evaluator: varies")
    print("=" * 80)

    k_str = f"{int(top_k_fractions[2]*100)}%"  # default: 15%

    print(f"\n  Metrics at top-{k_str} token removal:")
    print(f"  {'Evaluator':<30} {'DevDrop':>10} {'Flip':>10} {'Nec':>10} {'Suf':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Self eval
    se = self_eval_results.get(k_str, {})
    print(f"  {'GuardLens (self)':<30} "
          f"{se.get('deviation_drop', 0):>10.3f} "
          f"{se.get('flip_rate', 0):>10.3f} "
          f"{se.get('necessity', 0):>10.3f} "
          f"{se.get('sufficiency', 0):>10.3f}")

    # External evaluators
    for name, data in external_results.items():
        row = data.get(k_str, {})
        print(f"  {name:<30} "
              f"{row.get('deviation_drop', 0):>10.3f} "
              f"{row.get('flip_rate', 0):>10.3f} "
              f"{row.get('necessity', 0):>10.3f} "
              f"{row.get('sufficiency', 0):>10.3f}")

    # Full breakdown by k
    print(f"\n  Full breakdown by k (DevDrop | FlipRate):")
    k_strs = [f"{int(k*100)}%" for k in top_k_fractions]

    print(f"\n  GuardLens (self-eval):")
    for ks in k_strs:
        se = self_eval_results.get(ks, {})
        print(f"    {ks}: DD={se.get('deviation_drop', 0):.3f}  "
              f"Flip={se.get('flip_rate', 0):.3f}  "
              f"Nec={se.get('necessity', 0):.3f}  "
              f"Suf={se.get('sufficiency', 0):.3f}")

    for name, data in external_results.items():
        print(f"\n  {name} (external eval):")
        for ks in k_strs:
            row = data.get(ks, {})
            print(f"    {ks}: DD={row.get('deviation_drop', 0):.3f}  "
                  f"Flip={row.get('flip_rate', 0):.3f}  "
                  f"Nec={row.get('necessity', 0):.3f}  "
                  f"Suf={row.get('sufficiency', 0):.3f}")

    # Gap analysis
    print(f"\n  Gap analysis (self minus external at {k_str}):")
    se_15 = self_eval_results.get(k_str, {})
    for name, data in external_results.items():
        ext_15 = data.get(k_str, {})
        for metric in ["deviation_drop", "flip_rate", "necessity", "sufficiency"]:
            gap = se_15.get(metric, 0) - ext_15.get(metric, 0)
            flag = " ⚠ large gap" if abs(gap) > 0.20 else ""
            print(f"    {name} | {metric}: {gap:+.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Self-eval vs external-eval")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--attribution-checkpoint", type=str, required=True,
                        help="GuardLens checkpoint for attribution scoring")
    parser.add_argument("--external-checkpoints", nargs="+", required=True,
                        help="External evaluator checkpoints (conversation_deberta, turn_level)")
    parser.add_argument("--output", type=str,
                        default="./results/external_eval.json")
    parser.add_argument("--top-k", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--subset", type=str, default=None,
                        choices=["implicit", "explicit", "clean_holdout", None],
                        help="Evaluate on a subset of the test set")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GuardLens attribution model
    print(f"\nLoading GuardLens attribution model from {args.attribution_checkpoint}...")
    gl_ckpt = torch.load(args.attribution_checkpoint, weights_only=False, map_location=device)
    gl_config = gl_ckpt["config"]
    gl_model_name = gl_ckpt.get("model_name", "guardlens")
    gl_model_cls = MODEL_REGISTRY.get(gl_model_name, MODEL_REGISTRY["guardlens"])
    gl_model = gl_model_cls(gl_config)
    gl_model.setup_backbone()
    gl_model.load_state_dict(gl_ckpt["model_state_dict"])
    gl_model = gl_model.to(device)
    gl_model.eval()
    print(f"  GuardLens loaded (epoch {gl_ckpt.get('epoch','?')}, "
          f"phase {gl_ckpt.get('phase','?')})")

    # Tokenizer (shared -- same backbone for all models)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(gl_config.backbone_name)
    collator = GuardLensCollator(tokenizer, gl_config)

    # Load data
    print(f"\nLoading data from {args.data}...")
    records = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"  {len(records)} total records")

    _, _, test_idx = pair_aware_split(records, seed=gl_config.seed)
    test_records = [records[i] for i in test_idx]

    # Optional subset filtering
    if args.subset == "implicit":
        test_idx_filtered = [
            i for i, r in enumerate(test_records)
            if any(t.get("implicit_trigger") for t in r.get("turns", []))
        ]
        print(f"  Implicit trigger subset: {len(test_idx_filtered)} samples")
    elif args.subset == "explicit":
        test_idx_filtered = [
            i for i, r in enumerate(test_records)
            if r.get("label") == 1 and
            not any(t.get("implicit_trigger") for t in r.get("turns", []))
        ]
        print(f"  Explicit trigger subset: {len(test_idx_filtered)} samples")
    elif args.subset == "clean_holdout":
        test_idx_filtered = [
            i for i, r in enumerate(test_records)
            if r.get("metadata", {}).get("clean_holdout")
        ]
        print(f"  Clean holdout subset: {len(test_idx_filtered)} samples")
    else:
        test_idx_filtered = list(range(len(test_records)))
        print(f"  Using full test set: {len(test_idx_filtered)} samples")

    dataset = GuardLensDataset(records, gl_config)
    # Map filtered local indices back to global record indices
    global_test_idx = [test_idx[i] for i in test_idx_filtered]

    test_loader = DataLoader(
        Subset(dataset, global_test_idx),
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"  Test loader: {len(test_loader)} batches")

    # Load external evaluators
    print(f"\nLoading {len(args.external_checkpoints)} external evaluator(s)...")
    external_evaluators = {}
    for ckpt_path in args.external_checkpoints:
        print(f"  {ckpt_path}...")
        ext_ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        ext_config = ext_ckpt["config"]
        ext_model_name = ext_ckpt.get("model_name", "conversation_deberta")
        ext_model_cls = MODEL_REGISTRY.get(ext_model_name, MODEL_REGISTRY["conversation_deberta"])
        ext_model = ext_model_cls(ext_config)
        ext_model.setup_backbone()
        ext_model.load_state_dict(ext_ckpt["model_state_dict"])
        ext_model = ext_model.to(device)
        ext_model.eval()

        evaluator_name = f"{ext_model_name} (ext.)"
        external_evaluators[evaluator_name] = ExternalEvaluatorWrapper(
            model=ext_model,
            model_name=ext_model_name,
            tokenizer=tokenizer,
            config=ext_config,
            device=device,
        )
        print(f"    Loaded {ext_model_name} (epoch {ext_ckpt.get('epoch','?')})")

    # Also run self-eval for comparison (GuardLens evaluates GuardLens)
    print("\nRunning self-eval (GuardLens -> GuardLens)...")
    self_evaluator = ExternalEvaluatorWrapper(
        model=gl_model,
        model_name="guardlens",
        tokenizer=tokenizer,
        config=gl_config,
        device=device,
    )
    self_eval_results_raw = run_external_evaluation(
        gl_model, {"guardlens (self)": self_evaluator},
        test_loader, device, args.top_k
    )
    self_eval_results = self_eval_results_raw["guardlens (self)"]

    # External eval
    print("\nRunning external evaluation...")
    external_results = run_external_evaluation(
        gl_model, external_evaluators,
        test_loader, device, args.top_k
    )

    # Print table
    print_comparison_table(self_eval_results, external_results, args.top_k)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "self_eval": self_eval_results,
        "external_eval": external_results,
        "subset": args.subset or "full_test",
        "n_test_samples": len(global_test_idx),
        "top_k_fractions": args.top_k,
        "attribution_checkpoint": args.attribution_checkpoint,
        "external_checkpoints": args.external_checkpoints,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()