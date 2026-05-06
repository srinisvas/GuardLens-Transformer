import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.data.splits import pair_aware_split
from guardlens.models import MODEL_REGISTRY


class ParaphraseBackend:
    def paraphrase(self, text: str, context: str = "") -> str:
        raise NotImplementedError

    def paraphrase_batch(self, texts: List[str]) -> List[str]:
        return [self.paraphrase(t) for t in texts]


class OllamaParaphraser(ParaphraseBackend):
    def __init__(self, model: str = "qwen2.5:3b",
                 url: str = "http://localhost:11434/api/generate"):
        import requests
        self.model = model
        self.url = url
        self.requests = requests

    def _build_prompt(self, text: str) -> str:
        return (
            f"Rewrite the following message in different words while preserving "
            f"the exact intent and meaning. Keep the same length approximately. "
            f"Output ONLY the rewritten message, nothing else.\n\n"
            f"Original: {text}\n\n"
            f"Rewritten:"
        )

    def paraphrase(self, text: str, context: str = "") -> str:
        try:
            resp = self.requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": self._build_prompt(text),
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 200},
                },
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()["response"].strip()
            for prefix in ["Rewritten:", "Here is", "Here's", "Sure,", "Certainly,"]:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].strip()
            return result if len(result) >= 10 else text
        except Exception:
            return text


class VLLMParaphraser(ParaphraseBackend):
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        batch_size: int = 16,
    ):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key="token-none")
        except ImportError:
            raise ImportError("openai package required for vLLM backend: pip install openai")
        self.model = model
        self.batch_size = batch_size

    def _build_messages(self, text: str) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a paraphrasing assistant. When given a message, "
                    "rewrite it in different words while preserving the exact "
                    "intent and meaning. Output ONLY the rewritten message."
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite this: {text}",
            },
        ]

    def paraphrase(self, text: str, context: str = "") -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(text),
                temperature=0.7,
                max_tokens=300,
            )
            result = resp.choices[0].message.content.strip()
            return result if len(result) >= 10 else text
        except Exception as e:
            print(f"    vLLM paraphrase error: {e}")
            return text

    def paraphrase_batch(self, texts: List[str]) -> List[str]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            results.extend([self.paraphrase(t) for t in batch])
        return results


class HuggingFaceParaphraser(ParaphraseBackend):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"    Loading paraphrase model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.model_name = model_name
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def paraphrase(self, text: str, context: str = "") -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a paraphrasing assistant. Rewrite the user's "
                        "message in different words while preserving the exact "
                        "intent and meaning. Output ONLY the rewritten message, "
                        "nothing else, no preamble."
                    ),
                },
                {"role": "user", "content": f"Rewrite this: {text}"},
            ]

            templated = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(templated, "input_ids"):
                input_ids = templated.input_ids
            else:
                input_ids = templated
            input_ids = input_ids.to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = out[0][input_ids.shape[1]:]
            result = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            for prefix in ["Sure,", "Sure!", "Sure ", "Here is", "Here's",
                           "Certainly,", "Certainly!", "Of course,",
                           "Rewrite:", "Rewritten:", "Paraphrased:",
                           "The rewritten message:", "Rewritten message:"]:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].strip(":,.! ")
            if len(result) >= 2 and result[0] in '"\'' and result[-1] in '"\'':
                result = result[1:-1]

            return result if len(result) >= 10 else text
        except Exception as e:
            print(f"    HF paraphrase error: {type(e).__name__}: {e}", flush=True)
            return text

def build_paraphrase_backend(args) -> ParaphraseBackend:
    if args.paraphrase_model == "vllm":
        print(f"  Using vLLM backend (model={args.vllm_model}, url={args.vllm_url})")
        return VLLMParaphraser(model=args.vllm_model, base_url=args.vllm_url)
    elif args.paraphrase_model == "hf":
        print(f"  Using HuggingFace backend (model={args.vllm_model})")
        return HuggingFaceParaphraser(model_name=args.vllm_model, device=args.device)
    else:
        print(f"  Using Ollama backend (model={args.ollama_model})")
        return OllamaParaphraser(model=args.ollama_model)


def paraphrase_record(
    record: Dict,
    paraphraser: ParaphraseBackend,
) -> Dict:
    import copy
    paraphrased = copy.deepcopy(record)

    user_turns = [i for i, t in enumerate(paraphrased["turns"]) if t["role"] == "user"]

    for idx in user_turns:
        original_text = paraphrased["turns"][idx]["text"]
        paraphrased["turns"][idx]["text"] = paraphraser.paraphrase(original_text)
        paraphrased["turns"][idx]["paraphrased"] = True
        paraphrased["turns"][idx]["original_text"] = original_text

    paraphrased["metadata"] = dict(paraphrased.get("metadata", {}))
    paraphrased["metadata"]["paraphrased"] = True
    paraphrased["conversation_id"] = paraphrased["conversation_id"] + "_para"

    return paraphrased


@torch.no_grad()
def get_attribution_scores(
    model: torch.nn.Module,
    record: Dict,
    all_records: List[Dict],
    collator: GuardLensCollator,
    config: GuardLensConfig,
    device: torch.device,
) -> Optional[torch.Tensor]:
    temp_records = [record]
    temp_dataset = GuardLensDataset(temp_records, config)
    temp_loader = DataLoader(
        temp_dataset,
        batch_size=1,
        collate_fn=collator,
        num_workers=0,
        shuffle=False,
    )

    model.eval()
    for batch in temp_loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            turn_mask=batch["turn_mask"].to(device),
            role_ids=batch["role_ids"].to(device),
            compute_attribution=True,
        )
        if out["attr_probs"] is None:
            return None
        return out["attr_probs"][0].cpu()

    return None


def compute_robustness_metrics(
    orig_attr: torch.Tensor,
    para_attr: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    top_k: float = 0.15,
) -> Dict:
    T, S = orig_attr.shape

    if attention_mask is not None:
        mask = attention_mask.float()
        orig_turn = (orig_attr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        para_turn = (para_attr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        orig_turn = orig_attr.mean(dim=1)
        para_turn = para_attr.mean(dim=1)

    orig_turn_np = orig_turn.numpy()
    para_turn_np = para_turn.numpy()

    if len(orig_turn_np) < 2:
        turn_rho = 0.0
        turn_p = 1.0
    else:
        corr = spearmanr(orig_turn_np, para_turn_np)
        turn_rho = float(corr.statistic) if not np.isnan(corr.statistic) else 0.0
        turn_p = float(corr.pvalue)

    if attention_mask is not None:
        valid = attention_mask.bool().flatten()
        orig_tok = orig_attr.flatten()[valid].numpy()
        para_tok = para_attr.flatten()[valid].numpy()
    else:
        orig_tok = orig_attr.flatten().numpy()
        para_tok = para_attr.flatten().numpy()

    if len(orig_tok) < 2:
        tok_rho = 0.0
    else:
        corr_tok = spearmanr(orig_tok, para_tok)
        tok_rho = float(corr_tok.statistic) if not np.isnan(corr_tok.statistic) else 0.0

    n_total = len(orig_tok)
    k = max(1, int(n_total * top_k))
    if n_total >= k:
        orig_topk = set(np.argsort(orig_tok)[-k:].tolist())
        para_topk = set(np.argsort(para_tok)[-k:].tolist())
        topk_stability = len(orig_topk & para_topk) / max(1, k)
    else:
        topk_stability = 0.0

    orig_pivot = int(np.argmax(orig_turn_np))
    para_pivot = int(np.argmax(para_turn_np))
    pivot_stable = int(orig_pivot == para_pivot)

    return {
        "turn_rho": turn_rho,
        "turn_p": turn_p,
        "token_rho": tok_rho,
        "topk_stability": topk_stability,
        "pivot_stable": pivot_stable,
        "orig_pivot_turn": orig_pivot,
        "para_pivot_turn": para_pivot,
    }


def main():
    parser = argparse.ArgumentParser(description="Paraphrase robustness evaluation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="./results/paraphrase_eval.json")
    parser.add_argument("--n-samples", type=int, default=150,
                        help="Number of adversarial test samples to paraphrase")
    parser.add_argument("--top-k", type=float, default=0.15,
                        help="Top-k fraction for stability metric")
    parser.add_argument("--paraphrase-model", type=str, default="vllm",
                        choices=["vllm", "hf", "ollama"])
    parser.add_argument("--vllm-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:3b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=str, default=None,
                        choices=["implicit", "explicit", None],
                        help="Evaluate only on implicit or explicit trigger subset")
    parser.add_argument("--save-examples", type=int, default=5,
                        help="Number of examples to save with full attribution scores")
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt["config"]
    model_name = ckpt.get("model_name", "guardlens")
    model_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["guardlens"])
    model = model_cls(config)
    model.setup_backbone()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded {model_name}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    collator = GuardLensCollator(tokenizer, config)

    print(f"\nLoading data from {args.data}...")
    records = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    _, _, test_idx = pair_aware_split(records, seed=config.seed)
    test_records = [records[i] for i in test_idx]

    adv_records = [r for r in test_records if r["label"] == 1]

    if args.subset == "implicit":
        adv_records = [
            r for r in adv_records
            if any(t.get("implicit_trigger") for t in r.get("turns", []))
        ]
        print(f"  Implicit trigger subset: {len(adv_records)} adversarial samples")
    elif args.subset == "explicit":
        adv_records = [
            r for r in adv_records
            if not any(t.get("implicit_trigger") for t in r.get("turns", []))
        ]
        print(f"  Explicit trigger subset: {len(adv_records)} adversarial samples")
    else:
        print(f"  Full adversarial test set: {len(adv_records)} samples")

    sampled = rng.sample(adv_records, min(args.n_samples, len(adv_records)))
    print(f"  Sampled: {len(sampled)} records for paraphrase evaluation")

    print(f"\nInitialising paraphrase backend ({args.paraphrase_model})...")
    paraphraser = build_paraphrase_backend(args)

    print(f"\nRunning paraphrase robustness evaluation...")
    all_metrics = []
    examples = []

    for sample_idx, record in enumerate(sampled):
        if (sample_idx + 1) % 10 == 0:
            print(f"  {sample_idx+1}/{len(sampled)}...")

        try:
            para_record = paraphrase_record(record, paraphraser)
        except Exception as e:
            print(f"    Paraphrase failed for sample {sample_idx}: {e}")
            continue

        orig_attr = get_attribution_scores(model, record, records, collator, config, device)
        if orig_attr is None:
            continue

        para_attr = get_attribution_scores(model, para_record, records, collator, config, device)
        if para_attr is None:
            continue

        T = min(orig_attr.shape[0], para_attr.shape[0])
        S = min(orig_attr.shape[1], para_attr.shape[1])
        orig_attr_trunc = orig_attr[:T, :S]
        para_attr_trunc = para_attr[:T, :S]

        metrics = compute_robustness_metrics(
            orig_attr_trunc,
            para_attr_trunc,
            top_k=args.top_k,
        )
        metrics["sample_idx"] = sample_idx
        metrics["family"] = record.get("family", "unknown")
        metrics["has_implicit"] = any(
            t.get("implicit_trigger") for t in record.get("turns", [])
        )
        all_metrics.append(metrics)

        if len(examples) < args.save_examples:
            orig_turns = [t["text"] for t in record["turns"] if t["role"] == "user"]
            para_turns = [t["text"] for t in para_record["turns"] if t["role"] == "user"]
            examples.append({
                "original_turns": orig_turns,
                "paraphrase_turns": para_turns,
                "orig_turn_attr": orig_attr_trunc.mean(dim=1).tolist(),
                "para_turn_attr": para_attr_trunc.mean(dim=1).tolist(),
                "metrics": metrics,
            })

    if not all_metrics:
        print("ERROR: No valid samples processed.")
        return

    turn_rhos = [m["turn_rho"] for m in all_metrics]
    tok_rhos = [m["token_rho"] for m in all_metrics]
    stabilities = [m["topk_stability"] for m in all_metrics]
    pivot_stables = [m["pivot_stable"] for m in all_metrics]

    print(f"\n{'='*60}")
    print(f"  Paraphrase Robustness Results")
    print(f"{'='*60}")
    print(f"  N samples evaluated: {len(all_metrics)}")
    print(f"  Turn-level Spearman ρ:  {np.mean(turn_rhos):.4f} ± {np.std(turn_rhos):.4f}")
    print(f"  Token-level Spearman ρ: {np.mean(tok_rhos):.4f} ± {np.std(tok_rhos):.4f}")
    print(f"  Top-{int(args.top_k*100)}% token stability: {np.mean(stabilities):.4f}")
    print(f"  Pivot turn stability:   {np.mean(pivot_stables):.4f}")

    implicit_metrics = [m for m in all_metrics if m["has_implicit"]]
    explicit_metrics = [m for m in all_metrics if not m["has_implicit"]]
    if implicit_metrics:
        print(f"\n  Implicit triggers (n={len(implicit_metrics)}):")
        print(f"    Turn ρ: {np.mean([m['turn_rho'] for m in implicit_metrics]):.4f}")
        print(f"    Tok  ρ: {np.mean([m['token_rho'] for m in implicit_metrics]):.4f}")
    if explicit_metrics:
        print(f"\n  Explicit triggers (n={len(explicit_metrics)}):")
        print(f"    Turn ρ: {np.mean([m['turn_rho'] for m in explicit_metrics]):.4f}")
        print(f"    Tok  ρ: {np.mean([m['token_rho'] for m in explicit_metrics]):.4f}")

    mean_turn_rho = np.mean(turn_rhos)
    if mean_turn_rho >= 0.6:
        interp = "Strong: attribution tracks semantic content across paraphrase."
    elif mean_turn_rho >= 0.4:
        interp = "Moderate: attribution partially tracks semantics; some surface sensitivity."
    else:
        interp = "Weak: attribution may be sensitive to surface wording."
    print(f"\n  Interpretation: {interp}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "n_samples": len(all_metrics),
        "paraphrase_model": args.paraphrase_model,
        "vllm_model": args.vllm_model,
        "subset": args.subset or "full",
        "aggregate": {
            "turn_rho_mean": float(np.mean(turn_rhos)),
            "turn_rho_std": float(np.std(turn_rhos)),
            "token_rho_mean": float(np.mean(tok_rhos)),
            "token_rho_std": float(np.std(tok_rhos)),
            "topk_stability": float(np.mean(stabilities)),
            "pivot_stability": float(np.mean(pivot_stables)),
        },
        "implicit_subset": {
            "n": len(implicit_metrics),
            "turn_rho": float(np.mean([m["turn_rho"] for m in implicit_metrics])) if implicit_metrics else None,
            "token_rho": float(np.mean([m["token_rho"] for m in implicit_metrics])) if implicit_metrics else None,
        } if implicit_metrics else None,
        "explicit_subset": {
            "n": len(explicit_metrics),
            "turn_rho": float(np.mean([m["turn_rho"] for m in explicit_metrics])) if explicit_metrics else None,
            "token_rho": float(np.mean([m["token_rho"] for m in explicit_metrics])) if explicit_metrics else None,
        } if explicit_metrics else None,
        "per_sample_metrics": all_metrics,
        "examples": examples,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()