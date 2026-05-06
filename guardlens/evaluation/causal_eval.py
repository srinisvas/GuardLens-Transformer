import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def token_f1(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> Dict[str, float]:

    tp = ((pred_mask == 1) & (true_mask == 1)).sum().item()
    fp = ((pred_mask == 1) & (true_mask == 0)).sum().item()
    fn = ((pred_mask == 0) & (true_mask == 1)).sum().item()
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-8, p + r)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _get_prob(model, batch, device, attribution_mask=None):
    model.eval()
    with torch.no_grad():
        kwargs = dict(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            turn_mask=batch["turn_mask"].to(device),
            role_ids=batch["role_ids"].to(device),
            compute_attribution=False,
        )
        if attribution_mask is not None:
            kwargs["attribution_mask"] = attribution_mask.to(device)
        out = model(**kwargs)
    return torch.sigmoid(out["cls_logits"])


def _build_topk_mask(scores: torch.Tensor, valid: torch.Tensor,
                     k_frac: float) -> torch.Tensor:
    flat_scores = scores[valid.bool()].flatten()
    n_tokens = flat_scores.numel()
    if n_tokens == 0:
        return torch.ones_like(scores)
    k = max(1, int(n_tokens * k_frac))
    threshold = flat_scores.topk(k).values[-1]
    return (scores < threshold).float()


def _build_sufficiency_mask(scores: torch.Tensor, valid: torch.Tensor,
                            k_frac: float,
                            context_window: int = 2) -> torch.Tensor:

    flat_scores = scores[valid.bool()].flatten()
    n_tokens = flat_scores.numel()
    if n_tokens == 0:
        return torch.zeros_like(scores)
    k = max(1, int(n_tokens * k_frac))
    threshold = flat_scores.topk(k).values[-1]

    # Core mask: top-k tokens
    core = (scores >= threshold).float()

    # Expand with context window along the S (token) dimension
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


def deviation_drop(
    model,
    batch: Dict,
    attr_scores: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.15,
    external_model=None,
) -> Dict[str, float]:

    eval_model = external_model if external_model is not None else model
    labels = batch["labels"]
    adv_idx = (labels == 1).nonzero(as_tuple=True)[0]
    if len(adv_idx) == 0:
        return {"absolute_drop": 0.0, "relative_drop": 0.0, "n_tested": 0}

    drops = []
    for i in adv_idx:
        valid_i = (batch["attention_mask"][i] * batch["turn_mask"][i].unsqueeze(-1))
        orig_prob = _get_prob(eval_model, _single_batch(batch, i), device)[0].item()
        if orig_prob < 0.5:
            continue

        mask = _build_topk_mask(attr_scores[i], valid_i, k_frac)
        masked_prob = _get_prob(
            eval_model, _single_batch(batch, i), device,
            attribution_mask=mask.unsqueeze(0),
        )[0].item()

        drops.append(orig_prob - masked_prob)

    if not drops:
        return {"absolute_drop": 0.0, "relative_drop": 0.0, "n_tested": 0}

    return {
        "absolute_drop": float(np.mean(drops)),
        "relative_drop": float(np.mean(drops)) / max(1e-8, float(np.mean([d + 0.5 for d in drops]))),
        "n_tested": len(drops),
    }


def flip_rate(
    model,
    loader: DataLoader,
    attribution_fn: Callable,
    top_k_fractions: List[float],
    device: torch.device,
    external_model=None,
) -> Dict[str, float]:

    eval_model = external_model if external_model is not None else model

    flip_counts = {f"flip@{int(k*100)}%": 0 for k in top_k_fractions}
    total_adversarial = 0

    model.eval()
    for batch in loader:
        labels = batch["labels"]
        attr_scores = attribution_fn(model, batch, device)
        attention_mask = batch["attention_mask"].to(device)
        turn_mask = batch["turn_mask"].to(device)

        B = labels.size(0)
        for i in range(B):
            if labels[i] != 1:
                continue

            orig_prob = _get_prob(eval_model, _single_batch(batch, i), device)[0].item()
            if orig_prob < 0.5:
                continue

            total_adversarial += 1
            valid_i = (batch["attention_mask"][i] * batch["turn_mask"][i].unsqueeze(-1))

            for k_frac in top_k_fractions:
                mask = _build_topk_mask(attr_scores[i], valid_i, k_frac)
                masked_prob = _get_prob(
                    eval_model, _single_batch(batch, i), device,
                    attribution_mask=mask.unsqueeze(0),
                )[0].item()

                if masked_prob < 0.5:
                    flip_counts[f"flip@{int(k_frac*100)}%"] += 1

    for k in flip_counts:
        flip_counts[k] = flip_counts[k] / max(1, total_adversarial)

    flip_counts["n_adversarial"] = total_adversarial
    return flip_counts


def necessity_test(
    model,
    batch: Dict,
    attr_scores: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.15,
    external_model=None,
) -> Dict[str, float]:

    eval_model = external_model if external_model is not None else model
    labels = batch["labels"]
    necessary = 0
    total = 0

    for i in range(labels.size(0)):
        if labels[i] != 1:
            continue
        orig_prob = _get_prob(eval_model, _single_batch(batch, i), device)[0].item()
        if orig_prob < 0.5:
            continue

        valid_i = batch["attention_mask"][i] * batch["turn_mask"][i].unsqueeze(-1)
        mask = _build_topk_mask(attr_scores[i], valid_i, k_frac)
        masked_prob = _get_prob(
            eval_model, _single_batch(batch, i), device,
            attribution_mask=mask.unsqueeze(0),
        )[0].item()

        total += 1
        if masked_prob < 0.5:
            necessary += 1

    return {"necessity_rate": necessary / max(1, total), "n_tested": total}


def sufficiency_test(
    model,
    batch: Dict,
    attr_scores: torch.Tensor,
    device: torch.device,
    k_frac: float = 0.15,
    context_window: int = 2,
    external_model=None,
) -> Dict[str, float]:

    eval_model = external_model if external_model is not None else model
    labels = batch["labels"]
    sufficient = 0
    total = 0

    for i in range(labels.size(0)):
        if labels[i] != 1:
            continue

        valid_i = batch["attention_mask"][i] * batch["turn_mask"][i].unsqueeze(-1)
        suff_mask = _build_sufficiency_mask(
            attr_scores[i], valid_i, k_frac, context_window,
        )
        prob = _get_prob(
            eval_model, _single_batch(batch, i), device,
            attribution_mask=suff_mask.unsqueeze(0),
        )[0].item()

        total += 1
        if prob >= 0.5:
            sufficient += 1

    return {"sufficiency_rate": sufficient / max(1, total), "n_tested": total}


def minimal_trigger_size(
    model,
    batch: Dict,
    attr_scores: torch.Tensor,
    device: torch.device,
    external_model=None,
) -> List[int]:
    eval_model = external_model if external_model is not None else model
    labels = batch["labels"]
    sizes = []

    for i in range(labels.size(0)):
        if labels[i] != 1:
            continue
        orig_prob = _get_prob(eval_model, _single_batch(batch, i), device)[0].item()
        if orig_prob < 0.5:
            continue

        valid_i = (batch["attention_mask"][i] * batch["turn_mask"][i].unsqueeze(-1)).bool()
        flat_scores = attr_scores[i][valid_i].flatten()
        if flat_scores.numel() == 0:
            continue

        sorted_idx = flat_scores.argsort(descending=True)
        n = flat_scores.numel()

        # Binary search for minimal k
        lo, hi, best_k = 1, n, n
        while lo <= hi:
            mid = (lo + hi) // 2
            threshold = flat_scores[sorted_idx[mid - 1]]
            mask = (attr_scores[i] < threshold).float()
            prob = _get_prob(
                eval_model, _single_batch(batch, i), device,
                attribution_mask=mask.unsqueeze(0),
            )[0].item()
            if prob < 0.5:
                best_k = mid
                hi = mid - 1
            else:
                lo = mid + 1

        sizes.append(best_k)

    return sizes


def pivot_turn_accuracy(
    attr_scores: torch.Tensor,
    batch: Dict,
) -> Dict[str, float]:
    correct = 0
    within_1 = 0
    total = 0

    for i, meta in enumerate(batch["metadata"]):
        true_pivot = meta.get("pivot_turn_id")
        if true_pivot is None:
            continue

        scores_i = attr_scores[i]
        turn_mask = batch["turn_mask"][i]
        attn_mask = batch["attention_mask"][i]

        T = turn_mask.size(0)
        turn_scores = []
        for t in range(T):
            if turn_mask[t] == 0:
                turn_scores.append(-1.0)
                continue
            valid = attn_mask[t].bool()
            if valid.any():
                turn_scores.append(scores_i[t][valid].mean().item())
            else:
                turn_scores.append(-1.0)

        pred_pivot = int(np.argmax(turn_scores))
        total += 1
        if pred_pivot == true_pivot:
            correct += 1
        if abs(pred_pivot - true_pivot) <= 1:
            within_1 += 1

    return {"exact_match": correct / max(1, total),
            "within_1": within_1 / max(1, total), "total": total}


def _single_batch(batch: Dict, idx: int) -> Dict:
    return {
        "input_ids": batch["input_ids"][idx:idx+1],
        "attention_mask": batch["attention_mask"][idx:idx+1],
        "turn_mask": batch["turn_mask"][idx:idx+1],
        "role_ids": batch["role_ids"][idx:idx+1],
        "labels": batch["labels"][idx:idx+1],
        "token_labels": batch["token_labels"][idx:idx+1],
        "metadata": [batch["metadata"][idx]],
    }

def guardlens_attribution(model, batch, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            turn_mask=batch["turn_mask"].to(device),
            role_ids=batch["role_ids"].to(device),
            compute_attribution=True,
        )
    if out["attr_probs"] is not None:
        return out["attr_probs"].cpu()
    return torch.zeros_like(batch["attention_mask"].float())


def attention_attribution(model, batch, device) -> torch.Tensor:

    model.eval()
    B, T, S = batch["input_ids"].shape
    captured_weights = []

    def hook_fn(module, args, output):

        if isinstance(output, tuple) and len(output) >= 2:
            if output[1] is not None:
                captured_weights.append(output[1].detach())

    hooks = []
    for layer in model.cross_turn.transformer.layers:
        if hasattr(layer, 'self_attn'):
            old_flag = layer.self_attn.batch_first
            h = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(h)

    try:
        with torch.no_grad():
            # Need to ensure attention weights are returned
            token_embeds = model.encode_turns(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            x = model.cross_turn.input_proj(token_embeds)
            turn_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            turn_pos = model.cross_turn.turn_pos(
                turn_idx, batch["role_ids"].to(device),
            )
            x = x + turn_pos.unsqueeze(2)
            x_flat = x.reshape(B, T * S, -1)

            flat_mask = (
                batch["attention_mask"].to(device)
                * batch["turn_mask"].to(device).unsqueeze(-1)
            ).reshape(B, T * S)
            padding_mask = flat_mask == 0

            # Forward through transformer (hooks capture attention)
            model.cross_turn.transformer(x_flat, src_key_padding_mask=padding_mask)

    finally:
        for h in hooks:
            h.remove()

    if captured_weights:

        avg_attn = torch.stack(captured_weights).mean(dim=0)  # [B, heads, T*S, T*S]
        token_importance = avg_attn.mean(dim=1).sum(dim=1)  # [B, T*S]
        token_importance = token_importance.reshape(B, T, S)
        # Normalize per sample
        for i in range(B):
            valid = batch["attention_mask"][i].to(device).bool()
            ti = token_importance[i]
            ti_valid = ti[valid]
            if ti_valid.numel() > 0 and ti_valid.max() > ti_valid.min():
                ti = (ti - ti_valid.min()) / (ti_valid.max() - ti_valid.min() + 1e-8)
            token_importance[i] = ti
        return token_importance.cpu()

    # Fallback: representation norm change (labeled as proxy, not attention)
    return _representation_change_proxy(model, batch, device)


def _representation_change_proxy(model, batch, device) -> torch.Tensor:
    B, T, S = batch["input_ids"].shape
    with torch.no_grad():
        token_embeds = model.encode_turns(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        projected = model.cross_turn.input_proj(token_embeds)
        cross_embeds = model.cross_turn(
            token_embeds, batch["attention_mask"].to(device),
            batch["turn_mask"].to(device), batch["role_ids"].to(device),
        )
        diff = (cross_embeds - projected).norm(dim=-1)
        for i in range(B):
            valid = batch["attention_mask"][i].to(device).bool()
            d = diff[i]
            d_valid = d[valid]
            if d_valid.numel() > 0 and d_valid.max() > d_valid.min():
                d = (d - d_valid.min()) / (d_valid.max() - d_valid.min() + 1e-8)
            diff[i] = d
    return diff.cpu()


def integrated_gradients_attribution(
    model, batch, device, n_steps: int = 20,
) -> torch.Tensor:

    model.eval()
    B, T, S = batch["input_ids"].shape

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    turn_mask = batch["turn_mask"].to(device)
    role_ids = batch["role_ids"].to(device)

    flat_ids = input_ids.reshape(B * T, S)
    flat_mask = attention_mask.reshape(B * T, S)

    # Get the embedding layer
    if hasattr(model.backbone, 'embeddings'):
        embed_layer = model.backbone.embeddings
    else:
        # Fallback to grad x input
        return gradient_x_input_attribution(model, batch, device)

    # Get actual embeddings (baseline = zeros)
    with torch.no_grad():
        actual_embeds = embed_layer(flat_ids)  # [B*T, S, D]
    baseline = torch.zeros_like(actual_embeds)

    # Accumulate gradients along the path
    accumulated_grads = torch.zeros_like(actual_embeds)

    for step in range(n_steps):
        alpha = (step + 0.5) / n_steps  # Midpoint rule
        interpolated = baseline + alpha * (actual_embeds - baseline)
        interpolated = interpolated.detach().requires_grad_(True)

        # Forward through backbone
        backbone_out = model.backbone(
            inputs_embeds=interpolated, attention_mask=flat_mask,
        )
        hidden = backbone_out.last_hidden_state.float()
        token_embeds = hidden.reshape(B, T, S, -1)

        # Forward through trainable layers
        cross_embeds = model.cross_turn(
            token_embeds, attention_mask, turn_mask, role_ids,
        )
        valid_mask = attention_mask * turn_mask.unsqueeze(-1)
        pooled = model.pool_with_mask(cross_embeds, valid_mask)

        gated = None
        if model.config.use_gated_fusion:
            attr_logits = model.attr_head(cross_embeds)
            attr_probs = torch.sigmoid(attr_logits / model.config.fusion_temperature)
            gate_weights = model.fusion_gate(cross_embeds)
            weighted = cross_embeds * attr_probs.unsqueeze(-1) * gate_weights
            gated = model.pool_with_mask(weighted, valid_mask)

        cls_logits = model.cls_head(pooled, gated)
        prob = torch.sigmoid(cls_logits).sum()
        prob.backward()

        accumulated_grads += interpolated.grad.detach()
        interpolated.grad = None

    # IG = (actual - baseline) * avg_gradient
    ig = (actual_embeds - baseline) * (accumulated_grads / n_steps)
    ig_scores = ig.norm(dim=-1)  # [B*T, S]
    ig_scores = ig_scores.reshape(B, T, S).detach()

    # Normalize
    for i in range(B):
        valid = attention_mask[i].bool()
        g = ig_scores[i]
        g_valid = g[valid]
        if g_valid.numel() > 0 and g_valid.max() > g_valid.min():
            g = (g - g_valid.min()) / (g_valid.max() - g_valid.min() + 1e-8)
        ig_scores[i] = g

    return ig_scores.cpu()


def gradient_x_input_attribution(model, batch, device) -> torch.Tensor:
    model.eval()
    B, T, S = batch["input_ids"].shape

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    turn_mask = batch["turn_mask"].to(device)
    role_ids = batch["role_ids"].to(device)

    flat_ids = input_ids.reshape(B * T, S)
    flat_mask = attention_mask.reshape(B * T, S)

    if hasattr(model.backbone, 'embeddings'):
        embed_layer = model.backbone.embeddings
    else:
        return torch.zeros(B, T, S)

    with torch.enable_grad():
        embeds = embed_layer(flat_ids)
        embeds = embeds.detach().requires_grad_(True)

        backbone_out = model.backbone(inputs_embeds=embeds, attention_mask=flat_mask)
        hidden = backbone_out.last_hidden_state.float()
        token_embeds = hidden.reshape(B, T, S, -1)

        cross_embeds = model.cross_turn(
            token_embeds, attention_mask, turn_mask, role_ids,
        )
        valid_mask = attention_mask * turn_mask.unsqueeze(-1)
        pooled = model.pool_with_mask(cross_embeds, valid_mask)

        gated = None
        if model.config.use_gated_fusion:
            attr_logits = model.attr_head(cross_embeds)
            attr_probs = torch.sigmoid(attr_logits / model.config.fusion_temperature)
            gate_weights = model.fusion_gate(cross_embeds)
            weighted = cross_embeds * attr_probs.unsqueeze(-1) * gate_weights
            gated = model.pool_with_mask(weighted, valid_mask)

        cls_logits = model.cls_head(pooled, gated)
        prob = torch.sigmoid(cls_logits).sum()
        prob.backward()

        gxi = (embeds.grad * embeds).norm(dim=-1)
        gxi = gxi.reshape(B, T, S).detach()

        for i in range(B):
            valid = attention_mask[i].bool()
            g = gxi[i]
            g_valid = g[valid]
            if g_valid.numel() > 0 and g_valid.max() > g_valid.min():
                g = (g - g_valid.min()) / (g_valid.max() - g_valid.min() + 1e-8)
            gxi[i] = g

    return gxi.cpu()


def surface_risk_attribution(model, batch, device, tokenizer=None) -> torch.Tensor:

    RISK_KEYWORDS = {
        "bypass", "exploit", "ignore", "override", "evasion", "evade",
        "inject", "injection", "jailbreak", "unsafe", "unrestricted",
        "detectable", "convincing", "realistic", "blend", "plausible",
        "success", "effective", "credential", "phishing", "malware",
        "exfiltrate", "manipulate", "persuasive", "stealth",
        "payload", "disguise", "impersonate", "pretexting",
    }

    B, T, S = batch["input_ids"].shape
    scores = torch.zeros(B, T, S)

    if tokenizer is not None:
        for i in range(B):
            for t in range(T):
                if batch["turn_mask"][i][t] == 0:
                    continue
                ids = batch["input_ids"][i, t]
                mask = batch["attention_mask"][i, t]
                for s_idx in range(S):
                    if mask[s_idx] == 0:
                        continue
                    token_text = tokenizer.decode([ids[s_idx].item()]).strip().lower()
                    # Check if token matches or is substring of a keyword
                    if any(kw.startswith(token_text) or token_text in kw
                           for kw in RISK_KEYWORDS if len(token_text) >= 3):
                        scores[i, t, s_idx] = 0.8
                    elif any(token_text.startswith(kw[:3])
                             for kw in RISK_KEYWORDS if len(token_text) >= 3):
                        scores[i, t, s_idx] = 0.4
                    else:
                        scores[i, t, s_idx] = 0.05
    else:
        scores = 0.1 * batch["attention_mask"].float()

    scores = scores * batch["attention_mask"].float()
    return scores


def random_attribution(model, batch, device) -> torch.Tensor:
    scores = torch.rand_like(batch["attention_mask"].float())
    return scores * batch["attention_mask"].float()


# Registry
ATTRIBUTION_METHODS = {
    "guardlens": guardlens_attribution,
    "attention": attention_attribution,
    "integrated_gradients": integrated_gradients_attribution,
    "grad_x_input": gradient_x_input_attribution,
    "surface_risk": surface_risk_attribution,
    "random": random_attribution,
}

def run_causal_evaluation(
    model,
    loader: DataLoader,
    device: torch.device,
    methods: List[str] = None,
    top_k_fractions: List[float] = None,
    external_model=None,
    context_window: int = 2,
    tokenizer=None,
) -> Dict:

    if methods is None:
        methods = ["guardlens", "attention", "integrated_gradients",
                    "grad_x_input", "surface_risk", "random"]
    if top_k_fractions is None:
        top_k_fractions = [0.05, 0.10, 0.15, 0.20]

    eval_model = external_model if external_model is not None else model
    using_external = external_model is not None
    print(f"  Evaluator: {'external model' if using_external else 'same model (self-eval)'}")

    model.eval()
    if external_model is not None:
        external_model.eval()

    results = {}

    for method_name in methods:
        print(f"\n  === {method_name} ===")
        attr_fn = ATTRIBUTION_METHODS.get(method_name)
        if attr_fn is None:
            print(f"    Skipped (not registered)")
            continue

        method_results = {
            "token_f1": {"precision": 0, "recall": 0, "f1": 0},
            "deviation_drops": {},
            "necessity": {},
            "sufficiency": {},
            "trigger_sizes": [],
            "pivot_turn": {},
        }

        all_f1 = []
        all_necessity = {k: [] for k in top_k_fractions}
        all_sufficiency = {k: [] for k in top_k_fractions}
        all_dev_drop = {k: [] for k in top_k_fractions}

        for batch in loader:
            # Get attribution
            if method_name in ("grad_x_input", "integrated_gradients"):
                with torch.enable_grad():
                    attr_scores = attr_fn(model, batch, device)
            elif method_name == "surface_risk":
                attr_scores = attr_fn(model, batch, device, tokenizer=tokenizer)
            else:
                attr_scores = attr_fn(model, batch, device)

            # Token F1 (secondary)
            token_labels = batch["token_labels"]
            pred_mask = (attr_scores > 0.5).long()
            valid = token_labels >= 0
            if valid.any():
                all_f1.append(token_f1(pred_mask[valid], token_labels[valid]))

            # Pivot turn
            piv = pivot_turn_accuracy(attr_scores, batch)
            if piv["total"] > 0:
                existing = method_results["pivot_turn"]
                for k in ("exact_match", "within_1"):
                    existing[k] = existing.get(k, 0) + piv[k] * piv["total"]
                existing["total"] = existing.get("total", 0) + piv["total"]

            # Per-k metrics
            for k_frac in top_k_fractions:
                dd = deviation_drop(
                    model, batch, attr_scores, device, k_frac, eval_model,
                )
                if dd["n_tested"] > 0:
                    all_dev_drop[k_frac].append(dd)

                nec = necessity_test(
                    model, batch, attr_scores, device, k_frac, eval_model,
                )
                if nec["n_tested"] > 0:
                    all_necessity[k_frac].append(nec)

                suf = sufficiency_test(
                    model, batch, attr_scores, device, k_frac,
                    context_window, eval_model,
                )
                if suf["n_tested"] > 0:
                    all_sufficiency[k_frac].append(suf)

            # Minimal trigger
            sizes = minimal_trigger_size(
                model, batch, attr_scores, device, eval_model,
            )
            method_results["trigger_sizes"].extend(sizes)

        # Aggregate token F1
        if all_f1:
            tp = sum(r["tp"] for r in all_f1)
            fp = sum(r["fp"] for r in all_f1)
            fn = sum(r["fn"] for r in all_f1)
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            method_results["token_f1"] = {
                "precision": p, "recall": r,
                "f1": 2 * p * r / max(1e-8, p + r),
            }

        # Aggregate pivot turn
        pt = method_results["pivot_turn"]
        if pt.get("total", 0) > 0:
            pt["exact_match"] /= pt["total"]
            pt["within_1"] /= pt["total"]

        # Aggregate per-k
        for k_frac in top_k_fractions:
            key = f"{int(k_frac*100)}%"
            # Deviation drop
            drops = [d["absolute_drop"] for d in all_dev_drop[k_frac]]
            method_results["deviation_drops"][key] = float(np.mean(drops)) if drops else 0.0
            # Necessity
            rates = [n["necessity_rate"] for n in all_necessity[k_frac]]
            method_results["necessity"][key] = float(np.mean(rates)) if rates else 0.0
            # Sufficiency
            rates = [s["sufficiency_rate"] for s in all_sufficiency[k_frac]]
            method_results["sufficiency"][key] = float(np.mean(rates)) if rates else 0.0

        # Trigger size
        sizes = method_results["trigger_sizes"]
        if sizes:
            method_results["trigger_size_stats"] = {
                "mean": float(np.mean(sizes)),
                "median": float(np.median(sizes)),
                "min": int(np.min(sizes)),
                "max": int(np.max(sizes)),
            }

        # Flip rate
        print(f"    Computing flip rates...")
        fr = flip_rate(model, loader, attr_fn, top_k_fractions, device, eval_model)
        method_results["flip_rates"] = fr

        results[method_name] = method_results

        # Print summary
        tf = method_results["token_f1"]
        pt = method_results.get("pivot_turn", {})
        print(f"    Token F1 (secondary): {tf['f1']:.4f}")
        print(f"    Pivot turn:  exact={pt.get('exact_match',0):.4f}  within1={pt.get('within_1',0):.4f}")
        print(f"    Dev drop:    {method_results['deviation_drops']}")
        print(f"    Necessity:   {method_results['necessity']}")
        print(f"    Sufficiency: {method_results['sufficiency']}")
        print(f"    Flip rates:  {method_results['flip_rates']}")
        if sizes:
            ts = method_results["trigger_size_stats"]
            print(f"    Trigger:     mean={ts['mean']:.1f}  median={ts['median']:.1f}")

    return results


def print_comparison_table(results: Dict):

    methods = list(results.keys())

    print("\n" + "=" * 100)
    print("  PRIMARY: Causal Attribution Comparison (model-grounded metrics)")
    print("=" * 100)
    header = (f"{'Method':<22} {'DevDrop@15%':>11} {'Flip@10%':>9} "
              f"{'Nec@15%':>9} {'Suf@15%':>9} {'TrigSize':>9} {'PivotAcc':>9}")
    print(header)
    print("-" * 100)

    for m in methods:
        r = results[m]
        dd = r["deviation_drops"].get("15%", 0)
        flip = r["flip_rates"].get("flip@10%", 0)
        nec = r["necessity"].get("15%", 0)
        suf = r["sufficiency"].get("15%", 0)
        trig = r.get("trigger_size_stats", {}).get("median", 0)
        piv = r.get("pivot_turn", {}).get("exact_match", 0)

        print(f"{m:<22} {dd:>11.4f} {flip:>9.4f} {nec:>9.4f} "
              f"{suf:>9.4f} {trig:>9.1f} {piv:>9.4f}")

    print()
    print("  SECONDARY: Token F1 (synthetic label overlap, de-emphasized)")
    print("-" * 50)
    for m in methods:
        tf = results[m]["token_f1"]
        print(f"  {m:<22} F1={tf['f1']:.4f}  P={tf['precision']:.4f}  R={tf['recall']:.4f}")
    print("=" * 100)