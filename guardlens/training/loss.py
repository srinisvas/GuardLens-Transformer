"""Loss functions for GuardLens training — v11 dataset compatible."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from guardlens.config import GuardLensConfig


class GuardLensLoss(nn.Module):
    """Combined phased loss with sample/span weighting and pivot supervision."""

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config
        self.pos_weight = None

    def set_pos_weight(self, pos_weight: float):
        self.pos_weight = torch.tensor([pos_weight])

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        token_labels: torch.Tensor,
        span_weights: torch.Tensor = None,
        sample_weights: torch.Tensor = None,
        pivot_labels: torch.Tensor = None,
        pivot_kind_labels: torch.Tensor = None,
        phase: int = 1,
        lambda_cls: float = 1.0,
        lambda_attr: float = 1.0,
        lambda_cf: float = 0.5,
        lambda_pivot: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        device = outputs["cls_logits"].device

        pw = self.pos_weight.to(device) if self.pos_weight is not None else None
        l_cls_raw = F.binary_cross_entropy_with_logits(
            outputs["cls_logits"], labels.float(), pos_weight=pw, reduction="none"
        )
        if sample_weights is not None:
            sw = sample_weights.to(device)
            l_cls = (l_cls_raw * sw).sum() / sw.sum().clamp(min=1e-8)
        else:
            l_cls = l_cls_raw.mean()
        losses["cls"] = l_cls
        total = l_cls if phase == 1 else lambda_cls * l_cls

        if phase >= 2 and outputs["attr_logits"] is not None:
            attr_logits = outputs["attr_logits"]
            valid = token_labels >= 0
            if valid.any():
                if span_weights is not None:
                    attr_sw = span_weights.to(device)
                    valid = valid & (attr_sw > 0)
                if valid.any():
                    attr_loss_raw = F.binary_cross_entropy_with_logits(
                        attr_logits[valid], token_labels[valid].float(), reduction="none"
                    )
                    if span_weights is not None:
                        tier_w = attr_sw[valid]
                        l_attr = (attr_loss_raw * tier_w).sum() / tier_w.sum().clamp(min=1e-8)
                    else:
                        l_attr = attr_loss_raw.mean()
                    losses["attr"] = l_attr
                    total = total + lambda_attr * l_attr
                    with torch.no_grad():
                        for tier_name, tier_val in [("causal", 1), ("incidental", 0)]:
                            mask = token_labels[valid] == tier_val
                            if mask.any():
                                losses[f"attr_{tier_name}"] = attr_loss_raw[mask].mean()

        if (
            phase >= 2
            and outputs.get("pivot_logits") is not None
            and pivot_labels is not None
        ):
            pivot_targets = pivot_labels.to(device)
            valid_pivot_target = pivot_targets != -1

            # A repaired minibatch can consist entirely of malicious records whose
            # pivot is unknown. Calling cross_entropy with every target ignored can
            # return NaN, so skip this auxiliary loss for such a minibatch.
            if valid_pivot_target.any():
                l_pivot = F.cross_entropy(
                    outputs["pivot_logits"][valid_pivot_target],
                    pivot_targets[valid_pivot_target],
                )
                losses["pivot"] = l_pivot
                total = total + lambda_pivot * l_pivot

                if (
                    outputs.get("pivot_kind_logits") is not None
                    and pivot_kind_labels is not None
                ):
                    no_pivot_class = outputs["pivot_logits"].size(1) - 1
                    has_pivot = (
                        (pivot_targets >= 0)
                        & (pivot_targets < no_pivot_class)
                    )
                    if has_pivot.any():
                        l_pkind = F.cross_entropy(
                            outputs["pivot_kind_logits"][has_pivot],
                            pivot_kind_labels.to(device)[has_pivot],
                        )
                        losses["pivot_kind"] = l_pkind
                        total = total + 0.1 * l_pkind

        losses["total"] = total
        return losses

    def counterfactual_loss(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        cf_progress: float = 0.0,
    ) -> torch.Tensor:
        attr_probs = outputs["attr_probs"]
        cls_logits = outputs["cls_logits"]
        token_embeds = outputs["token_embeds"]
        labels = batch["labels"]

        adv_mask = labels == 1
        if not adv_mask.any():
            return torch.tensor(0.0, device=cls_logits.device, requires_grad=True)

        threshold = 0.3 + 0.2 * cf_progress
        soft_mask = 1.0 - attr_probs
        hard_mask = (attr_probs < threshold).float()
        cf_mask = hard_mask + (soft_mask - soft_mask.detach())
        cf_outputs = model.forward_cf(
            token_embeds=token_embeds,
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
            role_ids=batch["role_ids"],
            attribution_mask=cf_mask,
        )
        original_prob = torch.sigmoid(cls_logits[adv_mask])
        cf_prob = torch.sigmoid(cf_outputs["cls_logits"][adv_mask])
        return F.relu(cf_prob - original_prob + self.config.cf_delta_threshold).mean()
