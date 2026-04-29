"""Loss functions for GuardLens training."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from guardlens.config import GuardLensConfig


class GuardLensLoss(nn.Module):
    """
    Combined loss with phased training support.

    Phase 1: L_cls only
    Phase 2: L_cls + lambda_attr * L_attr
    Phase 3: L_cls + lambda_attr * L_attr + lambda_cf * L_cf
    """

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config
        self.cls_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        token_labels: torch.Tensor,
        phase: int = 1,
        lambda_cls: float = 1.0,
        lambda_attr: float = 1.0,
        lambda_cf: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # Classification loss
        l_cls = self.cls_loss(outputs["cls_logits"], labels.float())
        losses["cls"] = l_cls

        if phase == 1:
            # Phase 1: classification only (full weight to bootstrap)
            total = l_cls
        else:
            # Phase 2+: classification is auxiliary, attribution is primary
            total = lambda_cls * l_cls

        # Attribution loss (phase 2+)
        if phase >= 2 and outputs["attr_logits"] is not None:
            attr_logits = outputs["attr_logits"]
            valid = token_labels >= 0
            if valid.any():
                l_attr = F.binary_cross_entropy_with_logits(
                    attr_logits[valid], token_labels[valid].float(),
                )
                losses["attr"] = l_attr
                total = total + lambda_attr * l_attr

        losses["total"] = total
        return losses

    def counterfactual_loss(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        cf_progress: float = 0.0,
    ) -> torch.Tensor:
        """
        Counterfactual consistency loss using forward_cf.

        Reuses cached backbone embeddings from the main forward pass.
        Only re-runs cross-turn attention + classification head (~10%
        of full forward cost).

        Args:
            cf_progress: 0.0-1.0 progress through phase 3, controls
                the attribution threshold curriculum.
        """
        attr_probs = outputs["attr_probs"]
        cls_logits = outputs["cls_logits"]
        token_embeds = outputs["token_embeds"]  # Cached from forward()
        labels = batch["labels"]

        adv_mask = labels == 1
        if not adv_mask.any():
            return torch.tensor(0.0, device=cls_logits.device, requires_grad=True)

        # Curriculum threshold: start permissive (0.3), tighten to 0.5
        # Early in phase 3, attr_probs are noisy so aggressive masking
        # would destabilize training. Gradually tightening lets the
        # attribution head stabilize before being held accountable.
        threshold = 0.3 + 0.2 * cf_progress

        # Straight-through estimator
        soft_mask = 1.0 - attr_probs
        hard_mask = (attr_probs < threshold).float()
        cf_mask = hard_mask + (soft_mask - soft_mask.detach())

        # Lightweight forward: reuses backbone embeddings, skips
        # attribution head and gated fusion
        cf_outputs = model.forward_cf(
            token_embeds=token_embeds,
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
            role_ids=batch["role_ids"],
            attribution_mask=cf_mask,
        )

        original_prob = torch.sigmoid(cls_logits[adv_mask])
        cf_prob = torch.sigmoid(cf_outputs["cls_logits"][adv_mask])

        l_cf = F.relu(
            cf_prob - original_prob + self.config.cf_delta_threshold
        ).mean()

        return l_cf