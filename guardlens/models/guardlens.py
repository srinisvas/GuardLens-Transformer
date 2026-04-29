"""GuardLens: the full model with backbone, cross-turn attention, dual heads, and gated fusion."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from guardlens.config import GuardLensConfig
from guardlens.models.components import (
    CrossTurnAttention,
    ClassificationHead,
    AttributionHead,
)


class GuardLens(nn.Module):
    """
    Hierarchical transformer for multi-turn adversarial prompt
    detection with causal token attribution.

    Forward pass:
      1. Encode each turn with frozen backbone
      2. Flattened token-level cross-turn attention
      3. Attribution head: per-token causal scores
      4. Gated fusion: attribution weights gate classification input
      5. Classification head: conversation-level prediction
    """

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config

        self.backbone = None
        self.backbone_loaded = False

        self.cross_turn = CrossTurnAttention(config)
        self.cls_head = ClassificationHead(config)
        self.attr_head = AttributionHead(config)

        if config.use_gated_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(config.cross_turn_dim, config.cross_turn_dim),
                nn.Sigmoid(),
            )

    def setup_backbone(self):
        """Load the pretrained backbone. Called once before training."""
        if self.backbone_loaded:
            return
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(
            self.config.backbone_name, output_hidden_states=False,
        )
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.backbone_loaded = True

    def encode_turns(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each turn independently with the frozen backbone.

        Args:
            input_ids: [B, T, S]
            attention_mask: [B, T, S]
        Returns:
            [B, T, S, D_backbone]
        """
        B, T, S = input_ids.shape
        flat_ids = input_ids.reshape(B * T, S)
        flat_mask = attention_mask.reshape(B * T, S)

        ctx = torch.no_grad() if self.config.freeze_backbone else torch.enable_grad()
        with ctx:
            outputs = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
            hidden = outputs.last_hidden_state

        # DeBERTa may load in float16 on some systems. Trainable layers
        # are float32, so cast here to avoid dtype mismatch.
        return hidden.reshape(B, T, S, -1).float()

    def pool_with_mask(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool over valid tokens. [B, T, S, D] -> [B, D]."""
        B, T, S, D = embeds.shape
        flat = embeds.reshape(B, T * S, D)
        flat_mask = mask.reshape(B, T * S).unsqueeze(-1).float()
        summed = (flat * flat_mask).sum(dim=1)
        counts = flat_mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
        role_ids: torch.Tensor,
        compute_attribution: bool = True,
        attribution_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            attribution_mask: [B, T, S] optional representation-level mask.
                1=keep, 0=zero the embedding. Used by causal evaluation
                metrics (deviation drop, flip rate, necessity, sufficiency).

        Returns token_embeds in the output dict so forward_cf can
        reuse them without re-running the backbone.
        """
        # 1. Encode turns
        token_embeds = self.encode_turns(input_ids, attention_mask)

        # Apply representation-level counterfactual mask if provided
        if attribution_mask is not None:
            token_embeds = token_embeds * attribution_mask.unsqueeze(-1).float()

        # 2. Cross-turn attention
        cross_embeds = self.cross_turn(
            token_embeds, attention_mask, turn_mask, role_ids,
        )

        # 3. Attribution
        attr_logits = None
        attr_probs = None
        if compute_attribution:
            attr_logits = self.attr_head(cross_embeds)
            attr_probs = torch.sigmoid(attr_logits / self.config.fusion_temperature)

        # 4. Pooling
        valid_mask = attention_mask * turn_mask.unsqueeze(-1)
        pooled = self.pool_with_mask(cross_embeds, valid_mask)

        # 5. Gated fusion
        gated = None
        if self.config.use_gated_fusion and attr_probs is not None:
            gate_weights = self.fusion_gate(cross_embeds)
            weighted = cross_embeds * attr_probs.unsqueeze(-1) * gate_weights
            gated = self.pool_with_mask(weighted, valid_mask)

        # 6. Classification
        cls_logits = self.cls_head(pooled, gated)

        return {
            "cls_logits": cls_logits.squeeze(-1),
            "attr_logits": attr_logits,
            "attr_probs": attr_probs,
            "pooled": pooled,
            "token_embeds": token_embeds.detach(),  # Cached for forward_cf
        }

    def forward_cf(
        self,
        token_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
        role_ids: torch.Tensor,
        attribution_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Lightweight counterfactual forward pass.

        Reuses pre-computed backbone embeddings from forward().
        Only runs cross-turn attention + classification head.
        Skips: backbone encoding, attribution head, gated fusion.

        Cost: ~10% of a full forward pass.
        """
        # Apply representation-level mask to cached embeddings
        masked_embeds = token_embeds * attribution_mask.unsqueeze(-1).float()

        # Cross-turn attention (trainable, receives gradients)
        cross_embeds = self.cross_turn(
            masked_embeds, attention_mask, turn_mask, role_ids,
        )

        # Pool + classify (no fusion -- attribution is disabled)
        valid_mask = attention_mask * turn_mask.unsqueeze(-1)
        pooled = self.pool_with_mask(cross_embeds, valid_mask)
        cls_logits = self.cls_head(pooled, None)

        return {
            "cls_logits": cls_logits.squeeze(-1),
        }