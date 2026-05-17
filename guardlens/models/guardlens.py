"""GuardLens: full model with backbone, cross-turn attention, dual heads, gated fusion, and pivot head."""

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

    v11 additions:
      - Pivot head: per-turn logit for pivot detection [B, T+1]
      - Pivot kind head: classifies pivot type [B, n_pivot_kinds]
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

        # Pivot head: predicts which turn is the pivot (or no-pivot)
        if config.use_pivot_head:
            self.pivot_head = nn.Sequential(
                nn.Linear(config.cross_turn_dim, config.cls_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.cls_hidden_dim, 1),
            )
            # No-pivot embedding (learned, appended as extra "turn")
            self.no_pivot_embedding = nn.Parameter(
                torch.randn(config.cross_turn_dim) * 0.02,
            )
            # Pivot kind classifier
            self.pivot_kind_head = nn.Sequential(
                nn.Linear(config.cross_turn_dim, config.cls_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.cls_hidden_dim, config.n_pivot_kinds),
            )

    def setup_backbone(self):
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

    def encode_turns(self, input_ids, attention_mask):
        B, T, S = input_ids.shape
        flat_ids = input_ids.reshape(B * T, S)
        flat_mask = attention_mask.reshape(B * T, S)

        ctx = torch.no_grad() if self.config.freeze_backbone else torch.enable_grad()
        with ctx:
            outputs = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
            hidden = outputs.last_hidden_state

        return hidden.reshape(B, T, S, -1).float()

    def pool_with_mask(self, embeds, mask):
        B, T, S, D = embeds.shape
        flat = embeds.reshape(B, T * S, D)
        flat_mask = mask.reshape(B, T * S).unsqueeze(-1).float()
        summed = (flat * flat_mask).sum(dim=1)
        counts = flat_mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def pool_per_turn(self, embeds, attention_mask, turn_mask):
        """Pool each turn independently: [B, T, S, D] -> [B, T, D]."""
        B, T, S, D = embeds.shape
        mask = (attention_mask * turn_mask.unsqueeze(-1)).float()  # [B, T, S]
        mask = mask.unsqueeze(-1)  # [B, T, S, 1]
        summed = (embeds * mask).sum(dim=2)  # [B, T, D]
        counts = mask.sum(dim=2).clamp(min=1)  # [B, T, 1]
        return summed / counts  # [B, T, D]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
        role_ids: torch.Tensor,
        compute_attribution: bool = True,
        attribution_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1. Encode turns
        token_embeds = self.encode_turns(input_ids, attention_mask)

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

        result = {
            "cls_logits": cls_logits.squeeze(-1),
            "attr_logits": attr_logits,
            "attr_probs": attr_probs,
            "pooled": pooled,
            "token_embeds": token_embeds.detach(),
        }

        # 7. Pivot head
        if self.config.use_pivot_head and compute_attribution:
            turn_pooled = self.pool_per_turn(
                cross_embeds, attention_mask, turn_mask,
            )  # [B, T, D]

            pivot_scores = self.pivot_head(turn_pooled).squeeze(-1)  # [B, T]
            # Mask padded turns
            pivot_scores = pivot_scores.masked_fill(turn_mask == 0, -1e9)

            # Append no-pivot score
            B = turn_pooled.size(0)
            no_pivot_score = self.no_pivot_embedding.unsqueeze(0).expand(B, -1)
            no_pivot_logit = self.pivot_head(no_pivot_score)  # [B, 1]
            pivot_logits = torch.cat([pivot_scores, no_pivot_logit], dim=1)  # [B, T+1]

            result["pivot_logits"] = pivot_logits

            # Pivot kind: use the predicted pivot turn's embedding
            # During training, use the ground-truth pivot turn
            # During inference, use argmax of pivot_logits
            pivot_kind_logits = self.pivot_kind_head(pooled)  # [B, n_pivot_kinds]
            result["pivot_kind_logits"] = pivot_kind_logits

        return result

    def forward_cf(
        self,
        token_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
        role_ids: torch.Tensor,
        attribution_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        masked_embeds = token_embeds * attribution_mask.unsqueeze(-1).float()
        cross_embeds = self.cross_turn(
            masked_embeds, attention_mask, turn_mask, role_ids,
        )
        valid_mask = attention_mask * turn_mask.unsqueeze(-1)
        pooled = self.pool_with_mask(cross_embeds, valid_mask)
        cls_logits = self.cls_head(pooled, None)
        return {"cls_logits": cls_logits.squeeze(-1)}