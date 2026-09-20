"""GuardLens hierarchical causal-localization model."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from guardlens.config import GuardLensConfig
from guardlens.models.components import (
    ClassificationHead,
    ContextualSpanHead,
    ConversationPooler,
    EvidenceTurnHead,
    TurnContextEncoder,
)


class GuardLens(nn.Module):
    """Joint trajectory detection, multi-turn evidence localization and spans.

    Detection and localization are sibling heads over shared contextual
    representations. Attribution never gates or otherwise feeds the detector.
    """

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config
        self.backbone = None
        self.backbone_loaded = False

        self.turn_context = TurnContextEncoder(config)
        self.pooler = ConversationPooler(config)
        self.cls_head = ClassificationHead(config)
        self.turn_head = EvidenceTurnHead(config)
        self.attr_head = ContextualSpanHead(config)

    def setup_backbone(self):
        if self.backbone_loaded:
            return
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(
            self.config.backbone_name,
            output_hidden_states=False,
        )
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.backbone_loaded = True

    def encode_turns(self, input_ids, attention_mask):
        if self.backbone is None:
            raise RuntimeError("setup_backbone() must be called before forward()")
        batch_size, turns, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * turns, seq_len)
        flat_mask = attention_mask.reshape(batch_size * turns, seq_len)

        ctx = (
            torch.no_grad()
            if self.config.freeze_backbone
            else torch.enable_grad()
        )
        with ctx:
            outputs = self.backbone(
                input_ids=flat_ids,
                attention_mask=flat_mask,
            )
            hidden = outputs.last_hidden_state

        return hidden.reshape(
            batch_size, turns, seq_len, -1
        ).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
        role_ids: torch.Tensor,
        compute_localization: bool = True,
        attribution_mask: Optional[torch.Tensor] = None,
        compute_attribution: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        # Temporary keyword compatibility while the evaluation suite is migrated.
        if compute_attribution is not None:
            compute_localization = bool(compute_attribution)

        token_embeds = self.encode_turns(input_ids, attention_mask)
        effective_attention = attention_mask

        if attribution_mask is not None:
            mask = attribution_mask.to(token_embeds.device).float()
            token_embeds = token_embeds * mask.unsqueeze(-1)
            effective_attention = attention_mask * (mask > 0).long()

        turn_context = self.turn_context(
            token_embeds,
            effective_attention,
            turn_mask,
            role_ids,
        )
        pooled, pool_weights = self.pooler(turn_context, turn_mask)
        cls_logits = self.cls_head(pooled)

        attr_logits = None
        attr_probs = None
        turn_logits = None
        turn_probs = None

        if compute_localization:
            turn_logits = self.turn_head(turn_context)
            turn_logits = turn_logits.masked_fill(turn_mask == 0, -1e9)
            turn_probs = torch.sigmoid(turn_logits)

            attr_logits = self.attr_head(token_embeds, turn_context)
            attr_probs = torch.sigmoid(attr_logits)

        return {
            "cls_logits": cls_logits,
            "turn_logits": turn_logits,
            "turn_probs": turn_probs,
            "attr_logits": attr_logits,
            "attr_probs": attr_probs,
            "pooled": pooled,
            "pool_weights": pool_weights,
            "turn_context": turn_context,
        }
