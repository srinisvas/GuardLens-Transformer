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

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if self.config.backbone_dtype not in dtype_map:
            raise RuntimeError(
                f"unsupported backbone_dtype={self.config.backbone_dtype!r}"
            )

        self.backbone = AutoModel.from_pretrained(
            self.config.backbone_name,
            revision=self.config.backbone_revision,
            output_hidden_states=False,
            torch_dtype=dtype_map[self.config.backbone_dtype],
            attn_implementation=self.config.backbone_attn_implementation,
        )
        max_positions = getattr(self.backbone.config, "max_position_embeddings", None)
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size != self.config.backbone_dim:
            raise RuntimeError(
                f"configured backbone_dim={self.config.backbone_dim} does not match "
                f"{self.config.backbone_name} hidden_size={hidden_size}"
            )
        if (
            isinstance(max_positions, int)
            and max_positions > 0
            and self.config.max_tokens_per_turn > max_positions
        ):
            raise RuntimeError(
                f"max_tokens_per_turn={self.config.max_tokens_per_turn} exceeds "
                f"backbone max_position_embeddings={max_positions}; select a "
                "backbone with native context coverage rather than truncating the turn"
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
        lengths = flat_mask.sum(dim=1)
        realized_idx = torch.nonzero(lengths > 0, as_tuple=False).squeeze(-1)
        if realized_idx.numel() == 0:
            raise RuntimeError("batch contains no realized turns")

        # With long-context turns, padding every realized turn to the single
        # longest turn in the conversation batch can multiply backbone compute.
        # Canonical training freezes ModernBERT, so we sort realized turns by
        # length and encode them in small length-local microbatches.
        if self.config.freeze_backbone:
            order = realized_idx[
                torch.argsort(lengths[realized_idx], stable=True)
            ]
            hidden = torch.zeros(
                batch_size * turns,
                seq_len,
                self.config.backbone_dim,
                device=input_ids.device,
                dtype=torch.float32,
            )
            microbatch = max(1, int(self.config.backbone_turn_microbatch))

            with torch.no_grad():
                for start in range(0, order.numel(), microbatch):
                    idx = order[start:start + microbatch]
                    local_len = int(lengths[idx].max().item())
                    outputs = self.backbone(
                        input_ids=flat_ids[idx, :local_len],
                        attention_mask=flat_mask[idx, :local_len],
                    )
                    chunk = outputs.last_hidden_state.float()
                    hidden[idx, :local_len] = chunk
        else:
            # Fine-tuning is deliberately kept simple until a dedicated
            # selective-unfreezing recipe is introduced.
            outputs = self.backbone(
                input_ids=flat_ids[realized_idx],
                attention_mask=flat_mask[realized_idx],
            )
            realized_hidden = outputs.last_hidden_state.float()
            hidden = realized_hidden.new_zeros(
                batch_size * turns,
                seq_len,
                realized_hidden.size(-1),
            )
            hidden[realized_idx] = realized_hidden

        return hidden.reshape(
            batch_size, turns, seq_len, -1
        )

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
            # Causal evidence turns are defined only over realized user turns.
            invalid_turn = (turn_mask == 0) | (role_ids != 0)
            turn_logits = turn_logits.masked_fill(invalid_turn, -1e9)
            turn_probs = torch.sigmoid(turn_logits)

            attr_logits = self.attr_head(token_embeds, turn_context)
            invalid_span = (
                (attention_mask == 0)
                | (role_ids.unsqueeze(-1) != 0)
            )
            attr_logits = attr_logits.masked_fill(invalid_span, -1e9)
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
