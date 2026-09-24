"""GuardLens dual-architecture causal-localization model."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from guardlens.config import GuardLensConfig
from guardlens.models.components import (
    ClassificationHead,
    ContextualSpanHead,
    CrossTokenContextEncoder,
    ConversationPooler,
    DirectSpanHead,
    EvidenceTurnHead,
    TurnContextEncoder,
)


class GuardLens(nn.Module):
    """Configurable multi-turn detector and causal-localization model.

    ``hierarchical_turn`` uses sibling heads over contextualized turn states.
    ``cross_token`` restores direct token-to-token attention across turns and
    optionally restores attribution-aware gated fusion into detection.
    """

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config
        self.backbone = None
        self.backbone_loaded = False
        self.backbone_fully_frozen = True

        self.architecture_mode = getattr(
            config, "architecture_mode", "hierarchical_turn"
        )
        if self.architecture_mode not in {
            "hierarchical_turn", "cross_token"
        }:
            raise ValueError(
                f"unsupported architecture_mode={self.architecture_mode!r}"
            )
        self.use_attribution_fusion = bool(
            getattr(config, "use_attribution_fusion", False)
        )
        if self.use_attribution_fusion and self.architecture_mode != "cross_token":
            raise ValueError(
                "attribution-aware fusion requires architecture_mode=cross_token"
            )

        self.turn_context = None
        self.cross_token_context = None
        self.pooler = None
        if self.architecture_mode == "hierarchical_turn":
            self.turn_context = TurnContextEncoder(config)
            self.pooler = ConversationPooler(config)
            self.attr_head = ContextualSpanHead(config)
        else:
            self.cross_token_context = CrossTokenContextEncoder(config)
            self.attr_head = DirectSpanHead(config)
        self.cls_head = ClassificationHead(config)
        self.turn_head = EvidenceTurnHead(config)
        self.fusion_gate = None
        if self.use_attribution_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(config.cross_turn_dim, config.cross_turn_dim),
                nn.Sigmoid(),
            )

    def setup_backbone(self):
        if self.backbone_loaded:
            return
        import transformers
        from packaging.version import Version
        from transformers import AutoModel

        if not Version("4.56.2") <= Version(transformers.__version__) < Version("5"):
            raise RuntimeError(
                f"transformers {transformers.__version__} cannot load the "
                "pinned GuardLens dtype contract; install transformers>=4.56.2,<5"
            )

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
            dtype=dtype_map[self.config.backbone_dtype],
            attn_implementation=self.config.backbone_attn_implementation,
        )
        expected_dtype = dtype_map[self.config.backbone_dtype]
        mismatched = [(name, str(param.dtype)) for name, param in self.backbone.named_parameters()
                      if param.is_floating_point() and param.dtype != expected_dtype]
        if mismatched:
            raise RuntimeError(
                f"backbone requested {expected_dtype} but loaded parameters with different dtype: "
                f"{mismatched[:3]}"
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
        trainable_layers = int(self.config.backbone_trainable_layers)
        if trainable_layers < 0:
            raise RuntimeError("backbone_trainable_layers must be nonnegative")
        if trainable_layers > 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
            layers = getattr(self.backbone, "layers", None)
            if layers is None:
                encoder = getattr(self.backbone, "encoder", None)
                layers = getattr(encoder, "layer", None)
            if layers is None or not isinstance(layers, (nn.ModuleList, list, tuple)):
                raise RuntimeError(
                    "cannot locate backbone transformer layers for selective fine-tuning"
                )
            if trainable_layers > len(layers):
                raise RuntimeError(
                    f"requested {trainable_layers} trainable backbone layers, "
                    f"but backbone exposes {len(layers)}"
                )
            for layer in layers[-trainable_layers:]:
                # Keep trainable weights and Adam moments in FP32. CUDA
                # autocast still executes supported A100 kernels in BF16 while
                # the frozen prefix remains stored in BF16.
                layer.float()
                for param in layer.parameters():
                    param.requires_grad = True
            final_norm = getattr(self.backbone, "final_norm", None)
            if final_norm is not None:
                final_norm.float()
                for param in final_norm.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                # Non-reentrant checkpointing still computes parameter gradients
                # when inputs from the frozen prefix do not require gradients.
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            self.backbone_fully_frozen = False
        elif self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            self.backbone_fully_frozen = True
        else:
            self.backbone_fully_frozen = False
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
        # Sort realized turns by length and encode them in small length-local
        # microbatches for both frozen and selectively trainable variants.
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
        trainable_backward = (
            not self.backbone_fully_frozen
            and self.training
            and torch.is_grad_enabled()
        )
        microbatch = max(1, int(self.config.backbone_turn_microbatch))
        if not trainable_backward and not self.backbone_fully_frozen:
            # Eight-turn inference microbatches are already covered by the
            # frozen-backbone memory smoke and avoid making top-layer variants
            # needlessly slow during dev diagnostics.
            microbatch = max(8, microbatch)

        # Length-local microbatches are required for both frozen and selectively
        # trainable backbones. The latter keeps autograd enabled, while gradient
        # checkpointing limits activation memory inside the trainable layers.
        grad_context = torch.enable_grad if trainable_backward else torch.no_grad
        with grad_context():
            for start in range(0, order.numel(), microbatch):
                idx = order[start:start + microbatch]
                local_len = int(lengths[idx].max().item())
                outputs = self.backbone(
                    input_ids=flat_ids[idx, :local_len],
                    attention_mask=flat_mask[idx, :local_len],
                )
                chunk = outputs.last_hidden_state.float()
                hidden[idx, :local_len] = chunk

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
        localization_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Temporary keyword compatibility while the evaluation suite is migrated.
        if compute_attribution is not None:
            compute_localization = bool(compute_attribution)

        token_embeds = self.encode_turns(input_ids, attention_mask)
        effective_attention = attention_mask
        if localization_mask is None:
            # Compatibility for historical callers. Canonical V4 training and
            # runtime always supply the offset-derived text-token mask.
            localization_mask = attention_mask
        if localization_mask.shape != attention_mask.shape:
            raise RuntimeError("localization_mask shape must match attention_mask")
        localization_mask = (
            localization_mask.to(attention_mask.device).ne(0).long()
            * attention_mask.ne(0).long()
        )

        if attribution_mask is not None:
            mask = attribution_mask.to(token_embeds.device).float()
            token_embeds = token_embeds * mask.unsqueeze(-1)
            effective_attention = attention_mask * (mask > 0).long()

        valid_token_mask = (
            effective_attention * turn_mask.unsqueeze(-1)
        )
        valid_localization_mask = (
            localization_mask
            * effective_attention
            * turn_mask.unsqueeze(-1)
            * (role_ids == 0).long().unsqueeze(-1)
        )
        token_context = None
        if self.architecture_mode == "hierarchical_turn":
            turn_context = self.turn_context(
                token_embeds,
                effective_attention,
                turn_mask,
                role_ids,
            )
            pooled, pool_weights = self.pooler(turn_context, turn_mask)
        else:
            token_context = self.cross_token_context(
                token_embeds,
                effective_attention,
                turn_mask,
                role_ids,
            )
            turn_context = self._pool_per_turn(
                token_context, valid_token_mask
            )
            pooled = self._pool_all_tokens(
                token_context, valid_token_mask
            )
            pool_weights = None

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

        # In the gated candidate, token attribution is part of the detector
        # itself and therefore must be evaluated even when callers request
        # detection-only output (including Phase 1 and threshold tuning).
        if compute_localization or self.use_attribution_fusion:
            if self.architecture_mode == "hierarchical_turn":
                attr_logits = self.attr_head(token_embeds, turn_context)
            else:
                attr_logits = self.attr_head(token_context)
            invalid_span = (
                (valid_localization_mask == 0)
                | (role_ids.unsqueeze(-1) != 0)
            )
            attr_logits = attr_logits.masked_fill(invalid_span, -1e9)
            attr_probs = torch.sigmoid(attr_logits)

        attributed = None
        if self.use_attribution_fusion and attr_probs is not None:
            gate = self.fusion_gate(token_context)
            attributed = self._pool_all_tokens(
                token_context * attr_probs.unsqueeze(-1) * gate,
                valid_localization_mask,
            )
        cls_logits = self.cls_head(pooled, attributed)

        return {
            "cls_logits": cls_logits,
            "turn_logits": turn_logits,
            "turn_probs": turn_probs,
            "attr_logits": attr_logits,
            "attr_probs": attr_probs,
            "pooled": pooled,
            "pool_weights": pool_weights,
            "turn_context": turn_context,
            "token_context": token_context,
            "attributed_pooled": attributed,
        }

    @staticmethod
    def _pool_all_tokens(
        token_context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        summed = (token_context * mask).sum(dim=(1, 2))
        count = mask.sum(dim=(1, 2)).clamp(min=1.0)
        return summed / count

    @staticmethod
    def _pool_per_turn(
        token_context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        summed = (token_context * mask).sum(dim=2)
        count = mask.sum(dim=2).clamp(min=1.0)
        return summed / count
