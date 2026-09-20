"""Reusable components for the hierarchical GuardLens redesign."""

import math

import torch
import torch.nn as nn

from guardlens.config import GuardLensConfig


class TurnPositionEncoding(nn.Module):
    """Sinusoidal turn index plus learned user/assistant role embedding."""

    def __init__(self, d_model: int, max_turns: int):
        super().__init__()
        self.role_embedding = nn.Embedding(2, d_model)

        pe = torch.zeros(max_turns, d_model)
        position = torch.arange(0, max_turns).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, turn_idx: torch.Tensor, role_ids: torch.Tensor):
        return self.pe[turn_idx] + self.role_embedding(role_ids)


class TurnContextEncoder(nn.Module):
    """Contextualize pooled turns with O(T^2), not O((T*S)^2), attention."""

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.backbone_dim, config.cross_turn_dim)
        self.turn_pos = TurnPositionEncoding(
            config.cross_turn_dim, config.max_turns
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.cross_turn_dim,
            nhead=config.cross_turn_heads,
            dim_feedforward=config.cross_turn_dim * 4,
            dropout=config.cross_turn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.cross_turn_layers
        )
        self.layer_norm = nn.LayerNorm(config.cross_turn_dim)

    @staticmethod
    def masked_token_mean(
        token_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeds * mask).sum(dim=2)
        counts = mask.sum(dim=2).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        token_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.masked_token_mean(token_embeds, attention_mask)
        x = self.input_proj(pooled)
        batch_size, turns, _ = x.shape
        turn_idx = torch.arange(
            turns, device=x.device
        ).unsqueeze(0).expand(batch_size, -1)
        x = x + self.turn_pos(turn_idx, role_ids)

        padding_mask = turn_mask == 0
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        # Use masked_fill rather than multiplication so a pathological NaN in
        # a padded query position cannot survive as NaN * 0.
        return x.masked_fill(turn_mask.unsqueeze(-1) == 0, 0.0)


class ConversationPooler(nn.Module):
    """Learned attention pooling over contextualized realized turns."""

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(config.cross_turn_dim, config.cross_turn_dim // 2),
            nn.Tanh(),
            nn.Linear(config.cross_turn_dim // 2, 1),
        )

    def forward(self, turn_context: torch.Tensor, turn_mask: torch.Tensor):
        logits = self.score(turn_context).squeeze(-1)
        logits = logits.masked_fill(turn_mask == 0, -1e9)
        weights = torch.softmax(logits, dim=-1)
        pooled = (turn_context * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class ClassificationHead(nn.Module):
    """Conversation-level unsafe-trajectory classifier."""

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.cross_turn_dim, config.cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.cls_hidden_dim, config.cls_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.cls_hidden_dim // 2, 1),
        )

    def forward(self, pooled: torch.Tensor):
        return self.mlp(pooled).squeeze(-1)


class EvidenceTurnHead(nn.Module):
    """Independent P(intervention-supported evidence | user turn)."""

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.cross_turn_dim, config.attr_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.attr_hidden_dim, 1),
        )

    def forward(self, turn_context: torch.Tensor):
        return self.mlp(turn_context).squeeze(-1)


class ContextualSpanHead(nn.Module):
    """Token causal-evidence logits conditioned on conversation-aware turn state."""

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.token_proj = nn.Linear(config.backbone_dim, config.attr_hidden_dim)
        self.context_proj = nn.Linear(
            config.cross_turn_dim, config.attr_hidden_dim
        )
        self.norm = nn.LayerNorm(config.attr_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(config.attr_hidden_dim, 1)

    def forward(
        self,
        token_embeds: torch.Tensor,
        turn_context: torch.Tensor,
    ) -> torch.Tensor:
        token_h = self.token_proj(token_embeds)
        context_h = self.context_proj(turn_context).unsqueeze(2)
        h = self.norm(token_h + context_h)
        h = self.dropout(torch.nn.functional.gelu(h))
        return self.out(h).squeeze(-1)


# Backward import alias for code that only imported the class name.
AttributionHead = ContextualSpanHead
