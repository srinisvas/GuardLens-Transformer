"""Reusable model components for GuardLens and baselines."""

import math

import torch
import torch.nn as nn

from guardlens.config import GuardLensConfig


class TurnPositionEncoding(nn.Module):
    """
    Turn-level position information:
      - Sinusoidal turn index
      - Learned role embedding (user=0, assistant=1)
    """

    def __init__(self, d_model: int, max_turns: int = 16):
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
        """
        Args:
            turn_idx: [B, T]
            role_ids: [B, T]
        Returns:
            [B, T, D]
        """
        return self.pe[turn_idx] + self.role_embedding(role_ids)


class CrossTurnAttention(nn.Module):
    """
    Flattened token-level cross-turn attention with turn/role
    position embeddings.

    Flattens all tokens from all turns into a single sequence
    [B, T*S, D] and runs transformer self-attention. Turn structure
    is encoded via additive position embeddings (sinusoidal turn
    index + learned role).

    This allows individual tokens in turn 5 to attend directly
    to individual tokens in turn 2.
    """

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.backbone_dim, config.cross_turn_dim)
        self.turn_pos = TurnPositionEncoding(
            config.cross_turn_dim, config.max_turns,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.cross_turn_dim,
            nhead=config.cross_turn_heads,
            dim_feedforward=config.cross_turn_dim * 4,
            dropout=config.cross_turn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.cross_turn_layers,
        )
        self.layer_norm = nn.LayerNorm(config.cross_turn_dim)

    def forward(
        self,
        token_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_embeds: [B, T, S, D_backbone]
            attention_mask: [B, T, S]
            turn_mask: [B, T]
            role_ids: [B, T]
        Returns:
            [B, T, S, D_cross]
        """
        B, T, S, D = token_embeds.shape

        x = self.input_proj(token_embeds)
        turn_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        turn_pos = self.turn_pos(turn_idx, role_ids)
        x = x + turn_pos.unsqueeze(2)

        x_flat = x.reshape(B, T * S, -1)
        flat_mask = (attention_mask * turn_mask.unsqueeze(-1)).reshape(B, T * S)
        padding_mask = flat_mask == 0

        x_flat = self.transformer(x_flat, src_key_padding_mask=padding_mask)
        x_flat = self.layer_norm(x_flat)

        return x_flat.reshape(B, T, S, -1)


class ClassificationHead(nn.Module):
    """
    Pools representations and produces P(adversarial).
    Optionally receives gated input from the attribution head.
    """

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.expects_fusion = config.use_gated_fusion
        input_dim = config.cross_turn_dim
        if config.use_gated_fusion:
            input_dim = config.cross_turn_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.cls_hidden_dim, config.cls_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.cls_hidden_dim // 2, 1),
        )

    def forward(self, pooled: torch.Tensor, gated: torch.Tensor = None):
        if gated is not None:
            x = torch.cat([pooled, gated], dim=-1)
        elif self.expects_fusion:
            # Phase 1: attribution not computed yet, but MLP expects
            # concatenated [pooled, gated]. Pad with zeros.
            zeros = torch.zeros_like(pooled)
            x = torch.cat([pooled, zeros], dim=-1)
        else:
            x = pooled
        return self.mlp(x)


class AttributionHead(nn.Module):
    """Per-token P(causal | token_i)."""

    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.cross_turn_dim, config.attr_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.attr_hidden_dim, 1),
        )

    def forward(self, token_embeds: torch.Tensor) -> torch.Tensor:
        """[B, T, S, D] -> [B, T, S]"""
        return self.mlp(token_embeds).squeeze(-1)