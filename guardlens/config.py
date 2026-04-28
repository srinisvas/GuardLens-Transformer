"""Model and training configuration."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GuardLensConfig:
    # Backbone
    backbone_name: str = "microsoft/deberta-v3-base"
    backbone_dim: int = 768
    freeze_backbone: bool = True

    # Cross-turn attention
    cross_turn_layers: int = 2
    cross_turn_heads: int = 8
    cross_turn_dim: int = 256
    cross_turn_dropout: float = 0.1

    # Heads
    cls_hidden_dim: int = 256
    attr_hidden_dim: int = 128
    n_classes: int = 2

    # Fusion
    use_gated_fusion: bool = True
    fusion_temperature: float = 1.0

    # Sequence limits
    max_turns: int = 16
    max_tokens_per_turn: int = 128
    max_total_tokens: int = 512

    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_epochs: int = 25
    batch_size: int = 8
    gradient_accumulation: int = 2
    max_grad_norm: float = 1.0

    # Loss weights (scheduled during training)
    lambda_attr: float = 0.5
    lambda_cf: float = 0.3

    # Training phases
    phase1_epochs: int = 5    # Classification only
    phase2_epochs: int = 15   # + Attribution
    phase3_epochs: int = 5    # + Counterfactual

    # Attribution labels
    causal_span_labels: Tuple = (
        "MALICIOUS_TRIGGER", "PAYLOAD_SPAN", "IMPLICIT_TRIGGER",
        "STRUCTURAL_TRIGGER",
    )
    non_causal_span_labels: Tuple = (
        "SAFE_CONSTRAINT", "FALSE_LEAD", "QUOTED_UNSAFE_CONTENT",
    )

    # Counterfactual
    cf_mask_token: str = "[MASK]"
    cf_delta_threshold: float = 0.3

    # Misc
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    eval_every: int = 1
    patience: int = 5
