"""Configuration for the NAACL causal-localization redesign."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GuardLensConfig:
    # Backbone
    backbone_name: str = "microsoft/deberta-v3-base"
    backbone_dim: int = 768
    freeze_backbone: bool = True

    # Hierarchical turn-context encoder
    cross_turn_layers: int = 2
    cross_turn_heads: int = 8
    cross_turn_dim: int = 256
    cross_turn_dropout: float = 0.1

    # Heads
    cls_hidden_dim: int = 256
    attr_hidden_dim: int = 128

    # Representation limits
    # max_tokens_per_turn is a hard, fail-closed ceiling. The collator uses
    # dynamic padding and never truncates a turn to fit this value.
    max_turns: int = 48
    max_tokens_per_turn: int = 512
    max_total_tokens: int = 2048  # legacy flat baseline only

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_epochs: int = 20
    batch_size: int = 2
    gradient_accumulation: int = 8
    max_grad_norm: float = 1.0

    # Two-stage training
    phase1_epochs: int = 5
    lambda_detection: float = 1.0
    lambda_turn: float = 1.0
    lambda_span: float = 1.0
    localization_ramp_start: float = 0.25

    # Counterfactual evidence weights. Unknown/legacy tiers fail closed at 0.
    span_tier_weights: dict = field(default_factory=lambda: {
        "cf_strong": 1.00,
        "cf_weak": 0.70,
        "incidental": 1.00,
        "ignore": 0.00,
    })
    turn_tier_weights: dict = field(default_factory=lambda: {
        "supported_strong": 1.00,
        "supported_weak": 0.70,
        "not_supported": 1.00,
    })

    # Detection class balance. <=0 means compute from weighted train mass.
    pos_weight: float = 0.0

    # Dev-only detection threshold tuning
    tune_threshold: bool = True
    default_threshold: float = 0.5

    # Misc
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    eval_every: int = 1
    patience: int = 8

    # Frozen input paths. Training intentionally has no test-path contract.
    train_path: str = ""
    dev_path: str = ""

    # Kept only for the legacy flat baseline until its evaluation migration.
    causal_span_labels: Tuple = (
        "MALICIOUS_TRIGGER", "PAYLOAD_SPAN", "CONTEXT_BRIDGE",
        "IMPLICIT_TRIGGER", "STRUCTURAL_TRIGGER",
    )
    incidental_span_labels: Tuple = (
        "SAFE_CONSTRAINT", "DECOY", "QUOTED_UNSAFE_CONTENT",
        "BENIGN_CONTEXT",
    )
