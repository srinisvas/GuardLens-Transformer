"""Model and training configuration — NAACL validity-repair compatible."""

from dataclasses import dataclass, field
from typing import Tuple, List


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

    # Pivot head
    use_pivot_head: bool = True
    n_pivot_kinds: int = 5  # lexical_pivot, contextual_pivot, distributed, misleading_decoy, none

    # Fusion
    use_gated_fusion: bool = True
    fusion_temperature: float = 1.0

    # Sequence limits. The repaired primary legacy corpus contains up to 48
    # realized user+assistant turns, so the NAACL default preserves every
    # primary training/evaluation turn rather than silently truncating at 32.
    max_turns: int = 48
    max_tokens_per_turn: int = 192
    max_total_tokens: int = 2048  # For flat baseline

    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_epochs: int = 25
    batch_size: int = 2  # 48-turn hierarchical window; effective batch kept at 16
    gradient_accumulation: int = 8
    max_grad_norm: float = 1.0

    # Loss weights (scheduled during training)
    lambda_cls: float = 0.2
    lambda_attr: float = 1.0
    lambda_cf: float = 0.5
    lambda_pivot: float = 0.3

    # Training phases
    phase1_epochs: int = 5
    phase2_epochs: int = 15
    phase3_epochs: int = 5

    # Attribution labels — v11 updated
    # Primary signal: span["causal_type"] == "causal"
    # Fallback: span["label"] in causal_span_labels
    causal_span_labels: Tuple = (
        "MALICIOUS_TRIGGER", "PAYLOAD_SPAN", "CONTEXT_BRIDGE",
        "IMPLICIT_TRIGGER", "STRUCTURAL_TRIGGER",
    )
    incidental_span_labels: Tuple = (
        "SAFE_CONSTRAINT", "DECOY", "QUOTED_UNSAFE_CONTENT",
        "BENIGN_CONTEXT",
    )

    # Supervision tier weights for attribution loss
    span_tier_weights: dict = field(default_factory=lambda: {
        "cf_strong": 1.00,
        "cf_weak": 0.70,
        "llm_confirmed": 0.60,
        "construction": 0.40,
        "llm_only": 0.25,
        "incidental": 1.00,
        "ignore": 0.00,
    })

    # Class balance
    pos_weight: float = 0.0  # 0 = auto-compute from data

    # Counterfactual
    cf_delta_threshold: float = 0.3

    # CF/tier oversampling
    oversample_cf: bool = True
    cf_oversample_factor: int = 3

    # Dev threshold tuning
    tune_threshold: bool = True
    default_threshold: float = 0.5

    # Misc
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    eval_every: int = 1
    patience: int = 8

    # Data paths
    train_path: str = ""
    dev_path: str = ""
    test_path: str = ""
