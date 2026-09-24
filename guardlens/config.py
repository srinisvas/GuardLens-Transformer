"""Configuration for the NAACL causal-localization redesign."""

from dataclasses import dataclass

@dataclass
class GuardLensConfig:
    # Backbone
    backbone_name: str = "answerdotai/ModernBERT-large"
    backbone_revision: str = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
    backbone_dim: int = 1024
    freeze_backbone: bool = True
    backbone_dtype: str = "bfloat16"
    backbone_attn_implementation: str = "sdpa"
    backbone_turn_microbatch: int = 8
    # Zero keeps the backbone frozen. A positive value selectively trains the
    # final N transformer layers while leaving embeddings and earlier layers
    # frozen. This is an explicit diagnostic axis, not an automatic fallback.
    backbone_trainable_layers: int = 0

    # Hierarchical turn-context encoder
    cross_turn_layers: int = 2
    cross_turn_heads: int = 8
    cross_turn_dim: int = 256
    cross_turn_dropout: float = 0.1
    turn_pooling: str = "attention"

    # Heads
    cls_hidden_dim: int = 256
    attr_hidden_dim: int = 128

    # Representation limits
    # max_tokens_per_turn is a hard, fail-closed ceiling. The collator uses
    # dynamic padding and never truncates a turn to fit this value.
    max_turns: int = 64
    max_tokens_per_turn: int = 8192

    # Optimization
    learning_rate: float = 2e-4
    backbone_learning_rate: float = 2e-5
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
    # Reach full localization weight early enough that OneCycleLR is still
    # materially high, then keep full weight for the remainder of Phase 2.
    localization_ramp_epochs: int = 5

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
    patience: int = 0  # canonical schedule runs all epochs; >0 enables optional early stop

    # Frozen input paths. Training intentionally has no test-path contract.
    train_path: str = ""
    dev_path: str = ""
    train_variant: str = "primary_plus_auxiliary"
    input_view: str = "pre_response"
