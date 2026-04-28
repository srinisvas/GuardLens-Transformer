"""
GuardLens: Multi-turn adversarial prompt detection with causal token attribution.

Usage:
    python -m guardlens.train --data data.jsonl --output ./checkpoints
    python -m guardlens.train --data data.jsonl --output ./checkpoints --model turn_level
    python -m guardlens.evaluate --data data.jsonl --checkpoint ./checkpoints/best.pt
"""

from guardlens.config import GuardLensConfig
from guardlens.models.guardlens import GuardLens
from guardlens.models.baselines import (
    TurnLevelClassifier,
    ConversationDeBERTa,
    GuardLensNoFusion,
    GuardLensNoCF,
)
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.training.loss import GuardLensLoss

__all__ = [
    "GuardLensConfig",
    "GuardLens",
    "TurnLevelClassifier",
    "ConversationDeBERTa",
    "GuardLensNoFusion",
    "GuardLensNoCF",
    "GuardLensDataset",
    "GuardLensCollator",
    "GuardLensLoss",
]