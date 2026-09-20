"""GuardLens causal-localization package.

The NAACL redesign exposes the canonical GuardLens model and the two retained
legacy detection baselines. Fusion/NoCF ablations from the EMNLP architecture
are intentionally not part of the canonical model registry.
"""

from guardlens.config import GuardLensConfig
from guardlens.models.guardlens import GuardLens
from guardlens.models.baselines import TurnLevelClassifier, ConversationDeBERTa
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.training.loss import GuardLensLoss

__all__ = [
    "GuardLensConfig",
    "GuardLens",
    "TurnLevelClassifier",
    "ConversationDeBERTa",
    "GuardLensDataset",
    "GuardLensCollator",
    "GuardLensLoss",
]
