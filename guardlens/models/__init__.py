from guardlens.models.guardlens import GuardLens
from guardlens.models.components import (
    TurnPositionEncoding,
    CrossTurnAttention,
    ClassificationHead,
    AttributionHead,
)
from guardlens.models.baselines import (
    TurnLevelClassifier,
    ConversationDeBERTa,
    GuardLensNoFusion,
    GuardLensNoCF,
)

MODEL_REGISTRY = {
    "guardlens": GuardLens,
    "guardlens_no_fusion": GuardLensNoFusion,
    "guardlens_no_cf": GuardLensNoCF,
    "turn_level": TurnLevelClassifier,
    "conversation_deberta": ConversationDeBERTa,
}