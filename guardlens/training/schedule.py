"""Training phase and loss weight scheduling."""

from typing import Tuple

from guardlens.config import GuardLensConfig


def get_current_phase(epoch: int, config: GuardLensConfig) -> int:
    """Determine training phase from epoch number."""
    if epoch < config.phase1_epochs:
        return 1
    elif epoch < config.phase1_epochs + config.phase2_epochs:
        return 2
    else:
        return 3


def get_lambda_schedule(
    epoch: int, config: GuardLensConfig,
) -> Tuple[float, float]:
    """
    Gradual lambda increase within each phase.

    Phase 1: (0, 0)
    Phase 2: lambda_attr ramps from 0.1 to config.lambda_attr
    Phase 3: lambda_attr at max, lambda_cf ramps from 0.1 to config.lambda_cf
    """
    phase = get_current_phase(epoch, config)

    if phase == 1:
        return 0.0, 0.0
    elif phase == 2:
        progress = (epoch - config.phase1_epochs) / max(1, config.phase2_epochs)
        la = 0.1 + progress * (config.lambda_attr - 0.1)
        return la, 0.0
    else:
        la = config.lambda_attr
        progress = (
            (epoch - config.phase1_epochs - config.phase2_epochs)
            / max(1, config.phase3_epochs)
        )
        lc = 0.1 + progress * (config.lambda_cf - 0.1)
        return la, lc
