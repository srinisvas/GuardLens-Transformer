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
) -> Tuple[float, float, float]:
    """
    Returns (lambda_cls, lambda_attr, lambda_cf) for the current epoch.

    Phase 1: cls=1.0, attr=0, cf=0  (bootstrap classification)
    Phase 2: cls ramps down to config.lambda_cls,
             attr ramps up to config.lambda_attr
    Phase 3: cls at config.lambda_cls,
             attr at config.lambda_attr,
             cf ramps up to config.lambda_cf
    """
    phase = get_current_phase(epoch, config)

    if phase == 1:
        return 1.0, 0.0, 0.0
    elif phase == 2:
        progress = (epoch - config.phase1_epochs) / max(1, config.phase2_epochs)
        # Classification ramps DOWN from 1.0 to config.lambda_cls
        lc = 1.0 - progress * (1.0 - config.lambda_cls)
        # Attribution ramps UP from 0.1 to config.lambda_attr
        la = 0.1 + progress * (config.lambda_attr - 0.1)
        return lc, la, 0.0
    else:
        lc = config.lambda_cls
        la = config.lambda_attr
        progress = (
            (epoch - config.phase1_epochs - config.phase2_epochs)
            / max(1, config.phase3_epochs)
        )
        lcf = 0.1 + progress * (config.lambda_cf - 0.1)
        return lc, la, lcf