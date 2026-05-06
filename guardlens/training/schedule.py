from typing import Tuple

from guardlens.config import GuardLensConfig


def get_current_phase(epoch: int, config: GuardLensConfig) -> int:
    if epoch < config.phase1_epochs:
        return 1
    elif epoch < config.phase1_epochs + config.phase2_epochs:
        return 2
    else:
        return 3


def get_lambda_schedule(
    epoch: int, config: GuardLensConfig,
) -> Tuple[float, float, float]:

    phase = get_current_phase(epoch, config)

    if phase == 1:
        return 1.0, 0.0, 0.0
    elif phase == 2:
        progress = (epoch - config.phase1_epochs) / max(1, config.phase2_epochs)
        lc = 1.0 - progress * (1.0 - config.lambda_cls)
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