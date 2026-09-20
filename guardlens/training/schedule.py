"""Two-stage training schedule for the causal-localization redesign."""

from typing import Tuple

from guardlens.config import GuardLensConfig


def get_current_phase(epoch: int, config: GuardLensConfig) -> int:
    return 1 if epoch < config.phase1_epochs else 2


def get_lambda_schedule(
    epoch: int,
    config: GuardLensConfig,
) -> Tuple[float, float, float]:
    """Return detection, turn-localization and span-localization weights.

    Detection remains fully weighted throughout training. Localization starts
    only after the detection bootstrap and ramps to its configured weight.
    """
    if get_current_phase(epoch, config) == 1:
        return config.lambda_detection, 0.0, 0.0

    joint_epochs = max(1, config.max_epochs - config.phase1_epochs)
    joint_index = epoch - config.phase1_epochs
    progress = min(1.0, (joint_index + 1) / joint_epochs)
    start = float(config.localization_ramp_start)
    scale = start + (1.0 - start) * progress
    return (
        config.lambda_detection,
        config.lambda_turn * scale,
        config.lambda_span * scale,
    )
