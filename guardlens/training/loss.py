"""Losses for joint detection, evidence-turn and span localization."""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from guardlens.config import GuardLensConfig


class GuardLensLoss(nn.Module):
    def __init__(self, config: GuardLensConfig):
        super().__init__()
        self.config = config
        self.pos_weight: Optional[torch.Tensor] = None
        self.turn_pos_weight: Optional[torch.Tensor] = None

    def set_pos_weight(self, pos_weight: float):
        self.pos_weight = torch.tensor([float(pos_weight)])

    def set_turn_pos_weight(self, pos_weight: float):
        self.turn_pos_weight = torch.tensor([float(pos_weight)])

    @staticmethod
    def _weighted_mean(raw: torch.Tensor, weights: Optional[torch.Tensor]):
        """Apply absolute confidence weights without batch-local renormalization.

        Dividing by ``weights.sum()`` would cancel a uniform 0.25 auxiliary
        weight or 0.70 weak-evidence weight whenever a microbatch contains only
        that tier. Dividing by the number of eligible targets preserves the
        intended absolute influence while keeping the ordinary mean scale for
        unit-weight targets.
        """
        if weights is None:
            return raw.mean()
        if raw.numel() == 0:
            raise RuntimeError("cannot reduce an empty weighted loss")
        return (raw * weights).sum() / raw.numel()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        token_labels: torch.Tensor,
        *,
        span_weights: torch.Tensor = None,
        detection_weights: torch.Tensor = None,
        turn_labels: torch.Tensor = None,
        turn_weights: torch.Tensor = None,
        phase: int = 1,
        lambda_detection: float = 1.0,
        lambda_turn: float = 1.0,
        lambda_span: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        device = outputs["cls_logits"].device
        losses: Dict[str, torch.Tensor] = {}

        det_pw = self.pos_weight.to(device) if self.pos_weight is not None else None
        det_raw = F.binary_cross_entropy_with_logits(
            outputs["cls_logits"],
            labels.float(),
            pos_weight=det_pw,
            reduction="none",
        )
        det_w = detection_weights.to(device) if detection_weights is not None else None
        l_detection = self._weighted_mean(det_raw, det_w)
        losses["detection"] = l_detection
        total = lambda_detection * l_detection

        if phase >= 2 and outputs.get("turn_logits") is not None and turn_labels is not None:
            tl = turn_labels.to(device)
            valid_turn = tl >= 0
            if turn_weights is not None:
                tw = turn_weights.to(device)
                valid_turn = valid_turn & (tw > 0)
            else:
                tw = None
            if valid_turn.any():
                turn_pw = (
                    self.turn_pos_weight.to(device)
                    if self.turn_pos_weight is not None
                    else None
                )
                turn_raw = F.binary_cross_entropy_with_logits(
                    outputs["turn_logits"][valid_turn],
                    tl[valid_turn].float(),
                    pos_weight=turn_pw,
                    reduction="none",
                )
                l_turn = self._weighted_mean(
                    turn_raw,
                    tw[valid_turn] if tw is not None else None,
                )
                losses["turn"] = l_turn
                total = total + lambda_turn * l_turn

        if phase >= 2 and outputs.get("attr_logits") is not None:
            valid_span = token_labels >= 0
            if span_weights is not None:
                sw = span_weights.to(device)
                valid_span = valid_span & (sw > 0)
            else:
                sw = None
            if valid_span.any():
                span_raw = F.binary_cross_entropy_with_logits(
                    outputs["attr_logits"][valid_span],
                    token_labels[valid_span].float(),
                    reduction="none",
                )
                l_span = self._weighted_mean(
                    span_raw,
                    sw[valid_span] if sw is not None else None,
                )
                losses["span"] = l_span
                total = total + lambda_span * l_span

        losses["total"] = total
        return losses
