#!/usr/bin/env python3
"""Regression tests for detection-only auxiliary loss isolation."""
import unittest

import torch

from guardlens.config import GuardLensConfig
from guardlens.training.loss import GuardLensLoss


class AuxiliaryLossIsolationTests(unittest.TestCase):
    @staticmethod
    def _detection_gradient(weight):
        loss_fn = GuardLensLoss(GuardLensConfig())
        logits = torch.tensor([0.0, 0.0], requires_grad=True)
        losses = loss_fn(
            {"cls_logits": logits, "turn_logits": None, "attr_logits": None},
            labels=torch.tensor([1, 1]),
            token_labels=torch.full((2, 1, 1), -1, dtype=torch.long),
            detection_weights=torch.tensor([weight, weight]),
            phase=1,
        )
        losses["total"].backward()
        return float(losses["detection"]), logits.grad.detach().clone()

    def test_ignored_localization_targets_produce_detection_only_loss(self):
        loss_fn = GuardLensLoss(GuardLensConfig())
        outputs = {
            "cls_logits": torch.tensor([1.0], requires_grad=True),
            "turn_logits": torch.tensor([[0.2, -0.3]], requires_grad=True),
            "attr_logits": torch.tensor(
                [[[0.1, 0.2], [0.3, 0.4]]], requires_grad=True
            ),
        }
        losses = loss_fn(
            outputs,
            labels=torch.tensor([1]),
            token_labels=torch.full((1, 2, 2), -1, dtype=torch.long),
            span_weights=torch.zeros(1, 2, 2),
            detection_weights=torch.tensor([0.25]),
            turn_labels=torch.full((1, 2), -1, dtype=torch.long),
            turn_weights=torch.zeros(1, 2),
            phase=2,
        )
        self.assertIn("detection", losses)
        self.assertNotIn("turn", losses)
        self.assertNotIn("span", losses)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_uniform_auxiliary_weight_does_not_cancel_within_microbatch(self):
        full_loss, full_grad = self._detection_gradient(1.0)
        aux_loss, aux_grad = self._detection_gradient(0.25)
        self.assertAlmostEqual(aux_loss / full_loss, 0.25, places=6)
        self.assertTrue(torch.allclose(aux_grad, full_grad * 0.25))

    def test_uniform_weak_turn_weight_remains_absolute(self):
        loss_fn = GuardLensLoss(GuardLensConfig())
        outputs = {
            "cls_logits": torch.tensor([0.0], requires_grad=True),
            "turn_logits": torch.tensor([[0.0, 0.0]], requires_grad=True),
            "attr_logits": None,
        }
        losses = loss_fn(
            outputs,
            labels=torch.tensor([1]),
            token_labels=torch.full((1, 1, 1), -1, dtype=torch.long),
            detection_weights=torch.tensor([1.0]),
            turn_labels=torch.tensor([[1, 1]]),
            turn_weights=torch.tensor([[0.70, 0.70]]),
            phase=2,
        )
        unit_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(0.0), torch.tensor(1.0)
        )
        self.assertAlmostEqual(
            float(losses["turn"]), float(unit_bce * 0.70), places=6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
