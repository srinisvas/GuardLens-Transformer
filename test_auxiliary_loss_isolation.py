#!/usr/bin/env python3
"""Regression tests for detection-only auxiliary loss isolation."""
import unittest

import torch

from guardlens.config import GuardLensConfig
from guardlens.training.loss import GuardLensLoss


class AuxiliaryLossIsolationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
