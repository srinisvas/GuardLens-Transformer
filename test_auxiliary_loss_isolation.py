#!/usr/bin/env python3
"""Regression tests for detection-only auxiliary loss isolation."""
from __future__ import annotations

import unittest

import torch

from guardlens.config import GuardLensConfig
from guardlens.training.loss import GuardLensLoss


class _MustNotRunCounterfactualModel:
    def forward_cf(self, **kwargs):
        raise AssertionError("counterfactual forward must not run for auxiliary-only positives")


class AuxiliaryLossIsolationTests(unittest.TestCase):
    def test_auxiliary_positive_is_excluded_from_counterfactual_loss(self):
        loss_fn = GuardLensLoss(GuardLensConfig())
        outputs = {
            "attr_probs": torch.full((1, 1, 2), 0.5),
            "cls_logits": torch.tensor([1.0]),
            "token_embeds": torch.zeros(1, 1, 2, 4),
        }
        batch = {
            "attention_mask": torch.ones(1, 1, 2, dtype=torch.long),
            "turn_mask": torch.ones(1, 1, dtype=torch.long),
            "role_ids": torch.zeros(1, 1, dtype=torch.long),
            "labels": torch.tensor([1], dtype=torch.long),
            "cf_loss_eligible": torch.tensor([False], dtype=torch.bool),
        }
        value = loss_fn.counterfactual_loss(
            _MustNotRunCounterfactualModel(), batch, outputs
        )
        self.assertEqual(float(value.detach()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
