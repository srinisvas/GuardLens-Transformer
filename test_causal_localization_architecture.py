#!/usr/bin/env python3
"""CPU architecture/loss regression tests for the causal-localization model."""
import unittest

import torch

from guardlens.config import GuardLensConfig
from guardlens.models.components import (
    ClassificationHead,
    ContextualSpanHead,
    ConversationPooler,
    EvidenceTurnHead,
    TurnContextEncoder,
)
from guardlens.models.guardlens import GuardLens
from guardlens.training.loss import GuardLensLoss


def tiny_config():
    return GuardLensConfig(
        backbone_dim=12,
        cross_turn_dim=8,
        cross_turn_heads=2,
        cross_turn_layers=1,
        attr_hidden_dim=6,
        cls_hidden_dim=8,
        max_turns=4,
    )


class CausalLocalizationArchitectureTests(unittest.TestCase):
    def test_hierarchical_components_have_expected_shapes(self):
        config = tiny_config()
        token = torch.randn(2, 3, 5, 12)
        attention = torch.ones(2, 3, 5, dtype=torch.long)
        turn_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        roles = torch.tensor([[0, 1, 0], [0, 1, 0]])

        context = TurnContextEncoder(config)(
            token, attention, turn_mask, roles
        )
        self.assertEqual(tuple(context.shape), (2, 3, 8))

        pooled, weights = ConversationPooler(config)(context, turn_mask)
        self.assertEqual(tuple(pooled.shape), (2, 8))
        self.assertEqual(tuple(weights.shape), (2, 3))

        turn_logits = EvidenceTurnHead(config)(context)
        span_logits = ContextualSpanHead(config)(token, context)
        cls_logits = ClassificationHead(config)(pooled)
        self.assertEqual(tuple(turn_logits.shape), (2, 3))
        self.assertEqual(tuple(span_logits.shape), (2, 3, 5))
        self.assertEqual(tuple(cls_logits.shape), (2,))

    def test_main_model_has_no_fusion_pivot_or_self_cf_path(self):
        model = GuardLens(tiny_config())
        self.assertFalse(hasattr(model, "fusion_gate"))
        self.assertFalse(hasattr(model, "pivot_head"))
        self.assertFalse(hasattr(model, "forward_cf"))

    def test_config_has_no_cf_oversampling_or_phase3_switches(self):
        config = tiny_config()
        self.assertFalse(hasattr(config, "oversample_cf"))
        self.assertFalse(hasattr(config, "cf_oversample_factor"))
        self.assertFalse(hasattr(config, "phase3_epochs"))
        self.assertFalse(hasattr(config, "lambda_cf"))

    def test_joint_loss_accepts_multiple_positive_turns(self):
        config = tiny_config()
        loss_fn = GuardLensLoss(config)
        loss_fn.set_pos_weight(1.0)
        loss_fn.set_turn_pos_weight(1.0)
        outputs = {
            "cls_logits": torch.tensor([0.3], requires_grad=True),
            "turn_logits": torch.tensor(
                [[1.0, -0.5, 0.8]], requires_grad=True
            ),
            "attr_logits": torch.tensor(
                [[[0.2, -0.2], [0.1, 0.3], [-0.1, 0.7]]],
                requires_grad=True,
            ),
        }
        losses = loss_fn(
            outputs,
            labels=torch.tensor([1]),
            token_labels=torch.tensor([[[1, 0], [-1, -1], [1, -1]]]),
            span_weights=torch.tensor(
                [[[1.0, 1.0], [0.0, 0.0], [0.7, 0.0]]]
            ),
            detection_weights=torch.tensor([1.0]),
            turn_labels=torch.tensor([[1, 0, 1]]),
            turn_weights=torch.tensor([[1.0, 1.0, 0.7]]),
            phase=2,
        )
        self.assertIn("turn", losses)
        self.assertIn("span", losses)
        self.assertTrue(torch.isfinite(losses["total"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
