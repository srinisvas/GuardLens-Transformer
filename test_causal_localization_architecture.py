#!/usr/bin/env python3
"""CPU architecture/loss regression tests for the causal-localization model."""
import unittest

import torch
import torch.nn as nn
from types import SimpleNamespace

from guardlens.config import GuardLensConfig
from guardlens.models.components import (
    ClassificationHead,
    ContextualSpanHead,
    CrossTokenContextEncoder,
    ConversationPooler,
    DirectSpanHead,
    EvidenceTurnHead,
    TurnContextEncoder,
)
from guardlens.models.guardlens import GuardLens
from guardlens.training.loss import GuardLensLoss
from guardlens.training.schedule import get_lambda_schedule


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
    def test_attention_pooling_ignores_padding_and_is_not_forced_to_mean(self):
        config = tiny_config()
        encoder = TurnContextEncoder(config)
        with torch.no_grad():
            encoder.token_score[0].weight.zero_()
            encoder.token_score[0].bias.zero_()
            encoder.token_score[2].weight.zero_()
            encoder.token_score[2].bias.zero_()
        token = torch.tensor([[[[1.0] * 12, [3.0] * 12, [99.0] * 12]]])
        mask = torch.tensor([[[1, 1, 0]]])
        pooled = encoder.pool_tokens(token, mask)
        self.assertTrue(torch.allclose(pooled, torch.full((1, 1, 12), 2.0)))

        with torch.no_grad():
            encoder.token_score[0].weight.fill_(0.1)
            encoder.token_score[2].weight.fill_(0.1)
        weighted = encoder.pool_tokens(token, mask)
        self.assertTrue(torch.isfinite(weighted).all())
        self.assertFalse(torch.allclose(weighted, pooled))

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

    def test_main_model_keeps_hierarchical_sibling_path(self):
        model = GuardLens(tiny_config())
        self.assertEqual(model.architecture_mode, "hierarchical_turn")
        self.assertIsNotNone(model.turn_context)
        self.assertIsNone(model.cross_token_context)
        self.assertIsNone(model.fusion_gate)
        self.assertFalse(hasattr(model, "pivot_head"))
        self.assertFalse(hasattr(model, "forward_cf"))

    def test_cross_token_encoder_compacts_all_realized_tokens_together(self):
        torch.manual_seed(7)
        config = tiny_config()
        config.architecture_mode = "cross_token"
        encoder = CrossTokenContextEncoder(config).eval()
        token = torch.randn(1, 3, 4, config.backbone_dim)
        attention = torch.tensor([[[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]]])
        turn_mask = torch.ones(1, 3, dtype=torch.long)
        roles = torch.tensor([[0, 1, 0]])
        seen = []
        hook = encoder.transformer.register_forward_pre_hook(
            lambda _module, args: seen.append(tuple(args[0].shape))
        )
        before = encoder(token, attention, turn_mask, roles)
        changed = token.clone()
        changed[0, 0, 0] += 5.0
        after = encoder(changed, attention, turn_mask, roles)
        hook.remove()
        self.assertEqual(seen, [(1, 6, 8), (1, 6, 8)])
        self.assertFalse(torch.allclose(before[0, 2, 0], after[0, 2, 0]))
        self.assertTrue(torch.all(before[attention == 0] == 0))

    def test_cross_token_heads_have_expected_shapes(self):
        config = tiny_config()
        config.architecture_mode = "cross_token"
        token = torch.randn(2, 3, 5, config.backbone_dim)
        attention = torch.ones(2, 3, 5, dtype=torch.long)
        turn_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        roles = torch.tensor([[0, 1, 0], [0, 1, 0]])
        context = CrossTokenContextEncoder(config)(
            token, attention, turn_mask, roles
        )
        self.assertEqual(tuple(context.shape), (2, 3, 5, 8))
        self.assertEqual(
            tuple(DirectSpanHead(config)(context).shape), (2, 3, 5)
        )

    def test_gated_cross_token_detection_backpropagates_through_attribution(self):
        config = tiny_config()
        config.architecture_mode = "cross_token"
        config.use_attribution_fusion = True
        model = GuardLens(config)
        model.encode_turns = lambda input_ids, attention_mask: torch.randn(
            input_ids.size(0), input_ids.size(1), input_ids.size(2),
            config.backbone_dim,
        )
        ids = torch.ones(1, 3, 2, dtype=torch.long)
        attention = torch.ones_like(ids)
        turns = torch.ones(1, 3, dtype=torch.long)
        roles = torch.tensor([[0, 1, 0]])
        output = model(
            ids, attention, turns, roles, compute_localization=False
        )
        self.assertIsNotNone(output["attr_probs"])
        output["cls_logits"].sum().backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.any(parameter.grad != 0)
            for parameter in model.attr_head.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None and torch.any(parameter.grad != 0)
            for parameter in model.fusion_gate.parameters()
        ))

    def test_hierarchical_fusion_is_rejected(self):
        config = tiny_config()
        config.use_attribution_fusion = True
        with self.assertRaisesRegex(ValueError, "requires architecture_mode=cross_token"):
            GuardLens(config)

    def test_config_has_no_cf_oversampling_or_phase3_switches(self):
        config = tiny_config()
        self.assertFalse(hasattr(config, "oversample_cf"))
        self.assertFalse(hasattr(config, "cf_oversample_factor"))
        self.assertFalse(hasattr(config, "phase3_epochs"))
        self.assertFalse(hasattr(config, "lambda_cf"))
        self.assertFalse(hasattr(config, "test_path"))

    def test_evidence_turn_head_masks_assistant_turns_in_both_designs(self):
        for mode in ("hierarchical_turn", "cross_token"):
            with self.subTest(mode=mode):
                config = tiny_config()
                config.architecture_mode = mode
                model = GuardLens(config)
                model.encode_turns = lambda input_ids, attention_mask: torch.randn(
                    input_ids.size(0), input_ids.size(1), input_ids.size(2),
                    config.backbone_dim,
                )
                input_ids = torch.ones(1, 3, 2, dtype=torch.long)
                attention = torch.ones_like(input_ids)
                turn_mask = torch.ones(1, 3, dtype=torch.long)
                roles = torch.tensor([[0, 1, 0]], dtype=torch.long)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    turn_mask=turn_mask,
                    role_ids=roles,
                    compute_localization=True,
                )
                self.assertLess(float(out["turn_probs"][0, 1]), 1e-8)

    def test_frozen_backbone_encodes_turns_in_length_local_microbatches(self):
        config = tiny_config()
        config.freeze_backbone = True
        config.backbone_turn_microbatch = 2

        class FakeBackbone(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.calls = []

            def forward(self, input_ids, attention_mask):
                self.calls.append(tuple(input_ids.shape))
                hidden = input_ids.float().unsqueeze(-1).repeat(
                    1, 1, self.dim
                )
                return SimpleNamespace(last_hidden_state=hidden)

        model = GuardLens(config)
        fake = FakeBackbone(config.backbone_dim)
        model.backbone = fake
        model.backbone_loaded = True

        input_ids = torch.tensor([[
            [1, 2, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 0, 0, 0],
        ]])
        attention = torch.tensor([[
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ]])

        hidden = model.encode_turns(input_ids, attention)
        self.assertEqual(tuple(hidden.shape), (1, 3, 6, config.backbone_dim))
        self.assertEqual(fake.calls, [(2, 3), (1, 6)])
        self.assertTrue(torch.all(hidden[0, 0, 2:] == 0))
        self.assertTrue(torch.all(hidden[0, 2, 3:] == 0))

    def test_trainable_backbone_microbatch_path_preserves_gradients(self):
        config = tiny_config()
        config.backbone_turn_microbatch = 1

        class FakeBackbone(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(2.0))
                self.dim = dim
                self.calls = []

            def forward(self, input_ids, attention_mask):
                self.calls.append(tuple(input_ids.shape))
                hidden = input_ids.float().unsqueeze(-1).repeat(
                    1, 1, self.dim
                )
                return SimpleNamespace(last_hidden_state=hidden * self.scale)

        model = GuardLens(config)
        fake = FakeBackbone(config.backbone_dim)
        model.backbone = fake
        model.backbone_loaded = True
        model.backbone_fully_frozen = False
        ids = torch.tensor([[[1, 2, 0], [1, 2, 3]]])
        mask = torch.tensor([[[1, 1, 0], [1, 1, 1]]])
        hidden = model.encode_turns(ids, mask)
        hidden.sum().backward()
        self.assertEqual(fake.calls, [(1, 2), (1, 3)])
        self.assertIsNotNone(fake.scale.grad)
        self.assertGreater(float(fake.scale.grad), 0.0)

    def test_span_head_masks_assistant_and_padding_tokens_in_both_designs(self):
        for mode in ("hierarchical_turn", "cross_token"):
            with self.subTest(mode=mode):
                config = tiny_config()
                config.architecture_mode = mode
                model = GuardLens(config)
                model.encode_turns = lambda input_ids, attention_mask: torch.randn(
                    input_ids.size(0), input_ids.size(1), input_ids.size(2),
                    config.backbone_dim,
                )
                input_ids = torch.ones(1, 2, 3, dtype=torch.long)
                attention = torch.tensor([[[1, 1, 0], [1, 1, 1]]])
                localization = torch.tensor([[[0, 1, 0], [0, 1, 1]]])
                turn_mask = torch.ones(1, 2, dtype=torch.long)
                roles = torch.tensor([[0, 1]], dtype=torch.long)
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    turn_mask=turn_mask,
                    role_ids=roles,
                    localization_mask=localization,
                    compute_localization=True,
                )
                self.assertLess(float(out["attr_probs"][0, 0, 0]), 1e-8)
                self.assertLess(float(out["attr_probs"][0, 0, 2]), 1e-8)
                self.assertTrue(torch.all(out["attr_probs"][0, 1] < 1e-8))

    def test_localization_ramp_reaches_full_weight_early_and_plateaus(self):
        config = GuardLensConfig(
            max_epochs=20,
            phase1_epochs=5,
            localization_ramp_epochs=5,
            localization_ramp_start=0.25,
            lambda_detection=1.0,
            lambda_turn=1.0,
            lambda_span=1.0,
        )
        self.assertEqual(get_lambda_schedule(4, config), (1.0, 0.0, 0.0))
        self.assertEqual(get_lambda_schedule(5, config), (1.0, 0.25, 0.25))
        self.assertEqual(get_lambda_schedule(9, config), (1.0, 1.0, 1.0))
        self.assertEqual(get_lambda_schedule(15, config), (1.0, 1.0, 1.0))
        self.assertEqual(get_lambda_schedule(19, config), (1.0, 1.0, 1.0))

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
