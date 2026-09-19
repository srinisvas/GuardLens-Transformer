#!/usr/bin/env python3
"""Regression tests for NAACL dataset representation contracts."""
from __future__ import annotations

import unittest

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset
from guardlens.data.training_contract import training_label


class NaaclDatasetContractTests(unittest.TestCase):
    def test_dataset_does_not_clip_turn_text_before_tokenization(self):
        text = "a" * 620 + "CAUSAL" + "b" * 80
        record = {
            "conversation_id": "long-text",
            "label": 1,
            "loss_weight": 1.0,
            "pivot_turn_id": 0,
            "pivot_kind": "contextual_pivot",
            "pivot_supervision_ignore": False,
            "supervision_tier": "cf_strong",
            "turns": [{
                "turn_id": 0,
                "role": "user",
                "text": text,
                "span_annotations": [{
                    "label": "EVIDENCE_CANDIDATE",
                    "causal_type": "causal",
                    "supervision_tier": "cf_strong",
                    "char_start": 620,
                    "char_end": 626,
                }],
            }],
        }
        item = GuardLensDataset(
            [record], GuardLensConfig(max_turns=48, max_tokens_per_turn=192)
        )[0]
        self.assertEqual(item["turn_texts"][0], text)
        self.assertEqual(len(item["char_labels"][0]), len(text))
        self.assertTrue(all(x == 1 for x in item["char_labels"][0][620:626]))

    def test_semantically_masked_evidence_candidate_remains_ignored(self):
        text = "ordinary construction language"
        record = {
            "conversation_id": "masked-span",
            "label": 1,
            "loss_weight": 1.0,
            "pivot_turn_id": 0,
            "pivot_kind": "contextual_pivot",
            "pivot_supervision_ignore": False,
            "supervision_tier": "cf_strong",
            "turns": [{
                "turn_id": 0,
                "role": "user",
                "text": text,
                "span_annotations": [{
                    "label": "EVIDENCE_CANDIDATE",
                    "evidence_status": "supported_strong",
                    "causal_type": "unvalidated",
                    "supervision_tier": "ignore",
                    "semantic_token_supervision_ignore": True,
                    "char_start": 0,
                    "char_end": len(text),
                }],
            }],
        }
        item = GuardLensDataset(
            [record], GuardLensConfig(max_turns=48, max_tokens_per_turn=192)
        )[0]
        self.assertTrue(all(x == -1 for x in item["char_labels"][0]))
        self.assertTrue(all(x == 0.0 for x in item["char_tier_weights"][0]))

    def test_unknown_pivot_ignore_is_preserved_for_collation_contract(self):
        record = {
            "conversation_id": "unknown-pivot",
            "label": 1,
            "loss_weight": 0.6,
            "pivot_turn_id": None,
            "pivot_kind": "none",
            "pivot_supervision_ignore": True,
            "supervision_tier": "llm_confirmed",
            "turns": [{
                "turn_id": 0,
                "role": "user",
                "text": "context",
                "span_annotations": [],
            }],
        }
        item = GuardLensDataset(
            [record], GuardLensConfig(max_turns=48, max_tokens_per_turn=192)
        )[0]
        self.assertTrue(item["pivot_supervision_ignore"])
        self.assertIsNone(item["pivot_turn_id"])


    def test_primary_localization_ignore_masks_all_span_targets(self):
        record = {
            "conversation_id": "primary-localization-unknown",
            "label": 1,
            "loss_weight": 0.6,
            "pivot_turn_id": None,
            "pivot_kind": "none",
            "pivot_supervision_ignore": True,
            "localization_supervision_ignore": True,
            "supervision_tier": "llm_confirmed",
            "turns": [{
                "turn_id": 0, "role": "user", "text": "supported-looking text",
                "span_annotations": [{
                    "label": "MALICIOUS_TRIGGER", "causal_type": "causal",
                    "supervision_tier": "cf_strong", "char_start": 0, "char_end": 9,
                }],
            }],
        }
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertTrue(all(x == -1 for x in item["char_labels"][0]))
        self.assertTrue(all(x == 0.0 for x in item["char_tier_weights"][0]))

    def test_auxiliary_uses_detection_label_and_detection_weight_only(self):
        record = {
            "conversation_id": "aux-benign-authored-unsafe",
            "label": 0, "authoring_intent_label": 0, "detection_label": 1,
            "loss_weight": 0.25, "detection_loss_weight": 0.25,
            "pivot_loss_weight": 0.0, "span_loss_weight": 0.0,
            "auxiliary_detection_only": True, "use_as": "auxiliary_detection_only",
            "pivot_turn_id": None, "pivot_kind": "none",
            "pivot_supervision_ignore": True,
            "localization_supervision_ignore": True,
            "supervision_tier": "auxiliary_detection",
            "turns": [{
                "turn_id": 0, "role": "user", "text": "unsafe realized outcome",
                "span_annotations": [{
                    "label": "MALICIOUS_TRIGGER", "causal_type": "causal",
                    "supervision_tier": "cf_strong", "char_start": 0, "char_end": 6,
                }],
            }],
        }
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(record["label"], 0)
        self.assertEqual(training_label(record), 1)
        self.assertEqual(item["label"], 1)
        self.assertEqual(item["loss_weight"], 0.25)
        self.assertFalse(item["cf_loss_eligible"])
        self.assertTrue(all(x == -1 for x in item["char_labels"][0]))

    def test_auxiliary_malicious_authored_safe_maps_to_negative(self):
        record = {
            "conversation_id": "aux-malicious-authored-safe",
            "label": 1, "authoring_intent_label": 1, "detection_label": 0,
            "loss_weight": 0.25, "detection_loss_weight": 0.25,
            "pivot_loss_weight": 0.0, "span_loss_weight": 0.0,
            "auxiliary_detection_only": True, "use_as": "auxiliary_detection_only",
            "pivot_turn_id": None, "pivot_kind": "none",
            "pivot_supervision_ignore": True,
            "localization_supervision_ignore": True,
            "supervision_tier": "auxiliary_detection",
            "turns": [{"turn_id": 0, "role": "user", "text": "resisted", "span_annotations": []}],
        }
        self.assertEqual(training_label(record), 0)
        self.assertEqual(GuardLensDataset([record], GuardLensConfig())[0]["label"], 0)

    def test_auxiliary_contract_fails_closed_if_localization_is_not_masked(self):
        record = {
            "conversation_id": "bad-aux", "label": 0, "detection_label": 1,
            "loss_weight": 0.25, "detection_loss_weight": 0.25,
            "pivot_loss_weight": 0.0, "span_loss_weight": 0.0,
            "auxiliary_detection_only": True,
            "pivot_supervision_ignore": True,
            "localization_supervision_ignore": False,
            "turns": [{"turn_id": 0, "role": "user", "text": "x", "span_annotations": []}],
        }
        with self.assertRaisesRegex(RuntimeError, "ignore localization"):
            GuardLensDataset([record], GuardLensConfig())[0]


if __name__ == "__main__":
    unittest.main(verbosity=2)
