#!/usr/bin/env python3
"""Regression tests for NAACL dataset representation contracts."""
from __future__ import annotations

import unittest

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
