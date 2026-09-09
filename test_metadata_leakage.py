#!/usr/bin/env python3
"""Regression test: construction/provenance metadata must not become model input."""
from __future__ import annotations

import copy
import unittest

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset


class TrainingMetadataLeakageTests(unittest.TestCase):
    def test_hidden_metadata_cannot_change_model_visible_turn_features(self):
        base = {
            "conversation_id": "visible-id-a",
            "pair_id": "PAIR_SECRET_A",
            "label": 1,
            "loss_weight": 0.6,
            "difficulty": "hard",
            "family": "frontier_authored_adversarial",
            "pivot_turn_id": 2,
            "pivot_kind": "contextual_pivot",
            "pivot_supervision_ignore": False,
            "supervision_tier": "llm_confirmed",
            "metadata": {
                "generator": "AUTHOR_SECRET_A",
                "corpus_version": "CORPUS_SECRET_A",
                "scenario_family": "SCENARIO_SECRET_A",
                "mechanism_family": "MECHANISM_SECRET_A",
            },
            "intended_structure": {
                "notes": "INTENDED_SECRET_A",
                "expected_pivot_turn_ids": [2],
            },
            "turns": [
                {
                    "turn_id": 0,
                    "role": "user",
                    "text": "VISIBLE USER TEXT",
                    "span_annotations": [],
                },
                {
                    "turn_id": 1,
                    "role": "assistant",
                    "text": "VISIBLE ASSISTANT TEXT",
                    "span_annotations": [],
                },
                {
                    "turn_id": 2,
                    "role": "user",
                    "text": "VISIBLE FINAL USER TEXT",
                    "span_annotations": [],
                },
            ],
        }
        changed = copy.deepcopy(base)
        changed["conversation_id"] = "visible-id-b"
        changed["pair_id"] = "PAIR_SECRET_B"
        changed["metadata"] = {
            "generator": "AUTHOR_SECRET_B",
            "corpus_version": "CORPUS_SECRET_B",
            "scenario_family": "SCENARIO_SECRET_B",
            "mechanism_family": "MECHANISM_SECRET_B",
        }
        changed["intended_structure"] = {
            "notes": "INTENDED_SECRET_B",
            "expected_pivot_turn_ids": [0],
        }

        config = GuardLensConfig(max_turns=48, max_tokens_per_turn=192)
        item_a = GuardLensDataset([base], config)[0]
        item_b = GuardLensDataset([changed], config)[0]

        # These are the only record-derived features that flow into tokenizer/
        # model inputs. Hidden provenance/construction metadata must not affect them.
        self.assertEqual(item_a["turn_texts"], item_b["turn_texts"])
        self.assertEqual(item_a["turn_roles"], item_b["turn_roles"])
        self.assertEqual(item_a["char_labels"], item_b["char_labels"])
        self.assertEqual(
            item_a["char_tier_weights"], item_b["char_tier_weights"]
        )

        # The metadata may remain available separately for auditing/reporting.
        self.assertNotEqual(item_a["conversation_id"], item_b["conversation_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
