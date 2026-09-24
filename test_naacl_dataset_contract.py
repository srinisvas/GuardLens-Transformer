#!/usr/bin/env python3
"""Regression tests for the redesigned NAACL representation contract."""
import json
import os
import tempfile
import unittest

from guardlens.config import GuardLensConfig
import guardlens.data.dataset as dataset_module
from guardlens.data.dataset import GuardLensCollator, GuardLensDataset
from guardlens.data.audit_representation import audit_split
from guardlens.data.training_contract import training_label


class _FakeTokenizer:
    pad_token_id = 0
    sep_token = "[SEP]"

    def __call__(
        self,
        text,
        padding=False,
        truncation=False,
        return_offsets_mapping=False,
        add_special_tokens=True,
        return_tensors=None,
        max_length=None,
    ):
        ids = [101]
        offsets = [(0, 0)]
        for idx, _ in enumerate(text):
            ids.append(10 + (idx % 50))
            offsets.append((idx, idx + 1))
        ids.append(102)
        offsets.append((0, 0))
        result = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def base_record(label=1):
    return {
        "conversation_id": "r1",
        "corpus_source": "frontier_authored_v3",
        "label": label,
        "loss_weight": 0.6,
        "supervision_tier": "llm_confirmed" if label else "benign_validated",
        "localization_supervision_ignore": False,
        "turns": [
            {
                "turn_id": 0,
                "role": "user",
                "text": "alpha",
                "span_annotations": [],
            },
            {
                "turn_id": 1,
                "role": "assistant",
                "text": "beta",
                "span_annotations": [],
            },
            {
                "turn_id": 2,
                "role": "user",
                "text": "gamma",
                "span_annotations": [],
            },
        ],
    }


class NaaclDatasetContractTests(unittest.TestCase):
    def test_pre_response_is_default_and_retrospective_is_explicit(self):
        record = base_record(0)
        record["turns"].append({
            "turn_id": 3,
            "role": "assistant",
            "text": "final answer unavailable at decision time",
            "span_annotations": [],
        })
        pre = GuardLensDataset([record], GuardLensConfig())[0]
        retrospective = GuardLensDataset(
            [record], GuardLensConfig(input_view="retrospective")
        )[0]
        self.assertEqual(pre["turn_texts"], ["alpha", "beta", "gamma"])
        self.assertEqual(len(retrospective["turn_texts"]), 4)

    def test_primary_detection_weight_is_not_localization_confidence(self):
        record = base_record(1)
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["detection_weight"], 1.0)

    def test_multi_evidence_turns_are_multi_label_targets(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_weak"
        record["evidence_turn_ids"] = [0, 2]
        record["frontier_evidence_analysis"] = {
            "turn_interventions": [
                {"turn_id": 0, "status": "supported_weak"},
                {"turn_id": 2, "status": "supported_weak"},
            ]
        }
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [1, -1, 1])
        self.assertEqual(item["evidence_turn_weights"], [0.70, 0.0, 0.70])

    def test_llm_confirmed_record_can_have_supported_turn_evidence(self):
        record = base_record(1)
        record["supervision_tier"] = "llm_confirmed"
        record["evidence_turn_ids"] = [2]
        record["frontier_evidence_analysis"] = {
            "turn_interventions": [
                {"turn_id": 0, "status": "not_supported"},
                {"turn_id": 2, "status": "supported_strong"},
            ]
        }
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [0, -1, 1])
        self.assertEqual(item["evidence_turn_weights"], [1.0, 0.0, 1.0])

    def test_llm_confirmed_evidence_id_requires_local_intervention_support(self):
        record = base_record(1)
        record["supervision_tier"] = "llm_confirmed"
        record["evidence_turn_ids"] = [2]
        record["pivot_turn_id"] = 2
        with self.assertRaisesRegex(
            RuntimeError,
            "evidence_turn_ids contains turn 2 without an explicit supported",
        ):
            GuardLensDataset([record], GuardLensConfig())[0]

    def test_supported_turn_must_be_declared_in_evidence_ids(self):
        record = base_record(1)
        record["supervision_tier"] = "llm_confirmed"
        record["evidence_turn_ids"] = []
        record["frontier_evidence_analysis"] = {
            "turn_interventions": [
                {"turn_id": 2, "status": "supported_weak"},
            ]
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "intervention-backed positive turns.*missing from evidence_turn_ids",
        ):
            GuardLensDataset([record], GuardLensConfig())[0]

    def test_legacy_single_pivot_is_not_a_turn_target(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_strong"
        record["pivot_turn_id"] = 2
        record["evidence_turn_ids"] = []
        with self.assertRaisesRegex(RuntimeError, "pivot_turn_id is never used"):
            GuardLensDataset([record], GuardLensConfig())[0]

    def test_record_strong_tier_does_not_upgrade_weak_evidence_turn(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_strong"
        record["evidence_turn_ids"] = [0, 2]
        record["turns"][0]["span_annotations"] = [{
            "causal_type": "causal",
            "evidence_status": "supported_weak",
            "supervision_tier": "cf_weak",
            "char_start": 0,
            "char_end": 5,
        }]
        record["turns"][2]["span_annotations"] = [{
            "causal_type": "causal",
            "evidence_status": "supported_strong",
            "supervision_tier": "cf_strong",
            "char_start": 0,
            "char_end": 5,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [1, -1, 1])
        self.assertEqual(item["evidence_turn_weights"], [0.70, 0.0, 1.0])

    def test_explicit_not_supported_turn_is_negative(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_strong"
        record["evidence_turn_ids"] = [2]
        record["frontier_evidence_analysis"] = {
            "turn_interventions": [
                {"turn_id": 0, "status": "not_supported"},
                {"turn_id": 2, "status": "supported_strong"},
            ]
        }
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [0, -1, 1])

    def test_contradictory_evidence_turn_and_negative_intervention_fails_closed(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_strong"
        record["evidence_turn_ids"] = [0]
        record["frontier_evidence_analysis"] = {
            "turn_interventions": [
                {"turn_id": 0, "status": "not_supported"},
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "evidence_turn_ids contains turn 0"):
            GuardLensDataset([record], GuardLensConfig())[0]

    def test_supported_span_can_override_negative_whole_turn_intervention(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_weak"
        record["evidence_turn_ids"] = [0]
        record["frontier_evidence_analysis"] = {
            "turn_interventions": [
                {"turn_id": 0, "status": "not_supported"},
            ]
        }
        record["turns"][0]["span_annotations"] = [{
            "causal_type": "causal",
            "evidence_status": "supported_weak",
            "supervision_tier": "cf_weak",
            "char_start": 0,
            "char_end": 5,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [1, -1, -1])
        self.assertEqual(item["evidence_turn_weights"], [0.70, 0.0, 0.0])

    def test_multiple_positive_turn_sources_keep_strongest_confidence(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_strong"
        record["evidence_turn_ids"] = [0]
        record["frontier_evidence_analysis"] = {
            "turn_interventions": [
                {"turn_id": 0, "status": "supported_weak"},
            ]
        }
        record["turns"][0]["span_annotations"] = [{
            "causal_type": "causal",
            "evidence_status": "supported_strong",
            "supervision_tier": "cf_strong",
            "char_start": 0,
            "char_end": 5,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [1, -1, -1])
        self.assertEqual(item["evidence_turn_weights"], [1.0, 0.0, 0.0])

    def test_benign_user_turns_are_negative_and_assistant_ignored(self):
        item = GuardLensDataset([base_record(0)], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [0, -1, 0])

    def test_legacy_anchor_intervention_can_supply_explicit_negative(self):
        record = base_record(1)
        record["supervision_tier"] = "llm_confirmed"
        record["evidence_analysis"] = {
            "fresh_anchor_turn_id": 0,
            "anchor_turn_intervention": {"status": "not_supported"},
        }
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [0, -1, -1])

    def test_llm_confirmed_legacy_anchor_can_supply_positive_turn(self):
        record = base_record(1)
        record["supervision_tier"] = "llm_confirmed"
        record["evidence_turn_ids"] = [0]
        record["evidence_analysis"] = {
            "fresh_anchor_turn_id": 0,
            "anchor_turn_intervention": {"status": "supported_weak"},
        }
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["evidence_turn_labels"], [1, -1, -1])
        self.assertEqual(item["evidence_turn_weights"], [0.70, 0.0, 0.0])

    def test_llm_confirmed_causal_looking_span_is_ignored(self):
        record = base_record(1)
        record["supervision_tier"] = "llm_confirmed"
        record["turns"][0]["span_annotations"] = [{
            "label": "MALICIOUS_TRIGGER",
            "causal_type": "causal",
            "supervision_tier": "llm_confirmed",
            "char_start": 0,
            "char_end": 5,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertTrue(all(x == -1 for x in item["char_labels"][0]))

    def test_counterfactually_supported_span_is_positive(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_strong"
        record["evidence_turn_ids"] = [0]
        record["turns"][0]["span_annotations"] = [{
            "label": "EVIDENCE_CANDIDATE",
            "causal_type": "causal",
            "evidence_status": "supported_strong",
            "supervision_tier": "cf_strong",
            "char_start": 0,
            "char_end": 5,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["char_labels"][0], [1, 1, 1, 1, 1])

    def test_overlapping_strong_and_weak_positive_spans_keep_strong_weight(self):
        record = base_record(1)
        record["supervision_tier"] = "cf_strong"
        record["evidence_turn_ids"] = [0]
        record["turns"][0]["span_annotations"] = [
            {
                "causal_type": "causal",
                "evidence_status": "supported_strong",
                "supervision_tier": "cf_strong",
                "char_start": 0,
                "char_end": 5,
            },
            {
                "causal_type": "causal",
                "evidence_status": "supported_weak",
                "supervision_tier": "cf_weak",
                "char_start": 1,
                "char_end": 4,
            },
        ]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["char_labels"][0], [1, 1, 1, 1, 1])
        self.assertEqual(item["char_tier_weights"][0], [1.0] * 5)

    def test_semantic_construction_mask_is_fail_closed(self):
        record = base_record(1)
        record["evidence_turn_ids"] = [0]
        record["turns"][0]["span_annotations"] = [{
            "causal_type": "causal",
            "evidence_status": "supported_strong",
            "supervision_tier": "cf_strong",
            "semantic_token_supervision_ignore": True,
            "char_start": 0,
            "char_end": 5,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertTrue(all(x == -1 for x in item["char_labels"][0]))
        self.assertEqual(item["evidence_turn_labels"], [1, -1, -1])

    def test_cf_tier_without_matching_intervention_status_is_ignored(self):
        record = base_record(1)
        record["turns"][0]["span_annotations"] = [{
            "causal_type": "causal",
            "evidence_status": "not_supported",
            "supervision_tier": "cf_strong",
            "char_start": 0,
            "char_end": 5,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertTrue(all(x == -1 for x in item["char_labels"][0]))

    def test_benign_negative_incidental_span_is_explicit_negative(self):
        record = base_record(0)
        record["turns"][0]["span_annotations"] = [{
            "causal_type": "incidental",
            "evidence_status": "benign_negative",
            "supervision_tier": "incidental",
            "char_start": 0,
            "char_end": 5,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(item["char_labels"][0], [0, 0, 0, 0, 0])

    def test_assistant_span_supervision_is_ignored_by_dataset(self):
        record = base_record(1)
        record["turns"][1]["span_annotations"] = [{
            "causal_type": "causal",
            "evidence_status": "supported_strong",
            "supervision_tier": "cf_strong",
            "char_start": 0,
            "char_end": 4,
        }]
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertTrue(all(x == -1 for x in item["char_labels"][1]))
        self.assertTrue(all(x == 0.0 for x in item["char_tier_weights"][1]))

    def test_representation_audit_rejects_target_bearing_assistant_span(self):
        record = base_record(1)
        record["turns"][1]["span_annotations"] = [{
            "causal_type": "causal",
            "evidence_status": "supported_strong",
            "supervision_tier": "cf_strong",
            "char_start": 0,
            "char_end": 4,
        }]
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".jsonl",
                delete=False,
                encoding="utf-8",
            ) as handle:
                path = handle.name
                handle.write(json.dumps(record) + "\n")
            result = audit_split(
                "train",
                path,
                _FakeTokenizer(),
                max_turns=48,
                max_tokens=20,
            )
            self.assertTrue(
                any(
                    "target-bearing span supervision" in error
                    for error in result["errors"]
                )
            )
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

    def test_legacy_flat_collator_is_removed(self):
        self.assertFalse(hasattr(dataset_module, "FlatConversationCollator"))

    def test_auxiliary_uses_detection_label_and_masks_localization(self):
        record = base_record(0)
        record.update({
            "authoring_intent_label": 0,
            "corpus_source": "frontier_authored_v3_auxiliary",
            "detection_label": 1,
            "detection_loss_weight": 0.25,
            "pivot_loss_weight": 0.0,
            "span_loss_weight": 0.0,
            "auxiliary_detection_only": True,
            "use_as": "auxiliary_detection_only",
            "pivot_supervision_ignore": True,
            "pivot_turn_id": None,
            "evidence_turn_ids": [],
            "localization_supervision_ignore": True,
            "supervision_tier": "auxiliary_detection",
        })
        item = GuardLensDataset([record], GuardLensConfig())[0]
        self.assertEqual(training_label(record), 1)
        self.assertEqual(item["label"], 1)
        self.assertEqual(item["detection_weight"], 0.25)
        self.assertTrue(all(x == -1 for x in item["evidence_turn_labels"]))

    def test_dataset_refuses_silent_turn_truncation(self):
        record = base_record(1)
        record["turns"] = [
            {"turn_id": i, "role": "user", "text": "x", "span_annotations": []}
            for i in range(4)
        ]
        with self.assertRaisesRegex(RuntimeError, "silent turn truncation"):
            GuardLensDataset(
                [record], GuardLensConfig(max_turns=3)
            )[0]

    def test_representation_audit_reports_turn_overflow_without_traceback(self):
        record = base_record(1)
        record["turns"] = [
            {
                "turn_id": i,
                "role": "user",
                "text": "x",
                "span_annotations": [],
            }
            for i in range(4)
        ]
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".jsonl",
                delete=False,
                encoding="utf-8",
            ) as handle:
                path = handle.name
                handle.write(json.dumps(record) + "\n")
            result = audit_split(
                "train",
                path,
                _FakeTokenizer(),
                max_turns=3,
                max_tokens=20,
            )
            self.assertTrue(
                any("exceed max_turns=3" in error for error in result["errors"])
            )
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

    def test_collator_refuses_silent_token_truncation(self):
        record = base_record(0)
        record["turns"] = [{
            "turn_id": 0,
            "role": "user",
            "text": "abcdefghij",
            "span_annotations": [],
        }]
        config = GuardLensConfig(max_tokens_per_turn=8)
        item = GuardLensDataset([record], config)[0]
        collator = GuardLensCollator(_FakeTokenizer(), config)
        with self.assertRaisesRegex(RuntimeError, "silent token truncation"):
            collator([item])

    def test_dynamic_padding_uses_batch_max_not_hard_ceiling(self):
        short = base_record(0)
        short["conversation_id"] = "short"
        short["turns"] = [{
            "turn_id": 0, "role": "user", "text": "abc", "span_annotations": []
        }]
        long = base_record(0)
        long["conversation_id"] = "long"
        long["turns"] = [{
            "turn_id": 0, "role": "user", "text": "abcdef", "span_annotations": []
        }]
        config = GuardLensConfig(max_tokens_per_turn=20)
        ds = GuardLensDataset([short, long], config)
        batch = GuardLensCollator(_FakeTokenizer(), config)([ds[0], ds[1]])
        self.assertEqual(batch["input_ids"].shape[-1], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
