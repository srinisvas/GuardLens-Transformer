#!/usr/bin/env python3
"""CPU regressions for restored-A training and checkpoint-selection contracts."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guardlens.training.trainer import (
    _average_precision,
    _canonical_detection_score,
    _detection_metrics,
    _prepare_output_dir,
    _resolve_device,
    _source_detection_metrics,
    _validate_frozen_schema,
)


def primary(cid, source, label):
    return {
        "conversation_id": cid,
        "corpus_source": source,
        "label": label,
        "supervision_tier": "llm_confirmed" if label else "benign_validated",
        "evidence_turn_ids": [],
    }


def auxiliary(cid, source, label, weight):
    return {
        "conversation_id": cid,
        "corpus_source": source,
        "label": 0,
        "detection_label": label,
        "detection_loss_weight": weight,
        "supervision_tier": "auxiliary_detection",
        "auxiliary_detection_only": True,
        "use_as": "auxiliary_detection_only",
        "localization_supervision_ignore": True,
        "pivot_supervision_ignore": True,
        "pivot_loss_weight": 0.0,
        "span_loss_weight": 0.0,
        "pivot_turn_id": None,
        "evidence_turn_ids": [],
        "auxiliary_original_annotations_preserved": source == "legacy_detection_aux",
    }


class TrainingReadinessContractTests(unittest.TestCase):
    def test_source_stratified_metrics_keep_b_as_canonical(self):
        probs = [0.9, 0.8, 0.7, 0.1, 0.9, 0.1, 0.8, 0.2]
        labels = [0, 1, 0, 1, 1, 0, 1, 0]
        families = ["A", "A", "A", "A", "B", "B", "B", "B"]
        by_source = _source_detection_metrics(probs, labels, families, 0.5)
        metrics = {
            "detection": _detection_metrics(probs, labels, 0.5),
            "detection_by_source": by_source,
        }
        self.assertLess(by_source["A"]["auprc"], by_source["B"]["auprc"])
        self.assertEqual(
            _canonical_detection_score(metrics), by_source["B"]["auprc"]
        )

    def test_checkpoint_selection_average_precision_is_tie_aware(self):
        labels = [1, 0, 1, 0, 1]
        scores = [.4, .4, .9, .1, .4]
        self.assertAlmostEqual(_average_precision(scores, labels), 5 / 6)

    def test_canonical_variant_requires_both_auxiliary_sources(self):
        train = [
            primary("ta", "legacy_restored_primary", 0),
            primary("tb", "frontier_authored_v3", 1),
            auxiliary("aa", "legacy_detection_aux", 0, 1.0),
            auxiliary("ab", "frontier_authored_v3_auxiliary", 1, 0.25),
        ]
        dev = [
            primary("da", "legacy_restored_primary", 0),
            primary("db0", "frontier_authored_v3", 0),
            primary("db1", "frontier_authored_v3", 1),
        ]
        _validate_frozen_schema(train, dev, "primary_plus_auxiliary")
        with self.assertRaisesRegex(RuntimeError, "A and B auxiliaries"):
            _validate_frozen_schema(train[:-1], dev, "primary_plus_auxiliary")

    def test_cuda_request_fails_closed_when_cuda_is_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA.*unavailable"):
                _resolve_device("cuda")
            self.assertEqual(str(_resolve_device("cpu")), "cpu")

    def test_output_directory_must_be_new(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "run", "checkpoints")
            _prepare_output_dir(output)
            self.assertTrue(os.path.isdir(output))
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                _prepare_output_dir(output)
            _prepare_output_dir(output, resume=True)
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                _prepare_output_dir(os.path.join(root, "missing"), resume=True)

    def test_canonical_launchers_use_auxiliary_training(self):
        root = Path(__file__).resolve().parent
        train_text = (root / "train_naacl.slurm").read_text(encoding="utf-8")
        smoke_text = (root / "smoke_naacl_window.slurm").read_text(encoding="utf-8")
        self.assertIn(
            'TRAIN_VARIANT="${TRAIN_VARIANT:-primary_plus_auxiliary}"',
            train_text,
        )
        self.assertIn(
            'TRAIN_VARIANT="${TRAIN_VARIANT:-primary_plus_auxiliary}"',
            smoke_text,
        )
        self.assertIn("splits_primary_plus_train_auxiliary/train.jsonl", smoke_text)
        self.assertIn('--variant "$VERIFY_VARIANT"', smoke_text)
        self.assertIn("guardlens.data.audit_representation", smoke_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
