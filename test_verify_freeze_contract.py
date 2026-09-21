"""Regression tests for the DataGen-to-Transformer freeze-report contract."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parent / "guardlens" / "data" / "verify_freeze.py"
)
SPEC = importlib.util.spec_from_file_location("verify_freeze_under_test", MODULE_PATH)
VERIFY_FREEZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY_FREEZE)
expected_artifacts = VERIFY_FREEZE.expected_artifacts
main = VERIFY_FREEZE.main


FINAL_TRAIN_SHA256 = (
    "2927d73c9c471ca9835159d6162e62fd8bb75f4a501b56536bfe1f41dffe2b80"
)
FINAL_DEV_SHA256 = (
    "16f500cd6a67806f834aa761e388116a2ba03b1b136232cd2ec504aa9261833d"
)


def restored_a_report(train_sha: str, dev_sha: str, train_n: int, dev_n: int):
    return {
        "status": "passed",
        "artifact_sha256": {
            "primary_train": "0" * 64,
            "primary_dev": dev_sha,
            "auxiliary_candidate_train": train_sha,
            "auxiliary_candidate_dev": dev_sha,
        },
        "counts": {"primary_splits": {"train": 1706, "dev": dev_n}},
        "auxiliary_candidate": {
            "train_records": train_n,
            "dev_records": dev_n,
            "a_auxiliary_records_in_train": 721,
            "b_auxiliary_records_in_train": 425,
        },
    }


class VerifyFreezeContractTest(unittest.TestCase):
    def test_final_restored_a_report_fields_are_consumed(self):
        report = restored_a_report(
            FINAL_TRAIN_SHA256,
            FINAL_DEV_SHA256,
            train_n=2852,
            dev_n=364,
        )
        self.assertEqual(
            expected_artifacts(report, "primary_plus_auxiliary"),
            (FINAL_TRAIN_SHA256, FINAL_DEV_SHA256, 2852, 364),
        )

    def test_missing_restored_a_hash_fails_with_field_name(self):
        report = restored_a_report("0" * 64, "1" * 64, 2852, 364)
        del report["artifact_sha256"]["auxiliary_candidate_train"]
        with self.assertRaisesRegex(
            RuntimeError,
            "artifact_sha256.auxiliary_candidate_train",
        ):
            expected_artifacts(report, "primary_plus_auxiliary")

    def test_cli_verifies_hashes_and_counts_and_records_expectations(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.jsonl")
            dev_path = os.path.join(tmp, "dev.jsonl")
            report_path = os.path.join(tmp, "report.json")
            output_path = os.path.join(tmp, "verification.json")
            with open(train_path, "w", encoding="utf-8") as handle:
                handle.write('{"conversation_id":"train-1"}\n')
                handle.write('{"conversation_id":"train-2"}\n')
            with open(dev_path, "w", encoding="utf-8") as handle:
                handle.write('{"conversation_id":"dev-1"}\n')

            def digest(path):
                with open(path, "rb") as handle:
                    return hashlib.sha256(handle.read()).hexdigest()

            report = restored_a_report(
                digest(train_path), digest(dev_path), train_n=2, dev_n=1
            )
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle)

            argv = [
                "verify_freeze",
                "--report",
                report_path,
                "--train",
                train_path,
                "--dev",
                dev_path,
                "--variant",
                "primary_plus_auxiliary",
                "--output",
                output_path,
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                io.StringIO()
            ):
                main()

            with open(output_path, encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["report"]["expected_train_records"], 2)
            self.assertEqual(result["report"]["expected_dev_records"], 1)


if __name__ == "__main__":
    unittest.main()
