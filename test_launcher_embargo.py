#!/usr/bin/env python3
"""Regression tests for the internal-signoff evaluation embargo."""
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEGACY = (
    "bootstrap_ci.slurm",
    "eval_ablation_causal.slurm",
    "eval_boundary.slurm",
    "eval_core.slurm",
    "eval_deconfound.slurm",
    "eval_full.slurm",
    "eval_human_benchmark.slurm",
    "eval_phase3.slurm",
    "eval_review_p2.slurm",
    "eval_review_p3.slurm",
    "eval_review_response.slurm",
    "eval_stage1_setup.slurm",
    "eval_stage2_metrics.slurm",
    "eval_stage3_cross_dataset.slurm",
    "eval_stage4_paraphrase.slurm",
    "eval_stage5_collate.slurm",
    "eval_stage6_precision_transfer.slurm",
    "eval_surrogate_target.slurm",
    "eval_target_llm_causal_v3_qwen.slurm",
    "eval_transfer_stable.slurm",
    "submit_eval_pipeline.sh",
)


class LauncherEmbargoTests(unittest.TestCase):
    def test_legacy_evaluation_launchers_fail_before_data_access(self):
        marker = "legacy evaluation launcher disabled pending V4 internal signoff"
        for name in LEGACY:
            with self.subTest(name=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn(marker, text[:2000])
                self.assertLess(text.index(marker), text.find("TEST_") if "TEST_" in text else len(text))

    def test_mhj_requires_explicit_post_signoff_token(self):
        marker = "EVAL_EXTERNAL_SIGNOFF"
        for name in ("submit_eval_mhj.sh", "eval_mhj.slurm", "eval_mhj_finalize.slurm"):
            with self.subTest(name=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn(marker, text[:1500])
                self.assertIn("APPROVED_AFTER_INTERNAL_SIGNOFF", text[:1500])

    def test_internal_matrix_uses_four_retrospective_controlled_candidates(self):
        expected = (
            "hierarchical_sibling_retro_frozen",
            "cross_token_sibling_retro_frozen",
            "cross_token_gated_retro_frozen",
            "cross_token_gated_retro_top4",
        )
        for name in (
            "submit_train_naacl_diagnostic.sh",
            "resume_train_naacl_diagnostic.sh",
        ):
            with self.subTest(name=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                for candidate in expected:
                    self.assertIn(candidate, text)
                self.assertNotIn("cross_token_gated_pre", text)
                self.assertIn(
                    "views=(retrospective retrospective retrospective retrospective)",
                    text,
                )
                self.assertIn("trainable_layers=(0 0 0 4)", text)
                self.assertIn("backbone_microbatches=(8 8 8 1)", text)
                self.assertIn("TRAIN_PATH=$TRAIN_PATH", text)
                self.assertIn("DEV_PATH=$DEV_PATH", text)
                self.assertIn("OUTPUT=", text)

        comparison = (ROOT / "compare_internal_dev_matrix.slurm").read_text(
            encoding="utf-8"
        )
        for candidate in expected:
            self.assertIn(candidate, comparison)


if __name__ == "__main__":
    unittest.main(verbosity=2)
