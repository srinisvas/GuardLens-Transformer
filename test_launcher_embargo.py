#!/usr/bin/env python3
"""Regression tests for the internal-signoff evaluation embargo."""
import os
import stat
import subprocess
import tempfile
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
    def test_submitter_builds_record_gates_before_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            capture = temporary / "sbatch.args"
            state = temporary / "sbatch.state"
            stub = temporary / "sbatch"
            stub.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "capture=$MOCK_SBATCH_CAPTURE\n"
                "state=$MOCK_SBATCH_STATE\n"
                "printf '%s\\n' \"$*\" >> \"$capture\"\n"
                "current=0\n"
                "[[ ! -f \"$state\" ]] || current=$(<\"$state\")\n"
                "current=$((current + 1))\n"
                "printf '%s\\n' \"$current\" > \"$state\"\n"
                "printf '%s\\n' \"$current\"\n",
                encoding="utf-8",
            )
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
            environment = {
                **os.environ,
                "PATH": f"{temporary}:{os.environ['PATH']}",
                "MATRIX_ROOT": str(temporary / "matrix"),
                "MOCK_SBATCH_CAPTURE": str(capture),
                "MOCK_SBATCH_STATE": str(state),
            }
            result = subprocess.run(
                ["bash", str(ROOT / "submit_train_naacl_diagnostic.sh")],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            submissions = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(submissions), 22)
            for smoke_id, train_id in zip((10, 13, 16, 19), (6, 7, 8, 9)):
                smoke = submissions[smoke_id - 1]
                array = submissions[smoke_id]
                self.assertIn(f"--dependency=afterok:{train_id}", smoke)
                self.assertIn("EVAL_RECORD_INDEX=136", smoke)
                self.assertIn("MAX_REQUEUES=0", smoke)
                self.assertIn(f"--dependency=afterok:{smoke_id}", array)
                self.assertIn("--array=0-3%4", array)
                self.assertNotIn("EVAL_RECORD_INDEX", array)
            self.assertIn("--dependency=afterok:12:15:18:21", submissions[-1])

    def test_resume_builds_record_gates_before_replacement_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            matrix = temporary / "matrix"
            (matrix / "shared" / "preflight").mkdir(parents=True)
            (matrix / "shared" / "internal-dev.jsonl").write_text(
                "{}\n" * 364, encoding="utf-8"
            )
            candidates = (
                "hierarchical_sibling_retro_frozen",
                "cross_token_sibling_retro_frozen",
                "cross_token_gated_retro_frozen",
                "cross_token_gated_retro_top4",
            )
            for candidate in candidates:
                checkpoint = matrix / "training" / candidate / "checkpoints"
                checkpoint.mkdir(parents=True)
                (checkpoint / "training_summary.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (checkpoint / "best.pt").write_bytes(b"fixture")
            capture = temporary / "sbatch.args"
            state = temporary / "sbatch.state"
            stub = temporary / "sbatch"
            stub.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$MOCK_SBATCH_CAPTURE\"\n"
                "current=0\n"
                "[[ ! -f \"$MOCK_SBATCH_STATE\" ]] || current=$(<\"$MOCK_SBATCH_STATE\")\n"
                "current=$((current + 1))\n"
                "printf '%s\\n' \"$current\" > \"$MOCK_SBATCH_STATE\"\n"
                "printf '%s\\n' \"$current\"\n",
                encoding="utf-8",
            )
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
            environment = {
                **os.environ,
                "PATH": f"{temporary}:{os.environ['PATH']}",
                "MOCK_SBATCH_CAPTURE": str(capture),
                "MOCK_SBATCH_STATE": str(state),
            }
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "resume_train_naacl_diagnostic.sh"),
                    str(matrix),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            submissions = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(submissions), 13)
            for smoke_id in (1, 4, 7, 10):
                smoke = submissions[smoke_id - 1]
                array = submissions[smoke_id]
                self.assertIn("EVAL_RECORD_INDEX=136", smoke)
                self.assertIn("MAX_REQUEUES=0", smoke)
                self.assertIn(f"--dependency=afterok:{smoke_id}", array)
                self.assertNotIn("EVAL_RECORD_INDEX", array)
            self.assertIn("--dependency=afterok:3:6:9:12", submissions[-1])

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

    def test_long_gpu_jobs_requeue_without_hiding_hard_failures(self):
        for name, checkpoint in (
            ("train_naacl.slurm", '"$OUTPUT/last.pt"'),
            ("eval_internal_dev.slurm", '"$EVAL_OUTPUT/manifest.json"'),
        ):
            with self.subTest(name=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("#SBATCH --requeue", text)
                self.assertIn("#SBATCH --signal=B:USR1@300", text)
                self.assertIn("SLURM_RESTART_COUNT", text)
                self.assertIn("scontrol requeue", text)
                self.assertIn(checkpoint, text)
                self.assertIn("if (( requeue_requested == 1 )); then", text)

        train = (ROOT / "train_naacl.slurm").read_text(encoding="utf-8")
        self.assertIn("if (( train_status != 0 )); then", train)
        diagnostic = (ROOT / "eval_internal_dev.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("if (( diagnostic_status != 0 )); then", diagnostic)
        self.assertIn("COMPLETED_AT_START=$(count_completed_work)", diagnostic)
        self.assertIn("COMPLETED_NOW=$(count_completed_work)", diagnostic)
        self.assertIn("no completed-record progress", diagnostic)

    def test_internal_diagnostics_use_all_four_gpus_and_matching_finalizer(self):
        for name in (
            "submit_train_naacl_diagnostic.sh",
            "resume_train_naacl_diagnostic.sh",
        ):
            with self.subTest(name=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn('EVAL_SHARDS="${EVAL_SHARDS:-4}"', text)
                self.assertIn('--array="0-$((EVAL_SHARDS - 1))%$EVAL_SHARDS"', text)
                self.assertIn("EVAL_SHARDS=$EVAL_SHARDS", text)
                self.assertIn('DIAGNOSTIC_SMOKE_INDEX="${DIAGNOSTIC_SMOKE_INDEX:-136}"', text)
                self.assertIn("EVAL_RECORD_INDEX=$DIAGNOSTIC_SMOKE_INDEX", text)
                self.assertIn("MAX_REQUEUES=0", text)
                self.assertIn('logs/eval_internal_dev_%A_%a.out', text)
                self.assertIn("--kill-on-invalid-dep=yes", text)
        diagnostic = (ROOT / "eval_internal_dev.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("SLURM_ARRAY_TASK_ID", diagnostic)
        self.assertIn('--shard-index "$EVAL_SHARD_INDEX"', diagnostic)
        self.assertIn('--shard-count "$EVAL_SHARDS"', diagnostic)
        self.assertIn('--record-index "$EVAL_RECORD_INDEX"', diagnostic)
        finalizer = (ROOT / "eval_internal_dev_finalize.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn('--shard-count "$EVAL_SHARDS"', finalizer)

    def test_resume_rejects_mixed_evaluator_provenance(self):
        text = (ROOT / "resume_train_naacl_diagnostic.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('manifest.get("code", {}).get("git_sha")', text)
        self.assertIn("archive the complete diagnostics directory", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
