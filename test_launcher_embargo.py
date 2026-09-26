#!/usr/bin/env python3
"""Regression tests for the internal-signoff evaluation embargo."""
import hashlib
import json
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

LAUNCHER_ENVIRONMENT_VARIABLES = (
    "MATRIX_ROOT",
    "CONDA_ENV",
    "FREEZE_DIR",
    "REPORT_PATH",
    "TRAIN_TIME",
    "DIAGNOSTIC_TIME",
    "DIAGNOSTIC_SMOKE_TIME",
    "DIAGNOSTIC_SMOKE_INDEX",
    "EVAL_SHARDS",
    "MAX_REQUEUES",
    "EVAL_RECORD_INDEX",
    "EVAL_SHARD_INDEX",
    "BACKBONE_TRAINABLE_LAYERS",
    "BACKBONE_TURN_MICROBATCH",
    "ARCHITECTURE_MODE",
    "ATTRIBUTION_FUSION",
    "INPUT_VIEW",
    "MODEL",
    "TURN_POOLING",
    "RUN_ROOT",
    "PRECHECK_DIR",
    "OUTPUT",
    "TRAIN_RESUME",
    "EVAL_DATA",
    "EVAL_CHECKPOINT",
    "EVAL_OUTPUT",
    "EVAL_PROTOCOL",
    "SMOKE_OUTPUT_DIR",
    "SHARED_PREFLIGHT_DIR",
    "EXPECTED_ARCHITECTURE_MODE",
    "EXPECTED_ATTRIBUTION_FUSION",
    "EXPECTED_BACKBONE_TRAINABLE_LAYERS",
    "EXPECTED_INPUT_VIEW",
    "EXPECTED_TRAIN_VARIANT",
    "BACKBONE",
    "BACKBONE_REVISION",
    "MAX_TURNS",
    "MAX_TOKENS",
    "BATCH_SIZE",
    "GRAD_ACCUMULATION",
    "HEAD_LR",
    "BACKBONE_LR",
    "EPOCHS",
    "LOCALIZATION_RAMP_EPOCHS",
    "LENGTH_AUC_CEILING",
    "TRAIN_VARIANT",
    "TRAIN_PATH",
    "DEV_PATH",
    "PREPARED_DEV",
)


def isolated_launcher_environment(**updates):
    environment = dict(os.environ)
    for name in LAUNCHER_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    environment.update(updates)
    return environment


def write_internal_artifact(matrix, data_path, candidate, architecture, fusion, layers):
    diagnostic = matrix / "diagnostics" / candidate
    diagnostic.mkdir(parents=True, exist_ok=True)
    manifest_id = f"fixture-{candidate}"
    manifest = {
        "manifest_id": manifest_id,
        "stage": "internal_dev_causal_diagnostic",
        "development_only": True,
        "held_out_test_accessed": False,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "protocol": {"view": "retrospective"},
        "detector": {
            "input_view": "retrospective",
            "architecture_mode": architecture,
            "use_attribution_fusion": fusion,
            "config": {
                "backbone_trainable_layers": layers,
                "train_variant": "primary_plus_auxiliary",
            },
        },
    }
    report = {
        "manifest_id": manifest_id,
        "coverage": {"failures": [], "scored": 364, "total": 364},
        "development_only": True,
        "held_out_test_accessed": False,
    }
    (diagnostic / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (diagnostic / "report.json").write_text(json.dumps(report), encoding="utf-8")


class LauncherEmbargoTests(unittest.TestCase):
    def test_missing_top4_sibling_controls_reuse_completed_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            matrix = temporary / "matrix"
            shared = matrix / "shared"
            shared.mkdir(parents=True)
            data_path = shared / "internal-dev.jsonl"
            data_path.write_text(
                "{}\n" * 364, encoding="utf-8"
            )
            (shared / "internal-dev.jsonl.manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            base_candidates = (
                ("hierarchical_sibling_retro_frozen", "hierarchical_turn", False, 0),
                ("cross_token_sibling_retro_frozen", "cross_token", False, 0),
                ("cross_token_gated_retro_frozen", "cross_token", True, 0),
                ("cross_token_gated_retro_top4", "cross_token", True, 4),
            )
            for candidate in base_candidates:
                write_internal_artifact(matrix, data_path, *candidate)

            capture = temporary / "sbatch.args"
            state = temporary / "sbatch.state"
            stub = temporary / "sbatch"
            stub.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "for name in MAX_REQUEUES EVAL_RECORD_INDEX ARCHITECTURE_MODE ATTRIBUTION_FUSION MODEL TURN_POOLING SHARED_PREFLIGHT_DIR EXPECTED_ARCHITECTURE_MODE EXPECTED_ATTRIBUTION_FUSION EXPECTED_BACKBONE_TRAINABLE_LAYERS EXPECTED_INPUT_VIEW EXPECTED_TRAIN_VARIANT; do\n"
                "  [[ ! -v $name ]] || exit 91\n"
                "done\n"
                "printf '%s\\n' \"$*\" >> \"$MOCK_SBATCH_CAPTURE\"\n"
                "current=0\n"
                "[[ ! -f \"$MOCK_SBATCH_STATE\" ]] || current=$(<\"$MOCK_SBATCH_STATE\")\n"
                "current=$((current + 1))\n"
                "printf '%s\\n' \"$current\" > \"$MOCK_SBATCH_STATE\"\n"
                "printf '%s\\n' \"$current\"\n",
                encoding="utf-8",
            )
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
            environment = isolated_launcher_environment(
                PATH=f"{temporary}:{os.environ['PATH']}",
                MOCK_SBATCH_CAPTURE=str(capture),
                MOCK_SBATCH_STATE=str(state),
                MAX_REQUEUES="99",
                EVAL_RECORD_INDEX="999",
                ARCHITECTURE_MODE="ambient-poison",
                ATTRIBUTION_FUSION="ambient-poison",
                MODEL="ambient-poison",
                TURN_POOLING="ambient-poison",
                SHARED_PREFLIGHT_DIR="ambient-poison",
                EXPECTED_ARCHITECTURE_MODE="ambient-poison",
                EXPECTED_ATTRIBUTION_FUSION="ambient-poison",
                EXPECTED_BACKBONE_TRAINABLE_LAYERS="ambient-poison",
                EXPECTED_INPUT_VIEW="ambient-poison",
                EXPECTED_TRAIN_VARIANT="ambient-poison",
            )
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "run_missing_top4_sibling_controls.sh"),
                    str(matrix),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            submissions = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(submissions), 12)
            self.assertTrue(all("ambient-poison" not in row for row in submissions))
            self.assertIn("preflight_naacl_top4_sibling_controls.slurm", submissions[0])
            for smoke_id, train_id, gate_id, array_id, finalize_id in (
                (2, 4, 5, 6, 7),
                (3, 8, 9, 10, 11),
            ):
                smoke = submissions[smoke_id - 1]
                train = submissions[train_id - 1]
                gate = submissions[gate_id - 1]
                array = submissions[array_id - 1]
                finalize = submissions[finalize_id - 1]
                self.assertIn("BACKBONE_TRAINABLE_LAYERS=4", smoke)
                self.assertIn("ATTRIBUTION_FUSION=0", smoke)
                self.assertIn("--dependency=afterok:1", smoke)
                self.assertIn("--dependency=afterok:2:3", train)
                self.assertIn(f"--dependency=afterok:{train_id}", gate)
                self.assertIn("EVAL_RECORD_INDEX=136", gate)
                self.assertIn("MAX_REQUEUES=0", gate)
                self.assertIn("EXPECTED_BACKBONE_TRAINABLE_LAYERS=4", gate)
                self.assertIn("EXPECTED_ATTRIBUTION_FUSION=0", gate)
                self.assertIn(f"--dependency=afterok:{gate_id}", array)
                self.assertIn("--array=0-3%4", array)
                self.assertNotIn("EVAL_RECORD_INDEX", array)
                self.assertIn("EXPECTED_INPUT_VIEW=retrospective", array)
                self.assertIn(f"--dependency=afterok:{array_id}", finalize)
            self.assertIn("--dependency=afterok:7:11", submissions[-1])
            self.assertIn("compare_internal_dev_extended.slurm", submissions[-1])

    def test_missing_top4_control_launcher_resumes_without_repeating_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            matrix = temporary / "matrix"
            shared = matrix / "shared"
            shared.mkdir(parents=True)
            data_path = shared / "internal-dev.jsonl"
            data_path.write_text("{}\n" * 364, encoding="utf-8")
            (shared / "internal-dev.jsonl.manifest.json").write_text("{}\n", encoding="utf-8")
            marker = (
                shared
                / "preflight-top4-sibling-controls"
                / "primary_plus_auxiliary"
                / "retrospective"
                / "marker.json"
            )
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({
                "version": 1,
                "code_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "variant": "primary_plus_auxiliary",
                "input_view": "retrospective",
                "backbone": "answerdotai/ModernBERT-large",
                "backbone_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
                "max_turns": 64,
                "max_tokens": 8192,
                "held_out_test_accessed": False,
            }), encoding="utf-8")

            candidates = (
                ("hierarchical_sibling_retro_frozen", "hierarchical_turn", False, 0),
                ("cross_token_sibling_retro_frozen", "cross_token", False, 0),
                ("cross_token_gated_retro_frozen", "cross_token", True, 0),
                ("cross_token_gated_retro_top4", "cross_token", True, 4),
                ("cross_token_sibling_retro_top4", "cross_token", False, 4),
            )
            for candidate in candidates:
                write_internal_artifact(matrix, data_path, *candidate)
            checkpoint = (
                matrix
                / "training"
                / "hierarchical_sibling_retro_top4"
                / "checkpoints"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "last.pt").write_bytes(b"fixture")

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
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "run_missing_top4_sibling_controls.sh"),
                    str(matrix),
                ],
                cwd=ROOT,
                env=isolated_launcher_environment(
                    PATH=f"{temporary}:{os.environ['PATH']}",
                    MOCK_SBATCH_CAPTURE=str(capture),
                    MOCK_SBATCH_STATE=str(state),
                ),
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            submissions = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(submissions), 5)
            self.assertIn("TRAIN_RESUME=1", submissions[0])
            self.assertNotIn("smoke_naacl_window.slurm", "\n".join(submissions))
            self.assertIn("--dependency=afterok:1", submissions[1])
            self.assertIn("EVAL_RECORD_INDEX=136", submissions[1])
            self.assertIn("--dependency=afterok:2", submissions[2])
            self.assertIn("--dependency=afterok:3", submissions[3])
            self.assertIn("--dependency=afterok:4", submissions[4])

    def test_extended_comparison_requires_six_complete_internal_reports(self):
        text = (ROOT / "compare_internal_dev_extended.slurm").read_text(
            encoding="utf-8"
        )
        for candidate in (
            "hierarchical_sibling_retro_frozen",
            "cross_token_sibling_retro_frozen",
            "cross_token_gated_retro_frozen",
            "cross_token_gated_retro_top4",
            "hierarchical_sibling_retro_top4",
            "cross_token_sibling_retro_top4",
        ):
            self.assertIn(candidate, text)
        self.assertIn("python_source_sha256", text)
        self.assertIn('"scored": 364', text)
        self.assertIn("held_out_test_accessed", text)
        self.assertIn("internal_dev_comparison_six_candidates.json", text)
        self.assertIn("expected_axes", text)
        self.assertIn("controlled-axis mismatch", text)

    def test_top4_control_preflight_checks_base_evaluator_before_gpu_work(self):
        preflight = (ROOT / "preflight_naacl_top4_sibling_controls.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("source_identity", preflight)
        self.assertIn("evaluator source/runtime incompatibility", preflight)
        self.assertIn('"dataset_sha256": (manifest.get("dataset_sha256"), file_hash(data))', preflight)
        launcher = (ROOT / "run_missing_top4_sibling_controls.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("MATRIX_ROOT=$MATRIX_ROOT", launcher)
        self.assertIn('fresh_training_dependency=(--dependency="afterok:$smoke_dependency_ids")', launcher)

    def test_internal_diagnostic_can_enforce_checkpoint_axes(self):
        text = (ROOT / "eval_internal_dev.slurm").read_text(encoding="utf-8")
        for name in (
            "EXPECTED_ARCHITECTURE_MODE",
            "EXPECTED_ATTRIBUTION_FUSION",
            "EXPECTED_BACKBONE_TRAINABLE_LAYERS",
            "EXPECTED_INPUT_VIEW",
            "EXPECTED_TRAIN_VARIANT",
        ):
            self.assertIn(name, text)
        self.assertIn("supply all five EXPECTED_* diagnostic axes", text)
        self.assertIn("diagnostic manifest axes mismatch", text)

    def test_submitter_builds_record_gates_before_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            capture = temporary / "sbatch.args"
            state = temporary / "sbatch.state"
            stub = temporary / "sbatch"
            stub.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "for name in MAX_REQUEUES EVAL_RECORD_INDEX ARCHITECTURE_MODE ATTRIBUTION_FUSION MODEL TURN_POOLING; do\n"
                "  [[ ! -v $name ]] || exit 91\n"
                "done\n"
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
            environment = isolated_launcher_environment(
                PATH=f"{temporary}:{os.environ['PATH']}",
                MATRIX_ROOT=str(temporary / "matrix"),
                MOCK_SBATCH_CAPTURE=str(capture),
                MOCK_SBATCH_STATE=str(state),
                MAX_REQUEUES="99",
                EVAL_RECORD_INDEX="999",
                ARCHITECTURE_MODE="ambient-poison",
                ATTRIBUTION_FUSION="ambient-poison",
                MODEL="ambient-poison",
                TURN_POOLING="ambient-poison",
            )
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
            self.assertTrue(all("ambient-poison" not in row for row in submissions))
            self.assertTrue(all("EVAL_RECORD_INDEX=999" not in row for row in submissions))
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
                "for name in MAX_REQUEUES EVAL_RECORD_INDEX ARCHITECTURE_MODE ATTRIBUTION_FUSION MODEL TURN_POOLING; do\n"
                "  [[ ! -v $name ]] || exit 91\n"
                "done\n"
                "printf '%s\\n' \"$*\" >> \"$MOCK_SBATCH_CAPTURE\"\n"
                "current=0\n"
                "[[ ! -f \"$MOCK_SBATCH_STATE\" ]] || current=$(<\"$MOCK_SBATCH_STATE\")\n"
                "current=$((current + 1))\n"
                "printf '%s\\n' \"$current\" > \"$MOCK_SBATCH_STATE\"\n"
                "printf '%s\\n' \"$current\"\n",
                encoding="utf-8",
            )
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
            environment = isolated_launcher_environment(
                PATH=f"{temporary}:{os.environ['PATH']}",
                MOCK_SBATCH_CAPTURE=str(capture),
                MOCK_SBATCH_STATE=str(state),
                MAX_REQUEUES="99",
                EVAL_RECORD_INDEX="999",
                ARCHITECTURE_MODE="ambient-poison",
                ATTRIBUTION_FUSION="ambient-poison",
                MODEL="ambient-poison",
                TURN_POOLING="ambient-poison",
            )
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
            self.assertTrue(all("ambient-poison" not in row for row in submissions))
            self.assertTrue(all("EVAL_RECORD_INDEX=999" not in row for row in submissions))
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
