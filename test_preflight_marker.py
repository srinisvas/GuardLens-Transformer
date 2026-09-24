"""CPU-only tests for reusable matrix preflight attestations."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from guardlens.data.preflight_marker import create, verify


class PreflightMarkerTests(unittest.TestCase):
    def fixture(self, root, auc=0.6):
        root = Path(root)
        paths = {}
        for name in ("train", "dev", "freeze", "representation"):
            path = root / f"{name}.json"
            path.write_text(f"{name}\n", encoding="utf-8")
            paths[name] = path
        length = root / "length.json"
        length.write_text(json.dumps({
            "input_view": "pre_response",
            "dev": {"roc_auc": auc},
        }), encoding="utf-8")
        paths["length"] = length
        return paths

    def create_args(self, root, paths, auc_ceiling=0.65):
        return SimpleNamespace(
            train=str(paths["train"]),
            dev=str(paths["dev"]),
            code_sha="a" * 40,
            variant="primary",
            input_view="pre_response",
            freeze_report=str(paths["freeze"]),
            representation_report=str(paths["representation"]),
            length_report=str(paths["length"]),
            length_auc_ceiling=auc_ceiling,
            output=str(Path(root) / "marker.json"),
        )

    def test_marker_binds_code_data_reports_and_view(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.fixture(root)
            args = self.create_args(root, paths)
            create(args)
            verify(SimpleNamespace(
                marker=args.output,
                train=args.train,
                dev=args.dev,
                code_sha=args.code_sha,
                variant=args.variant,
                input_view=args.input_view,
            ))
            paths["train"].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "train SHA256 mismatch"):
                verify(SimpleNamespace(
                    marker=args.output,
                    train=args.train,
                    dev=args.dev,
                    code_sha=args.code_sha,
                    variant=args.variant,
                    input_view=args.input_view,
                ))

    def test_length_auc_ceiling_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.fixture(root, auc=0.651)
            with self.assertRaisesRegex(RuntimeError, "exceeds ceiling"):
                create(self.create_args(root, paths))


if __name__ == "__main__":
    unittest.main(verbosity=2)

