#!/bin/bash
#SBATCH --job-name=gl-dev-compare6
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/compare_internal_dev_extended_%j.out
#SBATCH --error=logs/compare_internal_dev_extended_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
: "${MATRIX_ROOT:?matrix root required}"

set +u
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

cd "$PROJECT_DIR"
OUTPUT="$MATRIX_ROOT/internal_dev_comparison_six_candidates.json"
[[ ! -e "$OUTPUT" ]] || { echo "ERROR: extended comparison exists: $OUTPUT"; exit 2; }

python - "$MATRIX_ROOT" "$OUTPUT" <<'PY'
import copy
import json
import sys
from pathlib import Path

from eval_platform.contract import write_json
from eval_platform.studies import compare_runs

root = Path(sys.argv[1])
output = Path(sys.argv[2])
names = (
    "hierarchical_sibling_retro_frozen",
    "cross_token_sibling_retro_frozen",
    "cross_token_gated_retro_frozen",
    "cross_token_gated_retro_top4",
    "hierarchical_sibling_retro_top4",
    "cross_token_sibling_retro_top4",
)
expected_axes = {
    "hierarchical_sibling_retro_frozen": ("hierarchical_turn", False, 0),
    "cross_token_sibling_retro_frozen": ("cross_token", False, 0),
    "cross_token_gated_retro_frozen": ("cross_token", True, 0),
    "cross_token_gated_retro_top4": ("cross_token", True, 4),
    "hierarchical_sibling_retro_top4": ("hierarchical_turn", False, 4),
    "cross_token_sibling_retro_top4": ("cross_token", False, 4),
}
runs = {}
code_identities = {}
for name in names:
    directory = root / "diagnostics" / name
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("coverage") != {"failures": [], "scored": 364, "total": 364}:
        raise SystemExit(f"ERROR: {name} does not have complete 364/364 coverage")
    if not report.get("development_only") or report.get("held_out_test_accessed") is not False:
        raise SystemExit(f"ERROR: {name} is not an internal-development-only report")
    detector = manifest.get("detector", {})
    config = detector.get("config", {})
    architecture, fusion, layers = expected_axes[name]
    expected = {
        "stage": (manifest.get("stage"), "internal_dev_causal_diagnostic"),
        "development_only": (manifest.get("development_only"), True),
        "held_out_test_accessed": (manifest.get("held_out_test_accessed"), False),
        "protocol.view": (manifest.get("protocol", {}).get("view"), "retrospective"),
        "detector.input_view": (detector.get("input_view"), "retrospective"),
        "detector.architecture_mode": (detector.get("architecture_mode"), architecture),
        "detector.use_attribution_fusion": (detector.get("use_attribution_fusion"), fusion),
        "detector.config.backbone_trainable_layers": (
            config.get("backbone_trainable_layers"), layers
        ),
        "detector.config.train_variant": (
            config.get("train_variant"), "primary_plus_auxiliary"
        ),
    }
    failures = [f"{key}: observed={observed!r}, expected={wanted!r}"
                for key, (observed, wanted) in expected.items() if observed != wanted]
    if failures:
        raise SystemExit(f"ERROR: controlled-axis mismatch for {name}: " + "; ".join(failures))
    runs[name] = {"manifest": manifest, "report": report}
    code_identities[name] = manifest["code"]

# These controls are added by launcher-only commits after the first four reports
# were frozen. Permit different Git SHAs only when the evaluated Python tree,
# interpreter, and package versions are byte-identical.
def evaluator_identity(code):
    return {key: value for key, value in code.items() if key != "git_sha"}

baseline = evaluator_identity(next(iter(code_identities.values())))
for name, code in code_identities.items():
    if evaluator_identity(code) != baseline:
        raise SystemExit(f"ERROR: evaluator source or runtime differs for {name}")

normalized = copy.deepcopy(runs)
for run in normalized.values():
    run["manifest"]["code"]["git_sha"] = "byte-identical-evaluator"
comparison = compare_runs(normalized)
comparison["comparison_contract"]["observed_git_shas"] = {
    name: code["git_sha"] for name, code in code_identities.items()
}
comparison["comparison_contract"]["git_sha_compatibility_rule"] = (
    "Git SHAs may differ only because launcher-only commits added the missing controls; "
    "python_source_sha256, interpreter, and package versions are identical."
)
write_json(output, comparison)
PY

echo "Extended internal dev matrix comparison: $OUTPUT"
echo "Candidates compared: 6"
echo "Held-out test accessed: NO"
