#!/bin/bash
set -euo pipefail
if [[ "${EVAL_EXTERNAL_SIGNOFF:-}" != "APPROVED_AFTER_INTERNAL_SIGNOFF" ]]; then
    echo "ERROR: MHJ/external evaluation is embargoed until internal V4 signoff." >&2
    echo "After signoff, export EVAL_EXTERNAL_SIGNOFF=APPROVED_AFTER_INTERNAL_SIGNOFF." >&2
    exit 2
fi
cd "$(dirname "$0")"
for name in EVAL_DATA EVAL_DATA_SHA256 EVAL_CHECKPOINT EVAL_TRAIN_EXCLUDE EVAL_DEV_EXCLUDE \
            EVAL_SHIELD_CONFIG EVAL_CALIBRATION EVAL_CALIBRATION_POLICY EVAL_RUN_ROOT; do
    if [[ -z "${!name:-}" ]]; then
        echo "Missing $name" >&2
        exit 2
    fi
done
if [[ ! "$EVAL_RUN_ROOT" = /* ]]; then
    echo "EVAL_RUN_ROOT must be an absolute path" >&2
    exit 2
fi
export EVAL_SHARDS="${EVAL_SHARDS:-4}"
if [[ ! "$EVAL_SHARDS" =~ ^[1-4]$ ]]; then
    echo "EVAL_SHARDS must be an integer from 1 to 4" >&2
    exit 2
fi
eval_python="${EVAL_CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}/bin/python"
if [[ ! -x "$eval_python" ]]; then
    echo "Missing evaluation Python: $eval_python" >&2
    exit 2
fi
"$eval_python" - <<'PY'
import transformers
from eval_platform.runtime import require_transformers_dtype_support
require_transformers_dtype_support(transformers.__version__)
print(f"Evaluation transformers: {transformers.__version__}")
PY
"$eval_python" - "$EVAL_DATA" "$EVAL_DATA_SHA256" <<'PY'
import sys
from eval_platform.adapters import require_mhj_cohort
from eval_platform.contract import file_hash, read_jsonl

path, expected = sys.argv[1:]
if file_hash(path) != expected:
    raise ValueError("EVAL_DATA bytes differ from EVAL_DATA_SHA256")
print(f"Prepared MHJ records: {require_mhj_cohort(list(read_jsonl(path)))}")
PY
mkdir -p logs
array_job=$(sbatch --parsable --array="0-$((EVAL_SHARDS - 1))%$EVAL_SHARDS" --export=ALL eval_mhj.slurm)
final_job=$(sbatch --parsable --dependency="afterok:$array_job" --export=ALL eval_mhj_finalize.slurm)
printf 'Array job: %s (%s workers, 1 A100 each)\nFinalize job: %s (after all workers succeed)\n' "$array_job" "$EVAL_SHARDS" "$final_job"
