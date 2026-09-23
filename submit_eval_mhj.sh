#!/bin/bash
set -euo pipefail
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
mkdir -p logs
array_job=$(sbatch --parsable --array="0-$((EVAL_SHARDS - 1))%$EVAL_SHARDS" --export=ALL eval_mhj.slurm)
final_job=$(sbatch --parsable --dependency="afterok:$array_job" --export=ALL eval_mhj_finalize.slurm)
printf 'Array job: %s (%s workers, 1 A100 each)\nFinalize job: %s (after all workers succeed)\n' "$array_job" "$EVAL_SHARDS" "$final_job"
