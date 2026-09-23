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
mkdir -p logs
array_job=$(sbatch --parsable --export=ALL eval_mhj.slurm)
final_job=$(sbatch --parsable --dependency="afterok:$array_job" --export=ALL eval_mhj_finalize.slurm)
printf 'Array job: %s (4 workers, 1 A100 each)\nFinalize job: %s (after all workers succeed)\n' "$array_job" "$final_job"
