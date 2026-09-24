#!/bin/bash
#SBATCH --job-name=gl-public-eval
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_public_%A_%a.out
#SBATCH --error=logs/eval_public_%A_%a.err

set -euo pipefail
if [[ "${EVAL_EXTERNAL_SIGNOFF:-}" != "APPROVED_AFTER_INTERNAL_SIGNOFF" ]]; then
  echo "ERROR: MHJ/external evaluation is embargoed until internal V4 signoff." >&2
  exit 2
fi
: "${EVAL_DATA:?canonical dataset required}"
: "${EVAL_DATA_SHA256:?frozen dataset hash required}"
: "${EVAL_CHECKPOINT:?explicit signed-off checkpoint required}"
: "${EVAL_TRAIN_EXCLUDE:?prepared training exclusion required}"
: "${EVAL_DEV_EXCLUDE:?prepared dev exclusion required}"
: "${EVAL_SHIELD_CONFIG:?pinned ShieldGemma config required}"
: "${EVAL_CALIBRATION:?dev-only detector calibration required}"
: "${EVAL_CALIBRATION_POLICY:?frozen detector operating point required}"
: "${EVAL_RUN_ROOT:?output directory required}"
: "${EVAL_SHARDS:?submit with submit_eval_mhj.sh}"
: "${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${EVAL_CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
export HF_HOME="${HF_HOME:-$HOME/work/hf_models}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
python - "$EVAL_DATA" "$EVAL_DATA_SHA256" <<'PY'
import sys
from eval_platform.adapters import require_mhj_cohort
from eval_platform.contract import file_hash, read_jsonl

path, expected = sys.argv[1:]
if file_hash(path) != expected:
    raise ValueError("EVAL_DATA bytes differ from EVAL_DATA_SHA256")
print(f"Prepared MHJ records: {require_mhj_cohort(list(read_jsonl(path)))}")
PY
mkdir -p "$EVAL_RUN_ROOT"
exec 9>"$EVAL_RUN_ROOT/.shard-${SLURM_ARRAY_TASK_ID}.lock"
flock -n 9 || { echo "Shard $SLURM_ARRAY_TASK_ID is already running" >&2; exit 1; }
python -u -m eval_platform run \
  --data "$EVAL_DATA" --data-sha256 "$EVAL_DATA_SHA256" \
  --checkpoint "$EVAL_CHECKPOINT" \
  --calibration "$EVAL_CALIBRATION" \
  --calibration-policy "$EVAL_CALIBRATION_POLICY" \
  --exclude "$EVAL_TRAIN_EXCLUDE" "$EVAL_DEV_EXCLUDE" \
  --protocol "${EVAL_PROTOCOL:-configs/eval_protocol.json}" \
  --guard both --shield-config "$EVAL_SHIELD_CONFIG" \
  --output "$EVAL_RUN_ROOT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "$EVAL_SHARDS"
