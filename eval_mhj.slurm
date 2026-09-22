#!/bin/bash
#SBATCH --job-name=gl-public-eval
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_public_%j.out
#SBATCH --error=logs/eval_public_%j.err

set -euo pipefail
: "${EVAL_DATA:?canonical dataset required}"
: "${EVAL_DATA_SHA256:?frozen dataset hash required}"
: "${EVAL_CHECKPOINT:?explicit joint checkpoint required}"
: "${EVAL_TRAIN_EXCLUDE:?prepared training exclusion required}"
: "${EVAL_DEV_EXCLUDE:?prepared dev exclusion required}"
: "${EVAL_SHIELD_CONFIG:?pinned ShieldGemma config required}"
: "${EVAL_CALIBRATION:?dev-only detector calibration required}"
: "${EVAL_CALIBRATION_POLICY:?frozen detector operating point required}"
: "${EVAL_RUN_ROOT:?output directory required}"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${EVAL_CONDA_ENV:-$HOME/work/conda_envs/guardlens_train}"
export HF_HOME="${HF_HOME:-$HOME/work/hf_models}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
python -u -m eval_platform run \
  --data "$EVAL_DATA" --data-sha256 "$EVAL_DATA_SHA256" \
  --checkpoint "$EVAL_CHECKPOINT" \
  --calibration "$EVAL_CALIBRATION" \
  --calibration-policy "$EVAL_CALIBRATION_POLICY" \
  --exclude "$EVAL_TRAIN_EXCLUDE" "$EVAL_DEV_EXCLUDE" \
  --protocol "${EVAL_PROTOCOL:-configs/eval_protocol.json}" \
  --guard both --shield-config "$EVAL_SHIELD_CONFIG" \
  --output "$EVAL_RUN_ROOT"
