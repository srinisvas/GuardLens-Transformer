#!/usr/bin/env bash
# submit_eval_pipeline.sh
#
# Submits the full GuardLens evaluation pipeline with proper dependencies.
#
# Usage:
#   bash submit_eval_pipeline.sh [--dry-run]
#
# Jobs:
#   Stage 1 (setup)      → no dependencies
#   Stage 2 (metrics)    → after stage 1
#   Stage 3 (cross-data) → after stage 1
#   Stage 4 (paraphrase) → after stage 1
#   Stage 5 (collate)    → after stages 2, 3, 4

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "DRY RUN: will print sbatch commands without submitting"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs

submit() {
    local script="$1"
    local dep="${2:-}"

    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo "ERROR: $script not found in $SCRIPT_DIR"
        exit 1
    fi

    if [ "$DRY_RUN" = "1" ]; then
        if [ -n "$dep" ]; then
            echo "[DRY] sbatch --dependency=$dep --parsable $script"
        else
            echo "[DRY] sbatch --parsable $script"
        fi
        echo "99999"
        return
    fi

    if [ -n "$dep" ]; then
        sbatch --dependency="$dep" --parsable "$SCRIPT_DIR/$script"
    else
        sbatch --parsable "$SCRIPT_DIR/$script"
    fi
}

echo "========================================================"
echo "  GuardLens Evaluation Pipeline Submission"
echo "  Script dir: $SCRIPT_DIR"
echo "========================================================"

# Stage 1: Data setup
echo ""
echo "Submitting Stage 1 (data setup)..."
STAGE1=$(submit eval_stage1_setup.slurm "")
echo "  Stage 1 job: $STAGE1"

# Stage 2: External eval + subset analysis (depends on stage 1)
echo "Submitting Stage 2 (external eval + subset analysis)..."
STAGE2=$(submit eval_stage2_metrics.slurm "afterok:$STAGE1")
echo "  Stage 2 job: $STAGE2"

# Stage 3: Cross-dataset (depends on stage 1)
echo "Submitting Stage 3 (cross-dataset)..."
STAGE3=$(submit eval_stage3_cross_dataset.slurm "afterok:$STAGE1")
echo "  Stage 3 job: $STAGE3"

# Stage 4: Paraphrase (depends on stage 1)
echo "Submitting Stage 4 (paraphrase robustness)..."
STAGE4=$(submit eval_stage4_paraphrase.slurm "afterok:$STAGE1")
echo "  Stage 4 job: $STAGE4"

# Stage 5: Precision + Transfer (depends on stage 1, runs parallel with 2/3/4)
echo "Submitting Stage 6 (precision + cross-model transfer)..."
STAGE6=$(submit eval_stage6_precision_transfer.slurm "afterok:$STAGE1")
echo "  Stage 6 job: $STAGE6"

# Stage 5: Collate (depends on stages 2, 3, 4, 6)
echo "Submitting Stage 5 (result collation)..."
STAGE5=$(submit eval_stage5_collate.slurm "afterok:$STAGE2:$STAGE3:$STAGE4:$STAGE6")
echo "  Stage 5 job: $STAGE5"

echo ""
echo "========================================================"
echo "  All jobs submitted."
echo "  Monitor: squeue -u $USER"
echo "  Logs:    logs/eval_*"
echo ""
echo "  Job IDs:"
echo "    Stage 1 (setup):      $STAGE1"
echo "    Stage 2 (metrics):    $STAGE2"
echo "    Stage 3 (cross-data): $STAGE3"
echo "    Stage 4 (paraphrase): $STAGE4"
echo "    Stage 6 (precision):  $STAGE6"
echo "    Stage 5 (collate):    $STAGE5"
echo ""
echo "  Expected wall time: ~10h total (stages 2/3/4/6 run in parallel)"
echo "  Results: ~/work/results/dataset_gen/results/paper_tables.md"
echo "========================================================"
