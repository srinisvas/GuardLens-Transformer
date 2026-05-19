#!/bin/bash
#SBATCH --job-name=gl-phase3
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_phase3_%j.out
#SBATCH --error=logs/train_phase3_%j.err

# ============================================================
# Resume Phase 3 (CF fine-tuning) from Phase 2 checkpoint
#
# Phase 3 adds counterfactual loss with CF-tier oversampling.
# This should differentiate GuardLens from GuardLens-NoCF.
#
# Loads best_attribution.pt (phase 2 best) and runs 5 CF epochs.
# ============================================================

set -uo pipefail

RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT_DIR="$RESULTS_DIR/checkpoints/guardlens"
TRAIN_DATA="$HOME/staging/dataset_gen_output/splits/train.jsonl"
DEV_DATA="$HOME/staging/dataset_gen_output/splits/dev.jsonl"

CONDA_ENV="$HOME/work/conda_envs/guardlens_train"
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HOME/work/hf_models"
export TRANSFORMERS_CACHE="$HOME/work/hf_models/hub"
export HF_HUB_OFFLINE=1
export TMPDIR="$HOME/work/.tmp"
mkdir -p "$TMPDIR" "$CKPT_DIR" logs

cd "$SLURM_SUBMIT_DIR"

echo "========================================================"
echo "  Phase 3 CF Training (resume from Phase 2)"
echo "  Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify checkpoint
PHASE2_CKPT="$CKPT_DIR/best_attribution.pt"
if [ ! -f "$PHASE2_CKPT" ]; then
    PHASE2_CKPT="$CKPT_DIR/best.pt"
fi

if [ ! -f "$PHASE2_CKPT" ]; then
    echo "ERROR: No phase 2 checkpoint found"
    exit 1
fi

echo "Phase 2 checkpoint: $PHASE2_CKPT"
echo "Train data: $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") records)"
echo "Dev data: $DEV_DATA ($(wc -l < "$DEV_DATA") records)"

# Run phase 3 training
python3 -c "
import sys, os, json, torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, '.')

from guardlens.config import GuardLensConfig
from guardlens.data.dataset import GuardLensDataset, GuardLensCollator
from guardlens.models import MODEL_REGISTRY
from guardlens.training.loss import GuardLensLoss
from guardlens.training.trainer import train_epoch, evaluate
from guardlens.training.schedule import get_lambda_schedule

# Load phase 2 checkpoint
device = torch.device('cuda')
ckpt = torch.load('$PHASE2_CKPT', weights_only=False, map_location=device)
config = ckpt['config']

print(f'Loaded checkpoint: epoch {ckpt.get(\"epoch\", \"?\")}, phase {ckpt.get(\"phase\", \"?\")}')
print(f'Config: phase1={config.phase1_epochs}, phase2={config.phase2_epochs}, phase3={config.phase3_epochs}')

# Force phase 3 config
config.phase3_epochs = 5
config.max_epochs = config.phase1_epochs + config.phase2_epochs + config.phase3_epochs
print(f'Forced phase3_epochs=5, max_epochs={config.max_epochs}')

# Load data
def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

train_records = load_jsonl('$TRAIN_DATA')
dev_records = load_jsonl('$DEV_DATA')
print(f'Train: {len(train_records)}, Dev: {len(dev_records)}')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
collator = GuardLensCollator(tokenizer, config)

train_dataset = GuardLensDataset(train_records, config)
dev_dataset = GuardLensDataset(dev_records, config)

# CF oversampling for phase 3
sample_weights = []
for r in train_records:
    tier = r.get('supervision_tier', 'construction')
    if tier in ('cf_strong', 'cf_weak'):
        sample_weights.append(3.0)
    elif tier == 'llm_confirmed':
        sample_weights.append(1.5)
    else:
        sample_weights.append(1.0)

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size,
    sampler=sampler, collate_fn=collator,
    num_workers=4, pin_memory=True,
)
dev_loader = DataLoader(
    dev_dataset, batch_size=config.batch_size,
    collate_fn=collator, num_workers=4,
)

# Build model
model = MODEL_REGISTRY['guardlens'](config)
model.setup_backbone()
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)

loss_fn = GuardLensLoss(config)

# Optimizer — lower LR for fine-tuning
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.learning_rate * 0.1,
    weight_decay=config.weight_decay,
)

# Phase 3 epoch range
phase3_start = config.phase1_epochs + config.phase2_epochs
phase3_end = config.max_epochs

print(f'\\nRunning Phase 3: epochs {phase3_start}-{phase3_end-1}')
print(f'  LR: {config.learning_rate * 0.1:.6f} (0.1x base)')
print(f'  CF oversampling: cf_strong/weak 3x, llm_confirmed 1.5x')

best_attr_score = ckpt.get('best_attr_score', 0.0)
best_threshold = ckpt.get('threshold', 0.5)
print(f'  Phase 2 best attr F1: {best_attr_score:.4f}')

# Dev threshold tuning
def find_best_threshold(probs, labels):
    best_f1, best_t = 0, 0.5
    for t in [i/20 for i in range(1, 20)]:
        preds = [1 if p > t else 0 for p in probs]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

for epoch in range(phase3_start, phase3_end):
    metrics = train_epoch(
        model, train_loader, optimizer, None,
        loss_fn, config, epoch, device,
    )

    # Tune threshold
    model.eval()
    dev_probs, dev_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                turn_mask=batch['turn_mask'].to(device),
                role_ids=batch['role_ids'].to(device),
                compute_attribution=False,
            )
            dev_probs.extend(torch.sigmoid(out['cls_logits']).cpu().tolist())
            dev_labels.extend(batch['labels'].tolist())
    best_threshold = find_best_threshold(dev_probs, dev_labels)

    val_metrics = evaluate(model, dev_loader, loss_fn, config, device, threshold=best_threshold)

    print(
        f'Ep {epoch:3d} P3 | '
        f'loss {metrics[\"loss\"]:.4f} cf_loss {metrics.get(\"cf_loss\", 0):.4f} | '
        f'val F1 {val_metrics[\"f1\"]:.3f} attr F1 {val_metrics[\"attr_f1\"]:.3f} | '
        f'thr {best_threshold:.2f}'
    )

    # Save if attribution improved
    composite = 0.3 * val_metrics['f1'] + 0.7 * val_metrics['attr_f1']
    if composite > best_attr_score or epoch == phase3_start:
        best_attr_score = composite
        save_path = os.path.join('$CKPT_DIR', 'best_phase3.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'model_name': 'guardlens',
            'epoch': epoch,
            'phase': 3,
            'threshold': best_threshold,
            'best_attr_score': best_attr_score,
            'val_metrics': val_metrics,
        }, save_path)
        print(f'  saved best_phase3.pt (composite={best_attr_score:.4f}, '
              f'F1={val_metrics[\"f1\"]:.4f}, attrF1={val_metrics[\"attr_f1\"]:.4f})')

# Also save final
final_path = os.path.join('$CKPT_DIR', 'final_phase3.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'model_name': 'guardlens',
    'epoch': phase3_end - 1,
    'phase': 3,
    'threshold': best_threshold,
}, final_path)
print(f'\\nPhase 3 complete. Saved final_phase3.pt')
print(f'Best composite: {best_attr_score:.4f}')
"

echo ""
echo "========================================================"
echo "  Phase 3 Training Complete: $(date)"
echo "========================================================"

# Check outputs
for f in best_phase3.pt final_phase3.pt; do
    if [ -f "$CKPT_DIR/$f" ]; then
        echo "  OK: $f ($(stat -c%s "$CKPT_DIR/$f" | numfmt --to=iec))"
    else
        echo "  MISSING: $f"
    fi
done
