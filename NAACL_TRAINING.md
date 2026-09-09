# GuardLens NAACL Training Handoff

This document is the execution contract between the repaired DataGen pipeline and the Transformer retraining/evaluation branch.

## Frozen input contract

Training begins only after Dataset A and Dataset B have independently passed their validity audits, have been merged, and have been split once using the consolidated group-aware splitter.

Expected split directory:

```text
$HOME/staging/dataset_gen_output/naacl_splits/
  train.jsonl
  dev.jsonl
  test.jsonl
  split_metadata.json
```

Dataset B standalone frontier hard-benign records are **not** part of these primary splits. They remain evaluation-only at:

```text
$HOME/staging/dataset_gen_output/naacl_frontier_benign_stress.jsonl
```

The frozen legacy benign stress corpus is also evaluation-only.

## Why the primary model window is 48 turns

The repaired Dataset A primary corpus contains 1,052 records and has a maximum of 48 realized user+assistant turns. Its class-conditional user-turn histograms are exactly matched. The old 32-turn model setting would truncate 314/1,052 primary Dataset A examples.

The NAACL branch therefore uses:

```text
max_turns = 48
max_tokens_per_turn = 192
microbatch = 2
gradient_accumulation = 8
effective batch = 16
```

GuardLens uses sinusoidal turn-position encoding, so extending the supported turn window does not introduce a new learned positional embedding table. All NAACL checkpoints are trained from scratch.

## Fail-closed guarantees

Before training, `audit_model_window.py` requires:

- every primary record fits wholly inside `max_turns`
- realized `turn_id` values equal their actual indices `0..N-1`
- supported pivots point to an in-window realized turn
- malicious records with unknown pivots use `pivot_supervision_ignore=true`
- no primary conversation label is trained on a silently truncated trajectory

The trainer additionally:

- aborts on CUDA OOM instead of skipping the affected batch
- steps a final partial gradient-accumulation group correctly
- computes scheduler length with `ceil(microbatches / accumulation)`

## Required execution order

### 1. CPU representation/shortcut preflight

`train_naacl.slurm` repeats these checks automatically, but they can be run independently:

```bash
python -m guardlens.evaluation.audit_model_window \
  --train $HOME/staging/dataset_gen_output/naacl_splits/train.jsonl \
  --dev $HOME/staging/dataset_gen_output/naacl_splits/dev.jsonl \
  --test $HOME/staging/dataset_gen_output/naacl_splits/test.jsonl \
  --max-turns 48 \
  --output $HOME/work/results/guardlens_naacl/preflight/model_window.json

python -m guardlens.evaluation.eval_length_probe_preflight \
  --train $HOME/staging/dataset_gen_output/naacl_splits/train.jsonl \
  --dev $HOME/staging/dataset_gen_output/naacl_splits/dev.jsonl \
  --output $HOME/work/results/guardlens_naacl/preflight/length_probe_dev.json
```

Do not train if the model-window audit fails or the train/dev length shortcut gate exceeds the locked threshold.

### 2. Mandatory one-GPU phase-3 memory smoke

```bash
sbatch smoke_naacl_window.slurm
```

This selects the longest malicious and longest benign training records and performs one phase-3 optimizer step, including attribution and counterfactual loss, using the locked 48-turn settings.

Proceed only if the log ends with:

```text
TRAINING WINDOW SMOKE PASSED
```

Any OOM is a hard failure. Reduce the microbatch and increase accumulation while keeping the effective batch fixed before retrying; never skip OOM batches.

### 3. Full retraining

```bash
sbatch train_naacl.slurm
```

The launcher retrains from scratch:

```text
guardlens
guardlens_no_fusion
guardlens_no_cf
turn_level
conversation_deberta
```

Default hierarchical input window is 48 turns for every model so all methods receive the same realized conversation prefix before model-specific token flattening/truncation behavior.

### 4. Final held-out evaluation

```bash
sbatch eval_naacl.slurm
```

The evaluation suite includes:

- held-out length-only shortcut report
- top-k evidence-turn hit rate
- leave-one-turn-out baseline
- deletion/intervention evidence evaluation
- supervision-tier breakdown
- utility grid
- NoCF intervention/utility ablation
- legacy benign stress FPR
- frontier-authored hard-benign stress FPR

The two benign stress sets are reported separately and never influence training or threshold selection.

## Scientific interpretation

The repaired experiment should be described as **fixed-user-trajectory counterfactual replay**. Later user turns are held fixed after an intervention. This supports evidence-localization claims under the observed/fixed trajectory but is not an adaptive causal simulation of how a user would have responded to a changed assistant message.
