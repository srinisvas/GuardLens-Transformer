# GuardLens NAACL causal-localization training contract

This document is the authoritative handoff for the redesigned GuardLens
architecture and training module on branch:

    naacl-causal-localization-redesign

The branch starts from the complete `naacl-validity-repair` tip
`e6c432c37635ea7af34b390cbfa971173c9b5f43`. All validity-repair work is
therefore inherited. The redesign then replaces the EMNLP-era single-pivot,
gated-fusion, Phase-3 self-counterfactual and CF-oversampling recipe.

Do not infer readiness from this document alone. The module is code-reviewed,
but the current branch still requires the CPU contract tests, frozen-data
preflight and one-GPU smoke described below before full training.

## 1. Frozen data contract

Canonical frozen artifacts live at:

    $HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze

Primary A+B corpus:

- 2,454 conversations
- 1,227 benign
- 1,227 malicious
- train 1,720
- dev 367
- test 367

Record-level supervision tiers:

| Tier | Train | Dev | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| benign_validated | 861 | 183 | 183 | 1,227 |
| cf_strong | 445 | 98 | 89 | 632 |
| cf_weak | 3 | 1 | 2 | 6 |
| llm_confirmed | 411 | 85 | 93 | 589 |

Span annotations:

| Span target tier | Train | Dev | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| cf_strong | 820 | 176 | 151 | 1,147 |
| cf_weak | 16 | 4 | 5 | 25 |
| incidental | 227 | 45 | 55 | 327 |
| ignore | 3,004 | 627 | 650 | 4,281 |
| all annotations | 4,067 | 852 | 861 | 5,780 |

The 14 reviewed construction-language spans surviving into primary are a subset
of the ignored spans. They are never positive token supervision. Their split is
12 train, 2 dev and 0 test.

Construction-language visibility is class-symmetric in Dataset B primary:
174 benign and 174 malicious records contain reviewed construction language.

Primary frozen SHA-256 values:

    train  164a6327f54adedd8268d30c71d10ba56484f4984321505af33cbf3bb205ad1e
    dev    659394391b035fa5fa06607f304bd118872e44e61388704288cf27684034dbd0
    test   82771ea6ddef43a73f02de05e66d774f2c2f694bfb552929d28379c5a331cf45

The test hash is documented for provenance only. Training must not open the test
file.

### Detection-only auxiliary candidate

The optional candidate split freezes the primary split and adds 424 audited
auxiliary outcomes to training only:

- train 2,144 = 1,720 primary + 424 auxiliary
- dev 367 and byte-identical to primary dev
- test 367 and byte-identical to primary test
- auxiliary train detection labels are 271 safe and 153 unsafe
- auxiliary localization supervision is always disabled

Frozen auxiliary candidate SHA-256 values:

    train  3533198efdb55fe33087170c50d62d301969b841422e283c5ebdc71f10c7490f
    dev    659394391b035fa5fa06607f304bd118872e44e61388704288cf27684034dbd0

The canonical run is primary-only. The auxiliary candidate is an ablation.

## 2. Scientific target

The model now learns:

1. unsafe-trajectory detection
2. multi-label causal evidence-turn localization
3. context-conditioned causal span localization

"Causal localization" is operationally restricted to the intervention protocol:
a turn or span is positive when the controlled counterfactual intervention
changes subsequent unsafe behavior under the frozen replay/evaluation protocol.

The model does not claim globally necessary and sufficient "true causal
tokens".

## 3. Canonical architecture

The main model is:

    realized turn text
        |
        v
    frozen DeBERTa-v3-base per turn
        |
        +--> token representations -----------------------------+
        |                                                       |
        v                                                       |
    masked mean pool within turn                                |
        |                                                       |
        + role embedding + sinusoidal turn position             |
        |                                                       |
        v                                                       |
    2-layer turn-context Transformer                            |
        |                                                       |
        +--> learned conversation pooling --> detection head     |
        |                                                       |
        +--> independent evidence-turn sigmoid head             |
        |                                                       |
        +-------------------------------------------------------+
                                |
                                v
                 context-conditioned span head

The turn-context Transformer operates over T turn vectors rather than flattening
T x S tokens into one global self-attention sequence. Cross-turn attention is
therefore O(T^2), while DeBERTa continues to model within-turn token context.

The contextual span head combines each raw token representation with the
conversation-aware representation of the containing turn.

### Removed from the canonical model

The following EMNLP-era mechanisms are intentionally absent:

- single softmax pivot head
- pivot-kind classifier
- gated attribution-to-detection fusion
- Phase-3 self-counterfactual consistency loss
- CF record oversampling

Reasons:

- the frozen corpus now contains direct intervention-backed turn/span targets
- multi-evidence trajectories cannot be represented by one pivot class
- gated fusion couples the explanation to the prediction it later explains
- the old self-CF objective could create a model-induced faithfulness loop
- CF examples are no longer rare enough to justify x3 sampling
- x3 CF sampling would distort the balanced detection prior

Fusion and SelfCF may return later only as explicit ablations against the
canonical model.

## 4. Causal supervision contract

All training and future evaluation target construction must use
`guardlens.data.causal_targets`. No evaluator may independently redefine
"causal".

### Span targets

Positive:

    cf_strong + causal_type=causal + evidence_status=supported_strong
    cf_weak   + causal_type=causal + evidence_status=supported_weak

Negative:

    incidental + causal_type=incidental
      + evidence_status in {negative_control_supported, benign_negative}

Ignore:

- llm_confirmed spans
- construction-derived spans
- not_supported positive candidates
- unassessed spans
- not-assessable spans
- semantic_token_supervision_ignore=true
- all other unmatched or stale combinations

This strict status/tier agreement is deliberate. A stale `causal_type` alone
cannot create positive causal supervision.

### Evidence-turn targets

Positive:

- every member of the full `evidence_turn_ids` set
- supported B4 tested turn interventions
- repaired Dataset A supported anchor intervention
- evidence turns established through intervention-supported spans

Negative:

- explicitly tested `not_supported` malicious turns
- validated benign user turns

Ignore:

- assistant turns
- untested malicious turns
- not-assessable malicious turns
- all auxiliary detection-only records

Evidence-turn prediction is defined only over realized user turns. Assistant
turn logits are masked from the model output.

### Confidence weights

Strong intervention evidence receives weight 1.0.

Weak intervention evidence receives weight 0.70.

A weak evidence turn is not upgraded merely because another turn in the same
record made the record-level tier `cf_strong`. Turn confidence is derived from
the intervention status on that turn or from the strongest supported span on
that turn.

## 5. Detection loss contract

Primary records are behaviorally validated. Therefore:

    primary detection weight = 1.0

The old general `loss_weight` field is localization confidence and must not
down-weight trajectory detection.

Auxiliary records use their audited:

    detection_label
    detection_loss_weight

and have zero localization supervision.

Detection `pos_weight` is derived from weighted detection mass rather than raw
record count, so the optional 0.25-weight auxiliary outcomes do not silently
change class balancing.

## 6. Training objective and schedule

The canonical joint objective is:

    L = lambda_d * L_detection
      + lambda_t * L_turn
      + lambda_s * L_span

Phase 1:

    detection only
    epochs 0..4 by default

Phase 2:

    detection + evidence-turn localization + span localization
    epochs 5..19 by default

Detection remains fully weighted throughout.

Localization begins at 0.25 of its configured weight at the first joint epoch
and ramps linearly to full weight at the final joint epoch.

There is no Phase 3 in the canonical recipe.

There is no weighted CF sampler.

## 7. Representation and truncation contract

Silent truncation is forbidden.

The Dataset object itself fails if:

- a trajectory is empty
- realized turn IDs do not equal their actual indices
- a record exceeds `max_turns`

The collator:

- tokenizes with `truncation=False`
- requires offset mappings
- pads dynamically to the longest turn in the current batch
- fails if a realized turn exceeds `max_tokens_per_turn`
- never slices a trajectory to make it fit

Default values are:

    max_turns = 48
    max_tokens_per_turn = 512

The 512 value is a hard ceiling, not a fixed padding length and not yet an
empirical claim that the corpus fits. The representation audit must establish
that.

The model also checks the backbone's `max_position_embeddings`. The configured
turn-token ceiling may not exceed the backbone's actual position limit. If a
frozen turn is longer than the backbone supports, the solution is an explicit
chunking/windowing design. Increasing the number past the backbone limit is not
allowed.

Padded turns are not sent through DeBERTa. Only realized turns enter the
backbone.

## 8. Frozen artifact verification

Before training, run:

    python -m guardlens.data.verify_freeze \
      --report "$FREEZE/data_prep_freeze_report.json" \
      --train "$FREEZE/splits_primary/train.jsonl" \
      --dev "$FREEZE/splits_primary/dev.jsonl" \
      --variant primary

The verifier checks both SHA-256 and record counts against the final DataGen
freeze report.

A Git LFS pointer, stale copy or regenerated split therefore fails before
training.

## 9. Representation audit

Run:

    python -m guardlens.data.audit_representation \
      --train "$FREEZE/splits_primary/train.jsonl" \
      --dev "$FREEZE/splits_primary/dev.jsonl" \
      --backbone microsoft/deberta-v3-base \
      --max-turns 48 \
      --max-tokens 512 \
      --output /tmp/guardlens_representation_audit.json

The audit is train/dev only and reports:

- maximum realized turns
- p95 tokens per turn
- p99 tokens per turn
- maximum tokens per turn
- turns beyond the configured ceiling
- intervention-backed positive spans beyond the ceiling
- positive / negative / ignored span target counts
- positive / negative / ignored evidence-turn target counts
- backbone maximum position capacity

A positive causal span beyond the representation ceiling is a hard failure.

## 10. Shortcut gate

`train_naacl.slurm` retains the repaired train/dev-only length probe.

Training must stop if the locked dev length-only ROC AUC exceeds the configured
gate, currently 0.65.

The held-out test is not accessed by this preflight.

## 11. Checkpoint selection

All checkpoint selection is dev-only.

The trainer saves:

    best_detection.pt
        maximum dev detection F1

    best_localization.pt
        maximum mean(dev turn AUPRC, dev span AUPRC)

    best_joint.pt
        maximum mean(
            dev detection F1,
            dev evidence-turn AUPRC,
            dev span AUPRC
        )

`best.pt` resolves to `best_joint.pt` when it exists. Localization-only and
detection-only checkpoints remain diagnostics.

Early stopping in the joint phase follows the same joint selection score.

Every checkpoint records:

- architecture_version=causal_localization_v1
- exact train SHA-256
- exact dev SHA-256
- dev threshold
- dev metrics
- epoch and phase

This prevents a localization-only peak from becoming the canonical checkpoint
while materially sacrificing detection.

## 12. Held-out test embargo

The trainer has no test-path configuration.

`guardlens.train` accepts only frozen train and dev paths.

The old single-file internal split fallback was removed.

The training module never performs final test evaluation.

Final test evaluation begins only after:

1. architecture/training contract tests pass
2. frozen artifact verification passes
3. representation audit passes
4. one-GPU architecture smoke passes
5. the evaluation migration is complete
6. model/training decisions are frozen

## 13. Current launcher status

Canonical:

    train_naacl.slurm
    smoke_naacl_window.slurm

The canonical training launcher permits only `MODEL=guardlens` until baseline
migration is complete.

The smoke launcher is pinned to the frozen primary train/dev hashes before it
touches the GPU.

The following historical launchers are intentionally disabled on this branch:

    train_all.slurm
    train_phase3.slurm
    train_and_eval_review.slurm
    eval_naacl.slurm

`guardlens.evaluate` also fails closed for redesigned checkpoints until the
evaluation migration is complete.

This is intentional. Old EMNLP evaluators use stale pivot/construction semantics
and must not silently run against `causal_localization_v1`.

## 14. Baseline status

The old baseline source files remain for migration/reference, but baseline
training is not part of the canonical launcher yet.

The previous flat ConversationDeBERTa baseline has a 2,048-token flattening cap
and therefore does not currently provide a fair full-context architecture
comparison.

The baseline migration must provide matched input coverage before paper runs.

Planned fair comparison:

- pooled-turn detection baseline with identical per-turn DeBERTa coverage but no
  turn-context Transformer
- independent turn-level detector
- canonical GuardLens

Fusion and SelfCF return only as explicit ablations if implemented.

## 15. Evaluation migration contract

No existing evaluation capability may be silently dropped.

Each existing evaluation must be either:

- migrated to the new causal target contract
- retained unchanged only when its semantics remain valid
- explicitly labeled legacy/diagnostic
- converted into an ablation when the old component is no longer canonical

The migrated suite must preserve at least:

Detection:

- accuracy / precision / recall / F1
- AUROC / AUPRC
- benign FPR
- hard-benign FPR

Turn causal localization:

- evidence-turn AUPRC / AUROC
- top-k hit / precision / recall
- nearest-evidence distance
- attribution mass on supported evidence turns
- LOTO
- random floor

Span causal localization:

- precision / recall / F1 / AUPRC
- strong-only and weak-only breakdowns
- negative-control specificity
- score versus counterfactual-delta calibration

Faithfulness / intervention:

- DD@5/10/15/20
- Flip@5/10/15/20
- necessity
- sufficiency
- minimal trigger
- target-LLM intervention
- external-evaluator intervention

Specificity and robustness:

- boundary benign stress
- frontier hard-benign stress
- surface-risk FPR
- deconfounded / SR-neutralized / SR-injected tests
- attribution utility and lambda grid
- paraphrase robustness
- implicit/contextual versus explicit/lexical analysis
- cross-dataset transfer
- cross-model transfer
- MHJ external evaluation
- human benchmark
- length-only shortcut probe
- bootstrap confidence intervals
- per-supervision-tier analysis

Ablations:

- no turn-context encoder
- no span supervision
- no turn supervision
- optional +SelfCF
- optional +Fusion
- auxiliary detection-only on/off

Old `NoCF` and `NoFusion` must not be treated as canonical-reference models.
If reintroduced, they become `+SelfCF` and `+Fusion` comparisons against the
new canonical GuardLens.

## 16. Code-review findings fixed on this branch

The architecture/training review found and fixed the following defects or stale
assumptions:

1. top-level `guardlens` still imported removed GuardLensNoFusion/GuardLensNoCF
   classes, which would break every `python -m guardlens.*` command
2. the repaired model still collapsed multi-evidence supervision to one pivot
3. primary detection loss was being down-weighted by localization-confidence
   weights
4. Phase-3 self-CF could train on malicious examples without direct CF evidence
5. the old CF forward path did not use the same gated classifier pathway
6. CF oversampling would distort the balanced detection prior
7. 192-token per-turn truncation was still silent
8. Dataset slicing could silently drop turns
9. training automatically opened and evaluated the held-out test split
10. the trainer still allowed an internal re-split bypassing the frozen split
11. frozen artifacts were not cryptographically verified at training time
12. model-registry fallback could silently instantiate GuardLens for bad names
13. unknown span supervision could fail open as construction supervision
14. assistant turns were not masked from evidence-turn predictions
15. a record-level strong tier could incorrectly upgrade a weak evidence turn
16. token overlap could let an incidental weight upgrade a weak positive token
17. padded turns were unnecessarily sent through the backbone with zero masks
18. a configured token ceiling could exceed the backbone position capacity
19. localization-only checkpoint selection could sacrifice detection
20. legacy training/evaluation launchers remained runnable despite stale
    semantics
21. the generic evaluation entry point imported a trainer evaluation function
    that no longer exists after the redesign
22. the first rewritten SLURM launchers contained literal escaped shell
    expansions (\${...}); the smoke launcher also referenced REPORT_PATH and
    DEV_PATH before defining them
23. the evaluation package eagerly re-exported legacy causal-evaluation
    semantics, making accidental use easier on the redesign branch

All of the above are addressed in the current redesign branch.

## 17. Required validation before full training

First update the branch and confirm the current HEAD:

    cd "$HOME/projects/GuardLens-Transformer"
    git fetch origin
    git checkout naacl-causal-localization-redesign
    git pull
    git rev-parse HEAD

Run CPU contracts:

    python -m unittest -v \
      test_naacl_dataset_contract.py \
      test_auxiliary_loss_isolation.py \
      test_causal_localization_architecture.py \
      test_metadata_leakage.py

Then run the frozen SHA verifier and representation audit from Sections 8 and 9.

Then submit:

    sbatch smoke_naacl_window.slurm

Proceed to full training only when the smoke ends with:

    TRAINING ARCHITECTURE SMOKE PASSED

Finally:

    sbatch train_naacl.slurm

Do not submit a test/evaluation job yet. Evaluation migration is the next module.

## 18. Current readiness status

Static/manual code review:

    completed

Runtime test execution:

    still required on HPC

Architecture redesign:

    implemented

Training redesign:

    implemented

Frozen-data SHA verification code:

    implemented, execution still required on HPC

Representation/truncation audit:

    implemented, execution still required on HPC

CPU unit/contract suite:

    execution still required on HPC

GPU architecture smoke:

    execution still required

Full training:

    not yet authorized

Evaluation migration:

    not yet implemented

Held-out test access:

    prohibited until evaluation migration and training freeze
