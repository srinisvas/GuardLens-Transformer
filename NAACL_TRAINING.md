# GuardLens NAACL causal-localization training contract

## 0. Environment setup

The canonical environment prefix is:

    $HOME/work/conda_envs/guardlens_train

Repair/update an existing environment:

    bash setup_guardlens_env.sh

For a clean rebuild:

    bash setup_guardlens_env.sh --recreate

The setup script installs the complete canonical training/preflight dependency
set from `requirements.txt`, verifies dependency consistency with `pip check`,
requires a CUDA-enabled PyTorch build, verifies the ModernBERT-large fast
tokenizer and offset mappings, verifies `max_position_embeddings=8192` and
`hidden_size=1024`, caches the canonical backbone snapshot, and imports the
redesigned GuardLens package/training schedule.

The canonical backbone is `answerdotai/ModernBERT-large`, pinned to Hugging
Face revision:

    45bb4654a4d5aaff24dd11d4781fa46d39bf8c13

Transformers 4.56.2+ and below 5 is required. The setup script
downloads that exact revision rather than following a mutable `main` ref.

Heavy evaluation backends such as vLLM and bitsandbytes remain optional and are
not installed by the canonical training environment until their corresponding
evaluation jobs are migrated.

This document is the authoritative handoff for the redesigned GuardLens
architecture and training module on branch:

    naacl-causal-localization-redesign

The branch consumes the final restored-A freeze produced by DataGen branch
`naacl-validity-repair`. Model and baseline choices are optimized for the NAACL
paper itself. Historical EMNLP architecture compatibility is not a scientific
constraint.

Do not infer readiness from this document alone. The module is code-reviewed,
but the current branch still requires the CPU contract tests, frozen-data
preflight and one-GPU smoke described below before full training.

## 1. Frozen data contract

Canonical frozen artifacts live at:

    $HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a

Primary A+B corpus:

- 2,434 conversations
- 1,217 benign
- 1,217 malicious
- train 1,706
- dev 364
- test 364

Record-level supervision tiers:

| Tier | Train | Dev | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| benign_validated | 853 | 182 | 182 | 1,217 |
| cf_strong | 446 | 91 | 95 | 632 |
| cf_weak | 2 | 3 | 1 | 6 |
| llm_confirmed | 405 | 88 | 86 | 579 |

Primary train supervision by source:

| Source | benign_validated | cf_strong | cf_weak | llm_confirmed | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (`legacy_restored_primary`) | 363 | 1 | 2 | 360 | 726 |
| B (`frontier_authored_v3`) | 490 | 445 | 0 | 45 | 980 |

Strong counterfactual localization supervision therefore comes almost entirely
from B, while restored A contributes primarily LLM-confirmed primary records and
its 721 detection auxiliaries. Evaluation and reporting must remain
source-stratified.

Exact turn/span target counts are generated from this freeze by the mandatory
representation audit. Construction-derived and LLM-confirmed spans remain
ignored unless the frozen span carries the exact intervention-backed tier,
causal type, and evidence-status combination required by the target contract.

Primary frozen SHA-256 values:

    train  5ccad79f1c3f0f0cc998f6ee2a4b04943e18938319ce5663e6d3d725e27de0d2
    dev    16f500cd6a67806f834aa761e388116a2ba03b1b136232cd2ec504aa9261833d
    test   6085baf91f129f5ed2acfc6e1c6b14270f9750f3c8aaa7c269307d9ea157bb0a

The test hash is documented for provenance only. Training must not open the test
file.

### Canonical detection-only auxiliary population

The canonical split freezes the primary partitions and adds 1,146 audited
auxiliary outcomes to training only:

- train 2,852 = 1,706 primary + 721 A auxiliary + 425 B auxiliary
- dev 364 and byte-identical to primary dev
- test 364 and byte-identical to primary test
- final train detection labels are 1,845 benign and 1,007 unsafe
- A auxiliary detection weight is 1.0
- B auxiliary detection weight is 0.25
- auxiliary localization supervision is always disabled

Frozen canonical training SHA-256 values:

    train  2927d73c9c471ca9835159d6162e62fd8bb75f4a501b56536bfe1f41dffe2b80
    dev    16f500cd6a67806f834aa761e388116a2ba03b1b136232cd2ec504aa9261833d

The canonical run uses `primary_plus_auxiliary`. Primary-only training is the
auxiliary ablation.

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

## 3. V3 architecture diagnostic

The completed V2 internal evaluation is not the final model. V3 first aligns
training, dev calibration and evaluation to the `pre_response` view, then runs a
controlled four-model diagnostic matrix before any held-out or external access.

The shared model skeleton is:

    realized turn text
        |
        v
    ModernBERT-large per turn (native 8,192-token context)
        |
        +--> token representations -----------------------------+
        |                                                       |
        v                                                       |
    masked mean or learned attention pool within turn           |
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
therefore O(T^2), while ModernBERT models the complete within-turn token
context natively.

The contextual span head combines each raw token representation with the
conversation-aware representation of the containing turn. Attention pooling
and selective fine-tuning of the final four ModernBERT layers are diagnostic
axes. They are not paper claims until the internal dev comparison is reviewed.

### Backbone choice

The canonical backbone is `answerdotai/ModernBERT-large`:

- encoder-only and bidirectional, which matches token/span localization
- native 8,192-token context, covering the observed train/dev turn lengths
  without truncation or within-turn chunking
- 1,024-dimensional token states for the contextual span head
- local/global alternating attention suitable for long inputs
- fast-tokenizer offset mappings required by the span supervision contract
- BF16 with PyTorch SDPA on the A100 path

The frozen train/dev representation audit showed that a 512-token encoder is
structurally mismatched to this corpus: p95 is about 700 tokens, p99 about 1.15K,
and observed maxima are 3,491 train / 2,113 dev. The model therefore preserves
each realized turn as one native encoder sequence instead of splitting it into
overlapping chunks. This keeps cross-token dependencies within the turn intact.

Long-context implementation must also avoid turning dynamic padding into hidden
quadratic work. The collator remains batch-padded for a simple model interface,
but the backbone does not encode every realized turn at the batch-wide
maximum length. Realized turns are sorted by their true token length and passed
through ModernBERT in length-local microbatches. Frozen candidates default to
eight turns per microbatch. Selectively fine-tuned candidates use one turn per
microbatch plus gradient checkpointing. Only each
microbatch's local maximum length enters the backbone. The dense tensor is
reconstructed afterward for the hierarchical heads.

All V3 candidates train on the exact `pre_response` view. The final observed
assistant response is never available to training, calibration or diagnostics.
Earlier assistant history remains visible.

Assistant/padded turns are masked from evidence-turn output, and assistant or
padding tokens are also masked from causal span output. Span supervision is
therefore user-turn-only end to end: the Dataset emits no assistant-turn span
targets, the training coverage gate counts only user-turn span targets, and the
representation audit fails closed if frozen data contains any target-bearing
assistant span annotation.

### Excluded from the canonical model

The following mechanisms are intentionally absent:

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

- supported B4 tested turn interventions
- repaired Dataset A supported anchor intervention
- evidence turns established through intervention-supported spans

The full `evidence_turn_ids` set must exactly match the turns established by
those local intervention sources. A declared ID without local support, or a
locally supported turn omitted from the declared set, fails closed.
`semantic_token_supervision_ignore=true` masks the span-token target only. A
valid supported span intervention may still establish its containing turn as a
turn-level target.

`supervision_tier` summarizes record/span evidence quality. Therefore an
`llm_confirmed` record may still carry positive evidence-turn supervision when
its whole-turn intervention is supported. Its spans remain ignored unless they
independently satisfy the strict counterfactual span contract above.

`pivot_turn_id` never creates a target. It is retained only as legacy provenance
and for diagnostic evaluation slices. A counterfactual-tier record with no
intervention-backed positive evidence-turn target fails closed.

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
that turn. `evidence_turn_ids` is a reconciled index, not an independent source
of supervision. If an ID has neither a supported turn-local intervention nor a
supported span, target construction fails rather than assigning a fallback
weight.

If a turn is simultaneously listed in `evidence_turn_ids` and has an explicit
`not_supported` whole-turn intervention, the record fails closed unless an
independently supported positive span exists on that same turn. In that case,
the supported span evidence may override the whole-turn negative result using
its own strong/weak confidence. When a previously negative or unset turn is
flipped to positive, its positive confidence is taken only from the positive
evidence source; the stale negative confidence is discarded. Confidence values
are combined with `max` only when the turn was already positive from another
positive source.

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
record count, so the 0.25-weight B auxiliary outcomes do not silently
change class balancing.

Confidence weights are absolute. Weighted BCE is divided by the number of
eligible targets, not by the sum of their confidence weights. Therefore, a
microbatch containing only 0.25-weight B auxiliaries still contributes exactly
one quarter of the equivalent unit-weight loss. The same rule preserves the
0.70 scale for weak turn/span evidence.

Evidence-turn BCE also balances labeled positive versus labeled negative turn
mass after applying strong/weak confidence weights. Untested turns do not enter
that balance. Span BCE uses only explicit positive/negative span targets and
their intervention-confidence weights; there is no additional span class
reweighting in the canonical recipe.

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

Localization begins at 0.25 of its configured weight at the first joint epoch.
With the canonical `localization_ramp_epochs=5`, it reaches full weight at
epoch 9 and then remains at full weight for epochs 9 through 19.

This early plateau is deliberate. The optimizer still uses OneCycleLR, whose
cosine decay reduces the learning rate in the latter part of training. Letting
the localization ramp reach 1.0 only at epoch 19 would make the advertised full
localization weight largely cosmetic because the LR is already near its floor.
The five-epoch ramp gives the turn/span heads substantial full-weight training
while the LR is still materially high, without changing the detection optimizer
schedule.

There is no Phase 3 in the canonical recipe.

There is no weighted CF sampler.

The canonical schedule runs all configured epochs by default. Early stopping is
disabled by default; `patience > 0` exists only as an explicit non-canonical
override. Training logs report both the current LR and localization lambda so
their interaction is directly auditable.

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

    max_turns = 64
    max_tokens_per_turn = 8192

The 8,192-token value is the native ModernBERT-large position capacity. It is a
hard architectural ceiling, not a padding length. Dynamic padding still uses
the longest realized turn in the current batch.

The train/dev audit observed a maximum of 3,491 tokens, so the canonical
backbone covers the complete observed turn distribution without chunking.
If a future frozen/test turn exceeds 8,192 tokens, evaluation must fail closed;
the response is not silent truncation.

Padded turns are not sent through ModernBERT. Only realized turns enter the
backbone.

## 8. Frozen artifact verification

Before training, run:

    python -m guardlens.data.verify_freeze \
      --report "$FREEZE/data_prep_freeze_report.json" \
      --train "$FREEZE/splits_primary_plus_train_auxiliary/train.jsonl" \
      --dev "$FREEZE/splits_primary_plus_train_auxiliary/dev.jsonl" \
      --variant primary_plus_auxiliary

The verifier checks both SHA-256 and record counts against the final DataGen
freeze report.

For `primary_plus_auxiliary`, the restored-A builder intentionally retains the
stable report fields `artifact_sha256.auxiliary_candidate_train`,
`artifact_sha256.auxiliary_candidate_dev`,
`auxiliary_candidate.train_records`, and
`auxiliary_candidate.dev_records`. The verifier requires all four fields and
fails with the missing field name if the report schema is incomplete.

A Git LFS pointer, stale copy or regenerated split therefore fails before
training.

## 9. Representation audit

Run:

    python -m guardlens.data.audit_representation \
      --train "$FREEZE/splits_primary_plus_train_auxiliary/train.jsonl" \
      --dev "$FREEZE/splits_primary_plus_train_auxiliary/dev.jsonl" \
      --backbone answerdotai/ModernBERT-large \
      --max-turns 64 \
      --max-tokens 8192 \
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

Structural audit failures such as empty trajectories, turn-ID mismatches,
unsupported roles, or trajectories exceeding `max_turns` are accumulated into
the failed audit report rather than escaping as a traceback. The audit therefore
remains fail-closed while still producing its intended diagnostic JSON.

The trainer independently fails if either train or dev lacks both detection
classes, both positive/negative evidence-turn targets, or both
positive/negative span targets. This prevents a target-extraction regression
from silently turning a joint run into a one-class or detection-only run.

## 10. Shortcut diagnostic

`train_naacl.slurm` retains the repaired train/dev-only length probe.

The length-only probe is computed on the exact configured model view. Training
fails closed if dev ROC AUC exceeds the frozen 0.650 ceiling. The ceiling must
not be raised to make a run pass. Repair the data if the pre-response view fails.
Train/dev direction reversal and all source-stratified detection results remain
visible in the reported diagnostics.

The held-out test is not accessed by this preflight.

## 11. Checkpoint selection

All checkpoint selection is dev-only.

The trainer saves:

    best_detection.pt
        maximum Dataset-B dev detection AUPRC

    best_localization.pt
        maximum mean(dev turn AUPRC, dev span AUPRC)

    best_joint.pt
        maximum mean(
            Dataset-B dev detection AUPRC,
            dev evidence-turn AUPRC,
            dev span AUPRC
        )

`best.pt` resolves to `best_joint.pt` when it exists. Localization-only and
detection-only checkpoints remain diagnostics.

The paper's primary detection claim must be based on Dataset B, matching the
selection population. Dataset A detection remains a separately reported
paired-generation diagnostic. Combined and macro A/B results are supplemental
and must not be substituted for the B-primary claim.

The canonical run does not early-stop; all 20 default epochs execute and the
joint dev score selects the checkpoint afterward. If early stopping is
explicitly enabled as a non-canonical override, it follows the same joint
selection score.

Every checkpoint records:

- architecture_version=causal_localization_v3
- training_contract_version=restored_a_pre_response_v3
- exact training-code Git SHA
- exact ModernBERT model revision through the stored config
- exact train SHA-256
- exact dev SHA-256
- Python / PyTorch / Transformers / CUDA runtime versions
- dev threshold
- dev metrics
- combined, A-only, B-only, and macro A/B detection diagnostics
- canonical_detection_source_family=B
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
    preflight_naacl_matrix.slurm
    eval_internal_dev.slurm
    compare_internal_dev_matrix.slurm
    submit_train_naacl_diagnostic.sh

The canonical training launcher permits only `MODEL=guardlens` until baseline
migration is complete.

Both canonical launchers require the checked-out branch to be
`naacl-causal-localization-redesign`, reject tracked uncommitted changes, and
export the exact Git commit SHA into the training process. The training
DataLoader also uses an explicit seed-bound `torch.Generator`, pinning shuffled
batch order to `config.seed`. This does not claim full CUDA bitwise
determinism; it removes batch-order nondeterminism.

The single-run launchers default to `primary_plus_auxiliary`. The trainer independently
validates that canonical training contains both A and B auxiliary sources and
that dev remains primary-only with both A and B source families. Every full run
uses a new run-specific output directory and refuses to reuse an existing path.
CUDA requests fail closed instead of falling back to CPU.

The matrix runs one shared CPU preflight before reserving any GPU. It verifies
both frozen training variants, audits the pre-response representation, enforces
the length-only AUC ceiling and prepares only internal dev. Its content-addressed
marker lets the GPU jobs verify and reuse those exact results instead of idling
four GPUs during duplicate audits.

The smoke launcher is pinned to the train/dev hashes and exact
ModernBERT revision before it touches the GPU. It executes a localization batch,
an A+B auxiliary-only batch when the variant includes auxiliaries, and a
worst-token-footprint batch chosen across the full training population. All
executed batches report peak CUDA allocation.

The four internal candidates are:

| Candidate | Pooling | Trainable backbone layers | Training population |
| --- | --- | ---: | --- |
| `mean_frozen_aux` | mean | 0 | primary + auxiliary |
| `attention_frozen_aux` | attention | 0 | primary + auxiliary |
| `attention_top4_aux` | attention | 4 | primary + auxiliary |
| `attention_top4_primary` | attention | 4 | primary only |

Selective fine-tuning uses a separate backbone learning rate of 2e-5 while the
hierarchical heads use 2e-4. All four candidates run concurrently on one A100
each. After training, four self-guard internal-dev causal diagnostics run
concurrently and report effects plus GuardLens-minus-baseline paired differences
by dataset, corpus source, family, pivot kind and supervision tier. The comparison job starts
only after all four reports complete.

The following historical launchers are intentionally disabled on this branch:

    train_all.slurm
    train_phase3.slurm
    train_and_eval_review.slurm
    eval_naacl.slurm

`guardlens.evaluate` now delegates to the versioned `eval_platform` CLI.
See `NAACL_EVALUATION.md` for review coverage, commands and experiment gates.

This is intentional. The pre-redesign evaluators use stale target and
representation semantics and must not silently run against
the current `causal_localization_v3` contract.

## 14. Baseline status

NAACL baselines are defined against the current scientific model, not against
historical EMNLP checkpoints.

Every paper baseline must use the same native full-turn ModernBERT coverage so a
result cannot be explained by one model seeing more of the input.

Planned matched-coverage baselines:

- ModernBERT pooled-turn detector: encode each full turn, pool turns directly,
  no turn-context Transformer
- ModernBERT independent-turn detector: classify turns independently and
  aggregate conversation risk
- canonical GuardLens: full-turn ModernBERT + turn-context Transformer +
  evidence-turn/span supervision

Primary ablations:

- no turn-context encoder
- no span supervision
- no turn supervision
- frozen versus selectively fine-tuned backbone if dev evidence justifies it
- auxiliary detection-only on/off
- optional +SelfCF
- optional +Fusion

There is no requirement to reproduce the old DeBERTa architecture in the NAACL
paper.

## 15. Evaluation migration contract

Implementation status: the unified platform now supports the current checkpoint,
public dataset preparation, evidence agreement, textual interventions, independent
guard transfer, paired target replay and review-facing reports. The detailed
coverage matrix and remaining empirical requirements are in `NAACL_EVALUATION.md`.
The list below remains the scientific scope, not a claim that every experiment
has already been run or every historical metric should retain its original name.

The NAACL evaluation suite must preserve every scientifically useful capability,
but it does not need to preserve historical checkpoint compatibility.

Each prior evaluation must be either:

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
- source-family-stratified and macro A/B detection results
- length-stratified or length-matched detection analysis alongside the probe
- bootstrap confidence intervals
- per-supervision-tier analysis

The final report must not rely on an unadjusted combined A/B detection metric
to dismiss the known Dataset A length artifact. It must report Dataset B as the
primary detection population and expose the length-controlled result explicitly.

Ablations:

- no turn-context encoder
- no span supervision
- no turn supervision
- optional +SelfCF
- optional +Fusion
- auxiliary detection-only on/off

`NoCF` and `NoFusion` are not reference models for NAACL. If those ideas are
reintroduced, they appear only as `+SelfCF` and `+Fusion` ablations against
the native-long-context canonical GuardLens.

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
24. the old default early-stopping patience could terminate Phase 2 before the
    scheduled localization ramp reached full weight
25. the trainer did not fail closed if strict target extraction accidentally
    produced one-class or empty turn/span localization supervision
26. training checkpoints recorded frozen data hashes but not the exact model
    code revision used to produce them
27. the localization lambda ramp reached full weight only when OneCycleLR had
    already annealed close to zero, materially weakening the intended
    localization training signal
28. an `evidence_turn_ids` member without turn-local status/span evidence could
    inherit weight 1.0 solely from a record-level `cf_strong` tier
29. overlapping strong/weak positive spans could become order-dependent at the
    character level and lose the strong 1.0 confidence weight
30. representation overflow could raise inside `GuardLensDataset.__getitem__`
    before the representation audit could emit its structured failed report
31. shuffled training batches relied on the global Torch RNG rather than an
    explicit seed-bound DataLoader generator
32. contradictory turn supervision could allow an `evidence_turn_ids` member to
    overwrite an explicit `not_supported` turn intervention even when no
    independent positive span evidence existed on that turn
33. when a supported span legitimately overrode a `not_supported` whole-turn
    result, the previous negative confidence weight 1.0 could leak through a
    final `max()` and silently upgrade weak 0.70 positive evidence to full
    confidence
34. the environment setup omitted the mandatory SentencePiece dependency and
    duplicated only a subset of `requirements.txt`; unquoted package version
    constraints in shell were also unsafe
35. the repaired DeBERTa environment still lacked protobuf, which exposed an
    incomplete tokenizer setup
36. the 512-token backbone itself was structurally mismatched to the frozen
    corpus: roughly 9% of train/dev turns exceeded 512, with train/dev maxima of
    3,491 / 2,113 tokens. The canonical architecture now uses native 8K
    ModernBERT-large full-turn encoding instead of truncation or chunking.
37. the first ModernBERT switch still followed the mutable Hugging Face `main`
    revision, so data/code hashes alone were insufficient for reproducibility;
    the exact backbone revision is now pinned.
38. batch-global turn padding would have sent every realized turn through
    ModernBERT at the single longest turn length in the batch, creating severe
    long-context compute inflation; frozen-backbone encoding is now
    length-sorted and microbatched.
39. ModernBERT would otherwise load and execute in FP32 despite A100 BF16/SDPA
    support; the frozen canonical backbone now runs BF16 with SDPA while
    downstream trainable heads receive FP32 hidden states.
40. causal span logits on assistant/padding tokens were unconstrained even
    though the localization contract is user-turn-only; those outputs are now
    hard masked.
41. the original GPU smoke selected worst cases using character length, which
    does not reliably stress tokenizer-level long-context memory; the smoke now
    runs a separate batch selected by actual token footprint.
42. scientific checkpoints recorded code/data provenance but not exact runtime
    library versions; Python, PyTorch, Transformers and CUDA runtime versions
    are now stored.
43. span logits were hard-masked on assistant turns while span supervision,
    loss-coverage accounting and representation auditing still admitted
    assistant-turn targets, creating an unlearnable positive-target failure
    mode. Span supervision is now user-turn-only across Dataset, coverage gate
    and audit, with the audit failing closed on target-bearing assistant spans.
44. the unreachable legacy `FlatConversationCollator` still referenced the
    removed `max_total_tokens` config field and could fail with an
    AttributeError if accidentally reused; the dead collator was removed.
45. some disabled legacy evaluator modules still reference removed historical
    model-registry keys such as `conversation_deberta`. They remain outside the
    canonical path and must be rewritten, not re-enabled as-is, during the
    evaluation migration.
46. canonical launchers defaulted to primary-only training and silently omitted
    all 1,146 detection auxiliaries; the canonical default and smoke now use
    `primary_plus_auxiliary`, and the trainer validates the declared variant.
47. confidence weights were normalized by the microbatch weight sum, cancelling
    uniform 0.25 auxiliary and 0.70 weak-evidence weights; loss reduction now
    preserves their absolute influence.
48. checkpoint selection used combined dev detection F1 despite the known A
    paired-generation artifact; canonical detection selection now uses B-dev
    AUPRC while logging combined, A-only, B-only and macro A/B diagnostics.
49. a legacy target fallback converted a single `pivot_turn_id` into turn
    supervision; it is removed, and CF-tier records without intervention-backed
    positive evidence fail closed.
50. run outputs could reuse an existing directory and mix checkpoints; launchers
    now create run-specific paths and the trainer refuses an existing output.
51. a requested CUDA run silently fell back to CPU when CUDA was unavailable;
    device resolution now fails closed.
52. source and auxiliary schema fields were carried inconsistently; all
    supervision-bearing fields and A/B source identity are now validated before
    model initialization.
53. the first restored-A validator incorrectly assumed `llm_confirmed` meant no
    positive evidence turns, although that tier can include a supported
    whole-turn anchor without a supported span; turn IDs are now accepted only
    after exact reconciliation with explicit turn/span intervention evidence.

All of the above are addressed or explicitly quarantined in the current redesign branch.

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
      test_metadata_leakage.py \
      test_training_readiness_contract.py \
      test_verify_freeze_contract.py

Submit the complete internal matrix from the repository root:

    bash submit_train_naacl_diagnostic.sh

The checked-in submitter creates four concurrent smoke jobs, four concurrent
training jobs, four concurrent internal-dev causal diagnostic jobs and one CPU
comparison job with `afterok` dependencies. It does not open held-out test data
and cannot load ShieldGemma or a public dataset.

Review `internal_dev_comparison.json` before selecting any candidate. Do not run
held-out or external evaluation until the architecture, training population and
internal causal results receive explicit signoff.

## 18. Current readiness status

Static/manual code review:

    completed

Runtime test execution:

    CPU contracts must pass at the current repair commit
    final-freeze verification and GPU architecture smoke must be rerun on HPC

Architecture redesign:

    V3 diagnostic matrix implemented, empirical selection pending

Training redesign:

    pre-response alignment and four-candidate launcher implemented

Frozen-data SHA verification code:

    passed for the restored-A primary_plus_auxiliary freeze

Representation/truncation audit:

    passed with zero records over the native 8K ceiling

CPU unit/contract suite:

    non-Torch evaluation suite passes locally
    Torch-dependent CPU contracts must run in the canonical HPC environment

GPU architecture smoke:

    historical V2 smoke passed
    all four V3 candidate smokes pending

Full training:

    historical V2 run completed but is not accepted as the final architecture
    four V3 candidate runs pending

Evaluation migration:

    unified platform implemented, see NAACL_EVALUATION.md
    internal dev-only causal diagnostic and stratified paired effects implemented
    held-out and external evaluation remain paused pending internal signoff

Held-out test access:

    not accessed during training; access follows frozen dev calibration
