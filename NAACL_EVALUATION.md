# NAACL evaluation platform

This platform evaluates the `causal_localization_v2` / restored-A v2 model being
trained at commit `8981856`. It replaces the historical single-pivot, self-mask,
and detected-only evaluation paths. Training architecture, objectives, sampling,
and checkpoint selection are unchanged.

Historical model-evaluation CLIs fail with a migration message on this branch.
Their helper functions remain available for historical inspection. Current
training preflight and representation audits remain active.

The scientific claim is **counterfactually anchored user-evidence localization**.
Evidence agreement, classifier faithfulness, independent guard sensitivity, and
live target behavior are four different outcomes. A ShieldGemma unsafe-to-safe
label change is not a target refusal or an attack-success reduction.

## Review coverage and remaining experiments

| Review concern | Implemented evaluation | Remaining empirical requirement |
|---|---|---|
| AC JdTM, MyZK: causal overclaim, weak supervision | Exact training target builder, unknown-label masking, supervision/source strata, separate outcome names | Report actual evaluated tier counts and limit causal claims to the interventions performed |
| i7qz: circular self-deletion | Text deletion before re-tokenization, separate self and independent ShieldGemma outcomes, independent live behavior judge | Run independent guard and at least two target families |
| i7qz: missing simple baselines | Same-checkpoint leave-one-user-turn-out, last-user, equal-budget lexical, repeated random, turn/run-matched random, structured LLM attribution backend | Run LLM baseline and learned comparison checkpoints |
| i7qz, aeLx: unclear model contribution | Same-cohort run matrix, checkpoint/config/train-hash provenance, context-input diagnostics | Train no-turn-context, no-span-loss, no-turn-loss, auxiliary-off and objective-weight variants. Input diagnostics are not substitutes for retraining |
| AC, 1FyB: weak external detection | Full-cohort MHJ recall, native MTID intent adapter, mixed-class metrics only where identifiable, failure bounds | Obtain the official exports, preserve splits, run all eligible records |
| 1FyB: LLM-as-judge attribution baseline | Independent JSON turn/span attributor, strict offsets, parse coverage and failed outcomes | Pin Qwen or another declared model, run without test tuning |
| aeLx: forgiving top-five hits | Top-1/2/3/5, coverage, precision lower bound, per-record random chance, chance-adjusted hit, multi-label assessed P/R/F1/AP | Independent human annotations and all baseline runs |
| MyZK: generator/validator artifacts | Source/family/tier/implicit strata, last-user and assistant-history diagnostics, aligned robustness reports, external datasets | Freeze independently checked paraphrase, noise and lexical-control pairs |
| i7qz: arbitrary utility penalty and FPR population | 5/10/15/20% budgets, lambda 0/.5/1/2/5, named benign cohorts, method-specific utility and break-even penalties | Freeze each method's detector policy using internal dev only |
| i7qz follow-up: incompatible checkpoint numbers | Full checkpoint SHA256, training hashes, code/runtime hashes, identical-manifest resume, matrix compatibility checks | Regenerate all paper tables from these artifacts. Never mix old rebuttal numbers |
| AC: user-only attribution overlooks assistants | Explicit scope, assistant-history input diagnostic, regenerated post-edit assistant suffix | State that assistant causal responsibility is outside this model's attribution space |
| Reproducibility: prompts, replay, annotation instructions | Versioned templates, exact edits, source conversion manifest, raw LLM responses, paired seeds, human task export | Publish approved data exports and target/model access instructions |

The canonical architecture already is a supervised hierarchical multi-label
tagger with independent detection, turn and span heads. It has no fusion or
self-counterfactual training loss. Calling its historical NoCF/NoFusion models
current ablations would be misleading. The evaluator accepts current-architecture
checkpoints only and does not silently substitute a model for an unknown name.

## Input and task contract

`python -m eval_platform --help` lists preparation, inference, replay, human,
robustness, utility and run-comparison commands. CPU reporting uses the standard
library. Model inference additionally uses the training PyTorch/Transformers
environment. `python -m guardlens.evaluate` uses the same CLI.

Canonical JSONL has one record per conversation:

```json
{"id":"dataset:record","cluster_id":"dataset:underlying-goal","dataset":"dataset","split":"test","source_revision":"immutable-revision","label":1,"label_semantics":"attack_intent","turns":[{"turn_id":0,"role":"user","text":"Example request"}],"gold":{},"strata":{"family":"unknown"},"objective":"Original task for independent behavior judging"}
```

Only role, text and turn ID enter model inference. The original objective is used
only by the independent behavior judge, never the detector or attributor.
Hidden rationales, synthesized knowledge, source labels and annotations stay out
of model inputs. System turns retain their roles. GuardLens cannot encode a system
role and explicitly reports these records as uncovered instead of relabeling them.
Blank, malformed and unknown-role source records fail preparation. Nothing is
silently skipped. Stable source IDs are preferred, with deterministic text hashes
as the fallback. Exact transcript, ID and declared cluster overlap is rejected.
This cannot establish absence of semantic or pretrained-model contamination.

Gold evidence is optional. When available, `gold.turns` maps assessed user turn
IDs to 0 or 1, `gold.spans` lists `turn_id/start/end/label`, and `gold.provenance`
identifies the annotation source. Unassessed text/turns are not negative labels.
Internal imports reuse `guardlens.data.causal_targets`, including restored-A
whole-turn support for records whose span tier is LLM-confirmed. Span agreement
uses assessed character positions covered by model tokens, reports coverage, and
does not pretend that unlabeled text or tokenizer gaps are negative evidence.

### Public datasets

* **MHJ:** the official release is successful human jailbreaks, so report detection
  recall over the full imported attack cohort. Precision, F1, AUROC and AP are
  null for a single-class dataset. Do not splice in internal benign records.
  `prepare --source mhj` accepts realized `turns`, `messages` or `conversation`
  arrays with explicit roles. For CSV exports use `--messages-column` for a JSON
  message-array column. Inspect the authorized export's schema instead of guessing
  a column or treating flattened prompts as natural multi-turn dialogues. Original
  redactions/missing completions may make live replay unavailable. No source flag
  becomes a gold evidence-turn label.
* **MTID:** `prepare --source mtid --subset harmful|benign` consumes its native
  `conversation`, `sample_index` and `rollout_id` fields. Use official test splits.
  Label comes from subset, never `is_jailbreak_success`, which can be true for benign
  examples. Cluster rollouts by underlying source sample. Run harmful and benign
  test exports together for intent classification after preparing each separately.
  Hidden rationale/knowledge and meta-intent are not model-visible. This is an
  **intent-transfer** task, not MTID's response-aware closure task, and not gold
  attribution evaluation. Results must carry that task label.
* **CoSafe / other exports:** supply the canonical schema with documented original
  roles, label semantics, source IDs and cluster IDs. There is no unverified native
  CoSafe parser. Do not fabricate assistant turns or prepend template conversations.
  Datasets with only attacks remain recall-only evaluations.

Sources: [MHJ](https://huggingface.co/datasets/ScaleAI/mhj),
[MTID](https://huggingface.co/datasets/Graph-COM/MTID),
[CoSafe](https://github.com/ErxinYu/CoSafe-Dataset),
[ShieldGemma model card](https://huggingface.co/google/shieldgemma-9b).

### Views, thresholds and coverage

`pre_response` stops after the last user message, before its observed response.
`retrospective` retains the realized transcript. Run and label these separately.
Earlier assistant history remains visible in both views. Never label every early
prefix unsafe merely because its final conversation is unsafe. The checkpoint's
threshold was chosen on internal dev trajectories. Applying it to pre-response
public inputs is a fixed-threshold transfer experiment, not a newly calibrated
pre-response detector. No external threshold fitting exists in this platform.

Turn and span cutoffs default to fixed .5. ShieldGemma's cutoff defaults to .5.
Pin these in the protocol before test access. Every representation limit is hard:
no turn, tokenizer or evaluator prompt truncation. Coverage failures remain in
the attempted population with explicit missing counts and detection bounds.
The runtime does not auto-discover or open held-out data.

## Intervention and transfer protocol

Budget unit is a Unicode non-whitespace word, including attached punctuation.
Offsets are Python character indices into the original text. Subword scores map
to words using maximum overlapping score. Selection is exactly
`ceil(fraction * eligible_word_count)` with deterministic turn/offset ties.
Assistant turns and special tokens are never eligible.

Methods use identical budgets: span head, repeated random, random runs matched
to reference run lengths and user turns, frozen lexical scores, last-user priority,
same-detector textual LOTO, and optional LLM attribution. The lexical lexicon is
explicitly versioned and differs from the historical phrase-weighted heuristic.
LOTO ranks signed detector probability drops. Its within-turn word scores are
constant, so deterministic word ties do not establish within-turn localization.
Native turn-head, LOTO, lexical, last-user and LLM turn agreement are reported
separately. The random-chance turn baseline accounts for each record's gold-set size.

Primary operation deletes exact word characters and leaves other bytes untouched.
`blank` replaces those characters with spaces for an artifact sensitivity run.
Neither inserts `[MASK]`. Run additional preregistered protocols with
`scope=context_only` to protect the final user request and `scope=non_surface`
to exclude lexical matches. Sufficiency keeps selected evidence **and protected
context** and deletes its eligible complement. Empty scopes are missing outcomes.

Every guard reports signed absolute probability drop, per-record relative drop,
unsafe-to-safe flips and sufficiency gap. All positives, detector-detected positives,
and guard-original-unsafe positives have separate denominators. Necessity at a
tested budget is the guard-original-unsafe flip rate, not independent evidence.
Smallest successful tested budget is descriptive, not proof of a minimal trigger.
End-to-end flips count detector misses as zero and give bounds for unavailable
outcomes. Random repeats are averaged inside each conversation before resampling.
Paired method differences use the same conversation IDs. Bootstrap samples whole
clusters, preserving twins and goal-level rollouts, with negative effects retained.
Utility is reported separately for all benign records, hard benign records,
frontier-authored benign records and interactive benign records using the frozen
dataset strata. Empty named populations remain explicit rather than being replaced.

ShieldGemma uses the pinned official chat template and the four official prompt
policies, scored separately with vocabulary `Yes`/`No` logits. The aggregate is the
maximum category probability. Its native template otherwise selects only recent
messages, so the adapter explicitly serializes the **entire visible dialogue**
with speaker roles inside one quoted JSON prompt. This is a documented transcript
adaptation, not a claim that ShieldGemma natively models multi-turn causality.
Per-category scores and prompt hashes are retained for every intervention.
Original guard detection is reported alongside transfer so low original coverage
cannot be hidden by a high conditional flip rate.

## Live behavior experiment

`replay` consumes frozen edits from an inference run. Within a conversation, every
method shares the earliest intervention point across the frozen plan. Observed
history before that point stays fixed. All later assistant responses are generated
fresh for both original and edited conditions, with matched per-turn random seeds.
Future user messages remain a fixed script. This is **scripted continuation**, not
an adaptive attacker or a total causal effect in an unconstrained conversation.
Declared seeds are deterministically namespaced by conversation ID. Original,
edited and method-control conditions share a stream within a conversation, while
unrelated conversations do not reuse an identical RNG stream.

Use at least three generation seeds and two target families, each in a separate
manifest. The original objective is held fixed for the independent behavior judge.
The judge receives generated responses and is blind to method names. Its raw JSON,
rubric, prompt inputs, generations, seeds and limits are saved. Invalid JSON,
context overflow, empty generation and output-limit exhaustion are missing outcomes,
never refusals. An original source-success flag cannot replace fresh baseline
generations. No originally unsafe trials means conditional rescue is null.
Benign trials report helpfulness loss and refusal increase. Human audit of judge
decisions is still required, including ambiguous and failed cases.

Plan replay size before reading outcomes. A 100–150-conversation pilot with three
seeds can estimate variability, but is not a power justification. Stratify frozen
IDs by dataset/intent/category/length and include detector misses. Do not choose
cases by ShieldGemma's deletion success. Smaller replay samples require a separately
frozen prepared dataset and inference run, not untracked filtering of result files.

## HPC commands

Prepare train/dev exclusions without opening test. Substitute the exact training
freeze directory used in the logs. These imports are audits, not additional training.

```bash
cd ~/projects/GuardLens-Transformer
conda activate ~/work/conda_envs/guardlens_train
FREEZE=~/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze-restored-a/splits_primary_plus_train_auxiliary
PREP=~/work/results/guardlens_eval_inputs_v1
mkdir -p "$PREP" logs
for SPLIT in train dev; do
  python -m eval_platform prepare --source internal --split "$SPLIT" \
    --source-revision restored-a-v2 \
    --input "$FREEZE/$SPLIT.jsonl" --output "$PREP/$SPLIT.jsonl"
done

# Use an authorized MHJ export with realized role-tagged messages.
python -m eval_platform prepare --source mhj --split test \
  --source-revision "$(sha256sum "$MHJ_RAW" | cut -d' ' -f1)" \
  --input "$MHJ_RAW" --output "$PREP/mhj.jsonl"

# Run online once on a machine with model access. This freezes the resolved SHA
# and copies the official policy text from that exact revision's model card.
python -m eval_platform shield-config --output "$PREP/shieldgemma.json"
# Pre-download the model/tokenizer at the SHA in that config before an offline job.
```

After training completes, explicitly select `best_joint.pt`. Do not replace it
with separate detection and attribution checkpoints within one reported run.

Before opening held-out test data, audit detector operating points on the exact
prepared dev partition. `calibrate` records the checkpoint, source data and code
hashes and freezes maximum-F1 plus 1%, 5% and 10% canonical-B FPR policies. This
does not overwrite the checkpoint. The evaluation launcher requires an explicit
policy and verifies the calibration artifact against both the checkpoint and its
original dev SHA256. The primary policy is maximum dev recall subject to at most
5% FPR on canonical source family B. The other frozen policies are reported as
an operating curve, not selected after test access.

```bash
export EVAL_CHECKPOINT=~/work/results/guardlens_naacl_redesign/primary_plus_auxiliary/guardlens/restored-a-v2-seed42-20260922/checkpoints/best_joint.pt
export EVAL_DEV="$PREP/dev.jsonl"
export EVAL_CALIBRATION=~/work/results/guardlens_eval_v1/calibration/dev_operating_points.json
export EVAL_CALIBRATION_POLICY=max_recall_at_b_fpr_5pct
python -m eval_platform calibrate --data "$EVAL_DEV" \
  --data-sha256 "$(sha256sum "$EVAL_DEV" | cut -d' ' -f1)" \
  --checkpoint "$EVAL_CHECKPOINT" \
  --output "$EVAL_CALIBRATION"

export EVAL_DATA="$PREP/mhj.jsonl"
export EVAL_DATA_SHA256=$(sha256sum "$EVAL_DATA" | cut -d' ' -f1)
export EVAL_TRAIN_EXCLUDE="$PREP/train.jsonl"
export EVAL_DEV_EXCLUDE="$PREP/dev.jsonl"
export EVAL_SHIELD_CONFIG="$PREP/shieldgemma.json"
export EVAL_RUN_ROOT=~/work/results/guardlens_eval_v1/mhj-pre-response-parallel-v2
export EVAL_SHARDS=4
./submit_eval_mhj.sh
```

The launcher requires `transformers>=4.56.2,<5` in the evaluation environment
and checks this before Slurm submission. To update an older environment, run
`~/work/conda_envs/guardlens_train/bin/python -m pip install 'transformers>=4.56.2,<5'`
before submitting. The ShieldGemma loader verifies all floating parameters have
the requested dtype before transfer to CUDA. The GuardLens backbone uses the
same `dtype=` loader argument and verifies its floating parameters after load.

The launcher accepts `EVAL_SHARDS` from 1 to 4, defaulting to 4. It submits a
Slurm array with one A100 per task and a CPU
finalization job that runs only after every task succeeds. Each task handles
indices `index % EVAL_SHARDS == task_id` and writes an atomic record result under
`$EVAL_RUN_ROOT/shards/`. The finalizer checks all input records, combines them
in original order, and writes `predictions.json`, `interventions.json`, and
`report.json`. The prior single-GPU run cannot reuse this directory because the
evaluation code identity has changed. Use the new root above for the full run.
The 26–30 GPU-hour estimate corresponds to roughly 6.5–7.5 hours of compute
with four busy A100s, plus scheduling and record-length imbalance.

Monitor with `squeue -u "$USER"` and `tail -f logs/eval_public_ARRAYID_0.out`
(substitute the printed array job ID and task 0–3). If a task times out or fails,
submit `./submit_eval_mhj.sh` again with the same environment and run root.
Completed records are skipped after their manifest, identity, and checksum
validate. A successful retry triggers a new finalization job. The old dependency
remains unsatisfied and can be cancelled with `scancel OLD_FINALIZE_JOB_ID`.

For the already completed four-shard MHJ run whose manifest has code commit
`fb30f115d49065013f072fa7b3d891d5ca8c7cb5`, an older audit serializer
hashed integer turn IDs before writing JSON. The recovery finalizer verifies
that exact legacy checksum after restoring only `audit.per_turn_count` keys.
It does not load models or rewrite any shard. With the original `EVAL_DATA`,
`EVAL_RUN_ROOT`, and `EVAL_SHARDS=4` exported, run:

```bash
sbatch --export=ALL,EVAL_RECOVER_LEGACY_SHARDS=1 eval_mhj_finalize.slurm
```

Use the existing `mhj-pre-response-parallel-v2` root for this recovery. Do not
submit the GPU array after updating the code because the new code identity
does not match that completed run's manifest.
The recovery finalizer streams the full predictions and interventions to disk,
retaining only metric inputs in memory. It requests 32 GB CPU memory and four
hours. An earlier 16 GB finalizer was OOM killed while assembling full records.

The launcher fails on any command error, uses the current environment, and has no
checkpoint/path fallback. Exact manifest resume is supported. Changing data,
checkpoint, protocol, policy, lexicon or evaluation source requires a new run root.
For a cheaper detection-only GPU pass, use `run --guard none` with the same required
arguments. For LLM attribution, add `llm` to a copied protocol's methods and pass
`--llm-config` on the CLI. The JSON model config is:

```json
{"model":"MODEL_ID","revision":"40_CHARACTER_COMMIT_SHA","max_input_tokens":16000,"max_new_tokens":1024,"temperature":0}
```

Use the same format for target and judge configs, choose capacities compatible
with each model, and set a sampling temperature for target generations.

```bash
python -m eval_platform replay --run "$EVAL_RUN_ROOT" --data "$EVAL_DATA" \
  --target-config target.json --judge-config judge.json \
  --seeds 42 43 44 --output ~/work/results/guardlens_eval_v1/mhj-target-one

python -m eval_platform human-tasks --data human.jsonl --output human_tasks.json
python -m eval_platform human-report --data human.jsonl \
  --annotations annotations.jsonl --run human-run --output human_report.json
python -m eval_platform robustness-report --original-run original-run \
  --variant-run paraphrase-run --pairs aligned_pairs.jsonl --output robustness.json
python -m eval_platform compare-runs --runs full=full-run no_span=no-span-run \
  --output ablation_matrix.json
```

`aligned_pairs.jsonl` requires `original_id`, `variant_id`, `kind`, one-to-one
`turn_mapping`, `semantic_equivalence_verified: true`, and
`verification_provenance`. It never computes token correlations from unaligned
paraphrase tokens. Human reports retain per-annotator outcomes, ambiguous labels,
missing predictions, pairwise Cohen kappa and turn Jaccard on nonempty unions.

`utility-report --rows rows.jsonl --output utility.json` supports method-specific
detectors. Rows require `id/cluster_id/method/label/population_id/detector_policy_id`,
signed `drop` for positives and boolean `flagged` for benigns. All methods must use
the same named IDs/labels. Choose detector policies on internal dev and retain
their calibration artifacts. The normal run's utility uses a shared detector,
clearly states that its FPR is identical across attribution methods, and cannot
support a claim of superior attribution specificity by itself.

## Outputs and paper-use gate

Each run stores `manifest.json`, content-addressed inference cache,
`predictions.json`, exact `interventions.json`, `progress.json` and `report.json`.
Replay stores its own parent-linked manifest, raw replay rows and behavior report.
No historical result JSON is imported. Do not claim a review concern empirically
resolved just because its evaluation code exists. Before paper tables are final:

1. Finish training, freeze the chosen checkpoint and all protocols.
2. Audit public import sizes, original splits, redactions, length coverage and
   train/dev overlap before running the internal held-out evaluation.
3. Run full public detection, independent guard controls, LLM attribution,
   source/tier-specific held-out evidence evaluation and human agreement.
4. Run independently judged live target experiments with benign controls and
   audit failure/ambiguous cases.
5. Train fair architectural and objective-weight comparisons, evaluate with the
   same manifests, and report unfavorable results as well as favorable ones.
6. Generate new tables with denominators and paired cluster intervals. Report
   changed claims honestly if gains or transfer do not replicate.

Tests: `python -m unittest test_eval_platform test_eval_runtime` exercises the
protocol, adapters, offsets, statistics, real tiny-model CPU inference, ShieldGemma
scoring contract, LLM parsing, replay pairing and end-to-end resume. It is not a
substitute for a GPU smoke with the actual completed checkpoint and pinned models.
