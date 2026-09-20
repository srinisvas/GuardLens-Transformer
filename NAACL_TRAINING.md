# GuardLens NAACL causal-localization training contract

This branch supersedes the old single-pivot, gated-fusion, Phase-3 self-CF and CF-oversampling training recipe.

## Canonical model

The main model jointly learns three sibling tasks over shared hierarchical representations.

1. trajectory detection
2. multi-label evidence-turn localization
3. context-conditioned token/span localization

DeBERTa encodes each realized turn. Token representations are pooled within each turn, then a two-layer turn Transformer contextualizes the turn sequence. Detection pools those contextualized turns. The evidence-turn head emits an independent sigmoid logit for each turn. The span head combines each token representation with the contextual state of its containing turn.

Localization never gates or otherwise feeds the detector.

The canonical model has no pivot classifier, no gated fusion, no Phase-3 self-counterfactual objective and no CF oversampling.

## Frozen data

Canonical artifacts are read from:

    $HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze

Primary training uses:

    splits_primary/train.jsonl
    splits_primary/dev.jsonl

The optional auxiliary ablation uses:

    splits_primary_plus_train_auxiliary/train.jsonl
    splits_primary_plus_train_auxiliary/dev.jsonl

The trainer never opens the held-out test split. Final test evaluation is a separate stage after model and training choices are frozen.

Before training, guardlens.data.verify_freeze checks train/dev SHA-256 values against data_prep_freeze_report.json.

## Supervision contract

Primary trajectory labels are behaviorally validated and always receive detection weight 1.0. Localization confidence never down-weights detection.

Detection-only auxiliary records use detection_label and detection_loss_weight. Their turn and span localization targets are completely ignored.

Span positives are only intervention-backed cf_strong or cf_weak spans. Incidental / negative_control_supported spans are explicit negatives. LLM-confirmed, construction-derived, unassessed and semantically masked spans are ignored for localization.

Turn positives are the full evidence_turn_ids set plus supported tested turn interventions. Explicit not_supported tested turns are negatives. Untested or not-assessable malicious turns are ignored. Validated benign user turns are negative turn-localization examples.

## Representation contract

There is no silent truncation.

- records over max_turns fail
- turns over max_tokens_per_turn fail
- the collator tokenizes with truncation disabled
- token padding is dynamic to the longest realized turn in each batch
- the representation audit reports p95, p99 and max turn-token lengths
- the audit reports whether any positive causal span would fall beyond the configured ceiling

The default max_tokens_per_turn value is 512. It is a hard ceiling, not a padding length.

## Training schedule

Phase 1 is detection-only bootstrap.

Phase 2 jointly trains detection, evidence-turn localization and span localization. Detection remains fully weighted while localization ramps from 0.25 to its configured weight.

Checkpoint selection is dev-only.

- best_detection.pt uses dev detection F1
- best_localization.pt uses the mean of dev turn and span AUPRC
- best.pt selects the localization checkpoint when available

Each checkpoint records the exact train/dev SHA-256 values.

## Required order

Run CPU preflight and training with:

    sbatch train_naacl.slurm

The launcher performs frozen SHA verification, representation coverage audit, train/dev length-shortcut probing and then training.

Before a full run, execute the one-GPU architecture smoke:

    sbatch smoke_naacl_window.slurm

The smoke selects a localizable malicious record plus a benign record and exercises detection, evidence-turn localization and span localization in the joint phase.

Do not run the legacy evaluation launchers against redesigned checkpoints until the evaluation migration is complete.
