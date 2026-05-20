# GuardLens Project Brief for Any AI Assistant

Use this as a startup brief at the beginning of a new ChatGPT, Claude, or Gemini chat. It is written as a self-contained “skill file” so the assistant can help continue the GuardLens project, especially paper writing for EMNLP/ARR.

---

## Role you should take

You are helping me write and finalize a research paper for EMNLP/ACL Rolling Review.

Act as a rigorous research collaborator, not a casual summarizer. Be skeptical, specific, and paper-oriented. Help with:

- paper structure
- contribution framing
- methods wording
- evaluation tables
- limitation wording
- reviewer-risk mitigation
- LaTeX-ready paragraphs
- concise result interpretation
- figure/table planning
- rebuttal-style reasoning
- checking whether claims are supported by results

Do **not** overclaim. If a result is weak, say so. If a claim is unsupported, push back.

Keep responses crisp unless I explicitly ask for detail.

Avoid em dashes.

---

# 1. Project identity

## Project name

**GuardLens**

Possible full title direction:

> GuardLens: Causal Token Attribution for Multi-Turn LLM Guardrail Failures

Other title direction:

> Beyond Surface Risk: Causal Token Attribution for Multi-Turn Guardrail Failures

## Intended venue

Primary target:

> **EMNLP Main Track / ACL Rolling Review**

Not Industry Track unless the paper is reframed as a deployed system. Current work is methodological, not production-deployment focused.

## Current recommendation

Submit to **EMNLP Main**, assuming the writing is careful and human audit is included.

Current competitiveness estimate after latest deconfounded/utility results:

- earlier core results only: borderline, about 40-55%
- with current utility/deconfounded results: credible/competitive, about 55-70%
- with clean human audit: stronger, about 60-75%

These are planning estimates, not guarantees.

---

# 2. Core research goal

The project studies:

> **Token-level and turn-level causal attribution of multi-turn LLM guardrail failures.**

The goal is not just to detect whether a conversation is adversarial. The key question is:

> Which user tokens, spans, and conversational regions caused or contributed to an LLM guardrail failure?

Multi-turn guardrail failures often arise through gradual escalation rather than a single obviously malicious prompt. The model must distinguish:

- truly causal adversarial evidence
- benign surface-risk vocabulary
- decoys / false leads
- academic, defensive, research, or safety discussions that contain suspicious words but are not adversarial

---

# 3. Core problem being solved

Existing safety evaluation usually asks:

> Did the model refuse or comply?

GuardLens asks:

> Which parts of the user conversation made the failure happen?

The project targets multi-turn settings where adversarial intent can be distributed across:

- setup turns
- roleplay
- fictional framing
- academic framing
- context injection
- gradual normalization
- task decomposition
- perspective shifts
- authority impersonation
- lexical payload turns
- final escalation

Important distinction:

> Sensitive words alone do not make a conversation adversarial.

A benign conversation can mention terms like “attack,” “exploit,” “payload,” “malware,” “jailbreak,” or “phishing” in safe contexts.

---

# 4. Strongest current paper story

Use this framing:

> Multi-turn jailbreaks can emerge through distributed conversational context rather than a single malicious prompt. Surface-risk heuristics can appear strong under raw deletion metrics because many adversarial examples contain compact lexical cues. However, those heuristics lack causal specificity and catastrophically overfire on benign conversations containing safe high-risk vocabulary. GuardLens offers a better attribution-specificity tradeoff by learning tier-supervised, multi-turn token attribution that remains causally effective while preserving very low false-positive behavior on hard benign and deconfounded stress tests.

The central insight is:

> Raw deletion alone is an incomplete attribution metric because it rewards aggressive keyword removal. Attribution should also penalize benign over-attribution.

This is why the paper introduces / uses **Attribution Utility**:

```text
Utility = DD@15 - λ · FPR
```

Headline setting:

```text
λ = 1.0
FPR = boundary benign false-positive rate
```

---

# 5. What not to claim

Avoid these claims:

1. “GuardLens beats all baselines.”
   - False. Surface-risk beats GuardLens on raw deletion DD@15.

2. “GuardLens beats surface-risk on raw deletion.”
   - False.

3. “Counterfactual Phase 3 improves attribution.”
   - Not supported. Phase 3 ran, but best attribution checkpoint was from Phase 2.

4. “Fusion is essential.”
   - Not supported. NoFusion is not worse.

5. “NoCF is worse.”
   - Not supported. NoCF is similar or slightly better in some metrics.

6. “Exact pivot localization is strong.”
   - False. Exact pivot is low.

7. “MHJ generalization is strong.”
   - False for detection. MHJ recall is weak.

8. “Surface-risk only works because exact dictionary phrases are present.”
   - Not fully supported. Neutralization did not break surface-risk much.

9. “Random token F1 is meaningful.”
   - It is suspiciously inflated. Do not report it.

10. “Fully causal ground truth.”
   - Use “counterfactually validated,” “tiered causal supervision,” or “causal evidence tiers.”

---

# 6. High-level methodology

The project pipeline is:

```text
Interactive multi-turn generation
  -> cross-model behavioral validation
  -> LLM span annotation
  -> pivot-turn and span-level counterfactual analysis
  -> validated benign pool
  -> final train/dev/test split
  -> GuardLens training
  -> causal attribution evaluation
  -> specificity/deconfounding/human/external evaluation
```

## Main generation setup

Generator models:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`

Target model:

- `meta-llama/Meta-Llama-3-8B-Instruct`

Validator / causal analyzer:

- `mistralai/Mistral-7B-Instruct-v0.3`

Other evaluation models:

- Qwen validation
- ShieldGemma as secondary external evaluator

## Key generation decision

Static generation and augmentation were dropped.

Final decision:

> Use interactive generation with feedback loop. Remove interactive augmentation. If hard negatives are needed, generate them separately as validated benign pools.

Why:
- Template augmentation diluted quality.
- Dedup removed most augmented records.
- Interactive 7B generation worked because feedback loop mattered more than model size.

---

# 7. Dataset details

## Final dataset

Final dataset after malicious postprocess + clean benign merge:

```text
Total records: 1,762
Train: 1,220
Dev: 266
Test: 276
```

Test split:

```text
87 adversarial
189 benign
```

Train split:

```text
383 malicious
837 benign
```

Train supervision tiers:

```text
llm_confirmed: 237
construction: 452
cf_weak: 22
cf_strong: 7
benign_validated: 502
```

Important:

```text
cf_weak + cf_strong in train = 29 / 1,220
```

This sparsity explains why Phase 3 CF optimization did not improve beyond Phase 2.

## Final split details

Train:
- 1,220 records
- labels:
  - malicious: 383
  - benign: 837
- pivot kinds:
  - contextual_pivot: 213
  - lexical_pivot: 160
  - misleading_decoy: 3
  - none: 844
- top families:
  - interactive_adversarial: 383
  - interactive_benign_twin: 335
  - clean_everyday: 206
  - research_technical: 132
  - topic_matched_safe: 72

Dev:
- 266 records
- malicious: 79
- benign: 187

Test:
- 276 records
- malicious: 87
- benign: 189
- supervision tiers:
  - llm_confirmed: 52
  - construction: 105
  - cf_weak: 4
  - cf_strong: 2
  - benign_validated: 113

## Benign pool

Separate benign pool was created and validated by both Llama and Mistral.

Files:

- `benign_pool_v11_llama_validated.jsonl`
- `benign_pool_v11_mistral_validated.jsonl`
- `benign_clean.jsonl`
- `benign_boundary.jsonl`

Clean benign rule:

> Keep only records accepted by both Llama and Mistral as non-jailbreak.

Boundary benign:

> Records rejected by either validator are excluded from training and used for boundary stress testing.

Boundary records are crucial because they test false positives.

---

# 8. Supervision tiers

The project uses hierarchical supervision tiers.

Tiers:

- `cf_strong`
- `cf_weak`
- `llm_confirmed`
- `construction`
- `benign_validated`
- `incidental`
- `ignore`

Training weights:

```python
span_tier_weights = {
    "cf_strong": 1.00,
    "cf_weak": 0.70,
    "llm_confirmed": 0.60,
    "construction": 0.40,
    "llm_only": 0.25,
    "incidental": 1.00,  # negative supervision
    "ignore": 0.00,
}
```

Important methodological point:

> Counterfactual validation currently contributes more as dataset quality control and supervision calibration than as a direct Phase 3 training signal, because CF-validated records are sparse.

---

# 9. Model and architecture

Main model:

> **GuardLens**

Backbone:

```text
microsoft/deberta-v3-base
```

Backbone is frozen.

Approximate model size:
- total: about 186M parameters
- trainable: about 2.17M parameters

Important config:

```python
max_turns = 32
max_tokens_per_turn = 192
max_total_tokens = 2048

batch_size = 4
gradient_accumulation = 4
learning_rate = 2e-4
weight_decay = 0.01
warmup_steps = 200
max_grad_norm = 1.0
```

Architecture components:

- frozen DeBERTa encoder
- multi-turn representation
- cross-turn attention / transformer layers
- classification head
- attribution head
- optional pivot head
- optional gated fusion

Loss weights:

```python
lambda_cls = 0.2
lambda_attr = 1.0
lambda_cf = 0.5
lambda_pivot = 0.3
```

Training phases:

```text
Phase 1: classification bootstrap, epochs 0-4
Phase 2: attribution / tier supervision, epochs 5-19
Phase 3: counterfactual consistency, epochs 20-24
```

Important:

> Phase 3 ran, but the best attribution checkpoint was still selected from Phase 2.

---

# 10. Training results

Final training summary:

| Model | F1 | Accuracy | AttrF1 | Pivot Acc | Threshold |
|---|---:|---:|---:|---:|---:|
| GuardLens | 0.9943 | 0.9964 | 0.8778 | 0.7536 | 0.56 |
| GuardLens-NoFusion | 1.0000 | 1.0000 | 0.8810 | 0.7536 | 0.20 |
| GuardLens-NoCF | 0.9943 | 0.9964 | 0.8841 | 0.7645 | 0.20 |
| Turn-level | 0.7582 | 0.8659 | 0.0000 | 0.0000 | 0.55 |
| ConversationDeBERTa | 0.8324 | 0.8949 | 0.0000 | 0.0000 | 0.59 |

ConversationDeBERTa exact:

```text
f1: 0.8323699421965317
accuracy: 0.894927536231884
precision: 0.8372093023255814
recall: 0.8275862068965517
threshold: 0.59
```

Interpretation:

- GuardLens strongly beats turn-level and flat DeBERTa detection baselines.
- NoFusion and NoCF do not underperform GuardLens.
- Therefore, the paper should not claim architecture modules independently drive gains.
- The main contribution is dataset + tiered supervision + evaluation framework + multi-turn modeling.

---

# 11. Evaluation suite

The evaluation is extensive and includes:

1. Core causal attribution
2. Per-supervision-tier attribution
3. Per-transfer-tier attribution
4. Classification baselines
5. Surface-risk FPR comparison
6. Boundary stress test
7. Contextual vs lexical pivot analysis
8. Paraphrase robustness
9. Attribution precision / minimality
10. Cross-dataset generalization
11. Cross-model transfer with ShieldGemma
12. MHJ external eval
13. Attribution Utility
14. Deconfounded surface-risk stress tests
15. Causal Turn Mass
16. Pivot Window Accuracy
17. Human audit benchmark, planned or in progress

---

# 12. Core causal attribution results

Core causal attribution methods:

- GuardLens
- attention
- integrated gradients
- grad×input
- surface-risk
- random

Key metrics:

- DD@k: deviation drop after removing top-k attributed tokens
- Flip@k: fraction of adversarial examples flipped benign
- Necessity
- Sufficiency
- Trigger size
- Token F1, but with caveats

Important raw result:

```text
Surface-risk beats GuardLens on raw deletion.
```

Approximate full DD@15:

```text
GuardLens: 0.511
Surface-risk: 0.568
Grad×Input: 0.361
Integrated Gradients: 0.297
Attention: 0.058
Random: about 0 or negative
```

GuardLens token attribution F1:

```text
0.8778
```

Surface-risk token F1:

```text
0.0583
```

Interpretation:

- Surface-risk deletes compact lexical cues efficiently.
- GuardLens aligns with causal-span supervision much better.
- Raw deletion alone is not enough.

Do not report random token F1 unless fixed. It is suspiciously inflated around 0.61 due to metric definition.

---

# 13. Surface-risk issue and resolution

## Problem

Surface-risk baseline beats GuardLens on raw causal deletion.

Reason:

- many adversarial records contain compact lexical risk cues
- deletion metric rewards removing these cues
- surface-risk is effectively a lexical-oracle stress baseline
- it does not need to learn semantics

## Resolution

Introduce specificity-aware evaluation:

```text
Utility = DD@15 - λ · FPR
```

Headline uses:

```text
λ = 1.0
FPR = boundary false-positive rate
```

Latest utility results:

| Method | DD@15 | Boundary FPR | Utility |
|---|---:|---:|---:|
| GuardLens | 0.511 | 0.007 | 0.504 |
| Surface-risk | 0.568 | 0.373 | 0.195 |

This is now the most important result.

Interpretation:

> Surface-risk is deletion-effective but not causally specific. GuardLens gives a better causal-specificity tradeoff.

---

# 14. Boundary and deconfounded results

## Boundary stress

Boundary rejected benign set:

```text
279 records
GuardLens FPR = 0.0072
False positives = 2/279
```

Surface-risk boundary FPR@0.5:

```text
0.3728
```

## Surface-risk FPR@0.5

```text
all_benign: 0.143
boundary_rejected: 0.373
boundary_false_lead_benign: 0.576
false_lead_benign: 0.667
validated_benign_twin: 0.224
```

## SR-injected benign stress test

Surface-risk phrases were inserted into benign records in safe contexts.

Result:

```text
189 injected benign records
GuardLens FPR = 0.005 = 1/189
Surface-risk FPR = 0.947 = 179/189
```

This is one of the strongest paper results.

Use wording:

> Surface-risk collapses under benign high-risk vocabulary injection, while GuardLens remains highly specific.

---

# 15. Deconfounded evaluation

Variants:

1. Original
2. SR-neutralized adversarial
3. SR-neutralized changed-only
4. Noise-equalized adversarial
5. Combined neutralized + noisy
6. SR-injected benign

Latest DD@15:

| Variant | GuardLens | Surface-risk | Random |
|---|---:|---:|---:|
| Original | 0.473 | 0.543 | -0.002 |
| SR-neutralized | 0.449 | 0.534 | -0.003 |
| SR-neutralized changed-only | 0.415 | 0.504 | -0.003 |
| Noise-equalized | 0.504 | 0.502 | -0.002 |
| Combined | 0.484 | 0.491 | -0.002 |

Important details:

```text
Original adversarial SR mean = 0.750
80/87 adversarial records had SR > 0.3
SR-neutralized changed 81/87 records
Neutralized SR mean = 0.000
0/87 had SR > 0.3 after neutralization
87/87 neutralized records still detected adversarial
```

Interpretation:

- Neutralization does not destroy surface-risk performance.
- Surface-risk is not only exact phrase matching.
- It likely exploits residual structural/lexical risk cues.
- Noise-equalized condition makes GuardLens match/slightly beat surface-risk.
- Combined condition makes the gap tiny.
- The strongest difference is specificity, not raw DD.

Paper wording:

> Deconfounding narrows the raw deletion gap and shows GuardLens is not purely lexical, but the decisive distinction is specificity: surface-risk overfires on benign high-risk language while GuardLens remains stable.

---

# 16. Pivot and causal turn results

Exact pivot accuracy is weak, so use better metrics.

Latest pivot-window results:

| Window | Accuracy |
|---|---:|
| exact | 0.092 = 8/87 |
| ±1 | 0.241 = 21/87 |
| ±2 | 0.379 = 33/87 |
| ±3 | 0.483 = 42/87 |
| ±5 | 0.678 = 59/87 |

Distance:

```text
mean = 5.1
median = 4.0
p75 = 7.0
p90 = 13.4
```

Causal turn mass:

```text
mean = 0.310
median = 0.307
```

Per-role attribution mass:

```text
adaptation mean = 0.0548, n=352
escalation mean = 0.0523, n=454
payload mean = 0.0512, n=62
setup mean = 0.0484, n=322
```

Paper interpretation:

> Exact pivot localization is too strict for long multi-turn conversations. GuardLens better localizes broader causal regions, reaching 67.8% within ±5 user turns.

Do not make exact pivot a headline.

---

# 17. External results

## Cross-dataset AdvBench + HarmBench

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| GuardLens | 0.652 | 0.992 | 0.437 | 0.607 |
| GuardLens-NoCF | 0.622 | 0.992 | 0.387 | 0.556 |
| ConversationDeBERTa | 0.174 | 0.129 | 0.060 | 0.082 |

Interpretation:

- GuardLens is conservative OOD.
- High precision, lower recall.
- Still stronger than NoCF and flat DeBERTa.

## MHJ external eval

MHJ classification:

```text
F1 = 0.0614
Recall = 0.0317
```

Weak detection generalization.

Detected-case attribution:

```text
GuardLens DD@15 = 0.4949
Surface-risk DD@15 = 0.2038
Random DD@15 = 0.0347
GuardLens Flip@15 = 0.6327
Surface-risk Flip@15 = 0.2857
```

Paper wording:

> MHJ remains challenging for detection, but when GuardLens detects a failure, its attributions are substantially more causally effective than surface-risk and random.

Do not claim strong MHJ detection.

## ShieldGemma transfer

Secondary evidence only.

| Subset | GuardLens | Surface-risk | Usable n |
|---|---:|---:|---:|
| transfer_success | 0.909 | 0.636 | 11 |
| lexical_pivot | 0.789 | 0.737 | 38 |
| contextual_pivot | 0.789 | 0.737 | 19 |

Caveat:
- small denominators
- random sometimes ties
- use in appendix or secondary table

---

# 18. Human audit benchmark

A human audit benchmark is planned or needed.

Target:

```text
100 conversations
50 adversarial
50 hard benign
50 double-annotated if possible
```

Annotators should fill:

- `classification`: ADVERSARIAL / BENIGN / UNCERTAIN
- `pivot_turn`: 0-indexed user turn, DISTRIBUTED, or blank
- `causal_spans`
- `confidence`: HIGH / MEDIUM / LOW
- `notes`

Agreement metrics:

- Cohen’s κ for classification, target ≥ 0.70
- pivot exact / within-one / distributed agreement
- span overlap F1

Important:

> Human-audited set must be excluded from training, threshold tuning, and model selection.

Purpose:

- reduce synthetic-label skepticism
- validate that GuardLens aligns with human causal judgments
- compare GuardLens vs surface-risk on human labels
- report benign hallucination rate

---

# 19. Code/file names to know

## Generation and dataset

- `build_semantic_datasetv11.py`
  - interactive generation script
- `launch_gen.slurm`
  - generation job
- `launch_val.slurm`
  - validation job
- `merge_validations.py`
  - merges validation outputs
- `launch_causal.slurm`
  - 4-pass causal analysis job
- `postprocess_causal.py`
  - prepares malicious final records
- `split_dataset.py`
  - creates train/dev/test and human benchmark splits
- `mhj_loader.py`
  - converts MHJ into GuardLens external test schema

## Training

- `guardlens/train.py`
- `guardlens/config.py`
- `guardlens/data/dataset.py`
- `guardlens/training/trainer.py`
- `guardlens/training/loss.py`
- `guardlens/models/*`
- `train_all.slurm`

## Evaluation

- `guardlens/evaluate.py`
- `guardlens/evaluation/eval_utils.py`
- `guardlens/evaluation/causal_eval.py`
- `guardlens/evaluation/eval_causal.py`
- `eval_boundary_stress.py`
- `eval_surface_risk_fpr.py`
- `eval_attribution_utility.py`
- `eval_deconfounded.py`
- `eval_deconfound.slurm`
- `eval_paraphrase.py`
- `eval_implicit_explicit.py`
- `eval_attribution_precision.py`
- `eval_cross_dataset.py`
- `eval_cross_model_transfer.py`
- `eval_mhj.slurm`
- `eval_full.slurm`

## Important result files

- `final_dataset.jsonl`
- `splits/train.jsonl`
- `splits/dev.jsonl`
- `splits/test.jsonl`
- `benign_boundary.jsonl`
- `causal_eval_results.json`
- `boundary_stress.json`
- `surface_risk_fpr.json`
- `attribution_utility.json`
- `deconfounded_results.json`

---

# 20. Known code/evaluation issues

1. `evaluate.py` should write clean JSON using `--output`; avoid mixed stdout logs.
2. ConversationDeBERTa must use `FlatConversationCollator`.
3. Token F1 for random is suspicious. Do not report unless fixed.
4. Token F1 on deconfounded text is invalid unless span offsets are remapped.
5. Exact pivot metric is too harsh.
6. Surface-risk transfer cache must be unique per subset in cross-model transfer.
7. ShieldGemma uses a different evaluator style than LlamaGuard; do not reuse Yes/No logit scoring blindly for LlamaGuard.
8. MHJ must remain external, not folded into train/test split.
9. Old v10 augmented data should not be used.
10. Old family names should be updated to v11 names.

---

# 21. Planned tables

## Table 1: Dataset statistics

Include:

- total records
- train/dev/test
- malicious/benign
- supervision tiers
- transfer tiers
- benign pool and boundary pool

## Table 2: Detection baselines

Rows:

- GuardLens
- NoFusion
- NoCF
- TurnLevel
- ConversationDeBERTa

Columns:

- Accuracy
- Precision
- Recall
- F1
- Threshold

## Table 3: Core causal attribution

Rows:

- GuardLens
- Attention
- Integrated Gradients
- Grad×Input
- Surface-risk
- Random

Columns:

- DD@15
- Flip@15
- Necessity@15
- Trigger size
- Token/span metric if fixed

Caveat:
- Do not report random token F1.

## Table 4: Attribution Utility

Main table:

| Method | DD@15 | Boundary FPR | Utility |
|---|---:|---:|---:|
| GuardLens | 0.511 | 0.007 | 0.504 |
| Surface-risk | 0.568 | 0.373 | 0.195 |

## Table 5: Deconfounded stress test

Rows:

- Original
- SR-neutralized
- SR-neutralized changed-only
- Noise-equalized
- Combined
- SR-injected benign FPR

## Table 6: Pivot / distributed causality

Rows:

- exact
- ±1
- ±2
- ±3
- ±5
- causal turn mass

## Table 7: External robustness

Rows:

- AdvBench/HarmBench
- MHJ detected-case attribution
- ShieldGemma optional appendix

## Table 8: Human audit

Pending.

---

# 22. Planned figures

1. Pipeline diagram:
   - generation → validation → causal annotation → training → evaluation

2. Multi-turn attribution example:
   - highlight setup / escalation / payload

3. Utility tradeoff plot:
   - x-axis FPR
   - y-axis DD@15
   - show GuardLens vs surface-risk

4. Deconfounded stress bar chart:
   - original, neutralized, noise, combined

5. Pivot-window curve:
   - window size vs accuracy

6. Surface-risk injected benign example:
   - benign conversation with risky vocabulary
   - surface-risk fires, GuardLens does not

---

# 23. Recommended paper structure

```latex
\section{Introduction}

\section{Problem Formulation}
\subsection{Multi-Turn Guardrail Failure Attribution}
\subsection{Causal Spans and Distributed Pivot Turns}

\section{Dataset Construction}
\subsection{Interactive Multi-Turn Generation}
\subsection{Cross-Model Behavioral Validation}
\subsection{Tiered Causal Annotation}
\subsection{Validated Benign and Boundary Controls}
\subsection{Dataset Statistics}

\section{GuardLens}
\subsection{Conversation Encoding}
\subsection{Token Attribution and Detection Heads}
\subsection{Tier-Weighted Supervision}
\subsection{Training Phases}

\section{Evaluation}
\subsection{Detection and Attribution Baselines}
\subsection{Deletion-Based Causal Metrics}
\subsection{Causal-Specificity Utility}
\subsection{Deconfounded Surface-Risk Stress Tests}
\subsection{External and Human-Audited Evaluation}

\section{Results}
\subsection{Detection Performance}
\subsection{Causal Attribution}
\subsection{Surface-Risk Specificity}
\subsection{Robustness and External Generalization}
\subsection{Human Audit}

\section{Analysis}
\subsection{Why Surface-Risk Wins Raw Deletion}
\subsection{Distributed Pivot Localization}
\subsection{Counterfactual Supervision Limitations}

\section{Limitations}

\section{Conclusion}
```

---

# 24. Suggested abstract skeleton

Use this as a starting point, not final prose:

> Multi-turn jailbreaks can induce unsafe LLM behavior through gradual contextual escalation, making it difficult to identify which user tokens or turns caused a guardrail failure. We introduce GuardLens, a framework for token-level causal attribution in multi-turn adversarial conversations. GuardLens combines interactive adversarial generation, cross-model behavioral validation, tiered causal supervision, and a multi-turn attribution model trained to identify causal user spans. We evaluate attribution with deletion-based metrics, paraphrase robustness, transfer-tier analysis, boundary benign stress tests, and a causal-specificity utility that penalizes false attribution on benign conversations. While lexical surface-risk heuristics achieve strong raw deletion performance, they overfire on benign high-risk contexts, producing 37.3% boundary FPR and 94.7% FPR on surface-risk-injected benign examples. GuardLens achieves a stronger attribution-specificity tradeoff, with DD@15 of 0.511, boundary FPR of 0.007, and utility of 0.504 versus 0.195 for surface-risk. Our results show that multi-turn guardrail attribution requires distinguishing causal adversarial evidence from benign surface-risk vocabulary.

---

# 25. Methodology paragraph seed

> We construct a multi-turn attribution dataset through an interactive generator-target loop rather than static prompt templating. A generator model proposes user turns, observes target-model responses, and adapts subsequent turns to induce or avoid guardrail failure. We validate adversarial behavior across target and validator models, assigning transfer tiers that distinguish target-only, cross-model, and jointly successful failures. For attribution supervision, we combine LLM-based span annotation with replay-based counterfactual validation: pivot turns and candidate spans are neutralized or removed, then replayed through a validator to estimate whether the intervention reduces unsafe behavior. This produces hierarchical supervision tiers, ranging from counterfactually strong spans to construction-derived weak spans, plus incidental spans used as explicit negative attribution evidence. We also construct a separately validated benign pool and a boundary benign set to test whether attribution methods overfire on safe conversations containing adversarial-looking vocabulary.

---

# 26. Results paragraph seed

> GuardLens achieves strong internal detection performance, substantially outperforming turn-level and flat conversation baselines. However, detection accuracy alone is insufficient for our goal. On causal deletion metrics, surface-risk is a strong lexical deletion heuristic and achieves higher raw DD@15 than GuardLens. This result is expected because many adversarial conversations contain compact lexical risk cues. The distinction emerges when deletion is evaluated together with specificity: GuardLens has boundary FPR of 0.007 compared with 0.373 for surface-risk, yielding attribution utility of 0.504 versus 0.195. In a surface-risk-injected benign stress test, surface-risk incorrectly flags 179/189 benign conversations, while GuardLens flags only 1/189. These results show that surface-risk is deletion-effective but non-specific, whereas GuardLens provides a better tradeoff between causal effectiveness and benign robustness.

---

# 27. Limitations paragraph seed

> Our work has several limitations. First, the dataset is generated rather than collected from deployed systems, although we mitigate this with cross-model validation, external benchmarks, deconfounded stress tests, and a human-audited subset. Second, counterfactually validated spans are sparse: only 29 `cf_weak/cf_strong` records appear in the training split. Consequently, although Phase 3 counterfactual-consistency training ran, the best attribution checkpoint was selected from Phase 2, suggesting that counterfactual validation currently contributes more as supervision calibration and quality control than as an independent optimization signal. Third, exact pivot-turn localization remains challenging because many failures exhibit distributed causality. We therefore report windowed pivot localization and causal-turn mass, but improving precise turn-level attribution remains future work. Finally, GuardLens has low detection recall on MHJ, indicating that broader external multi-turn generalization remains open.

---

# 28. Human audit instruction for future assistant

If asked to help with human audit:

- Keep the task simple.
- Use 100 conversations.
- 50 adversarial, 50 hard benign.
- 50 double-annotated if possible.
- Do not ask annotators to understand model internals.
- Ask them to label:
  - adversarial/benign/uncertain
  - pivot turn or distributed
  - causal spans
  - confidence
  - notes
- Report:
  - Cohen’s κ
  - pivot exact/±1/distributed agreement
  - span overlap F1
  - model vs human span precision/recall/F1
  - benign hallucination rate

---

# 29. Instructions for the next AI assistant

When continuing this project:

1. Do not restart from broad brainstorming.
2. Treat the evaluation package as mostly complete.
3. Focus on writing, table construction, claims, and reviewer-risk mitigation.
4. Be explicit when a result supports a claim or does not.
5. Preserve exact numbers.
6. Keep the surface-risk nuance central.
7. Use Attribution Utility and SR-injected benign FPR as headline results.
8. Do not overclaim NoCF/NoFusion/Phase 3.
9. Do not bury limitations.
10. Help produce LaTeX-ready paper text and compact EMNLP-style tables.
11. Prefer main-track framing: dataset + method + evaluation framework.
12. If asked for next steps, prioritize:
    - human audit
    - final tables
    - abstract/introduction
    - methodology section
    - results section
    - limitations
    - figures

---

# 30. One-paragraph project brief

GuardLens is a proposed EMNLP Main paper on causal token attribution for multi-turn LLM guardrail failures. The project builds a 1,762-record v11 dataset through interactive Qwen-to-Llama generation, cross-model validation, Mistral-based causal analysis, and a separately validated benign pool. GuardLens uses a frozen DeBERTa-v3 backbone with multi-turn encoding and token attribution heads, achieving F1 0.994 on internal detection and AttrF1 0.878. Surface-risk heuristics beat GuardLens on raw deletion DD@15, but fail specificity: GuardLens boundary FPR is 0.007 versus surface-risk 0.373, and on benign conversations injected with safe high-risk vocabulary, GuardLens FPR is 0.005 versus surface-risk 0.947. The key paper framing is that raw deletion rewards aggressive lexical removal, while reliable guardrail attribution requires causal specificity. The main evidence should be detection baselines, causal attribution, Attribution Utility, deconfounded stress tests, boundary robustness, paraphrase robustness, external stress tests, and a human audit benchmark.
