# GuardLens Paper Writing Context Pack

Portable handoff for writing the GuardLens paper in a new ChatGPT or Claude chat.

This pack focuses only on paper strategy, framing, claims, reviewer risks, methodology wording, tables/figures, and writing constraints. It incorporates the latest deconfounded evaluation and attribution-utility results.

---

## 1. Intended venue

### Primary target

**EMNLP Main Track / ACL Rolling Review**

Current recommendation: **submit to EMNLP Main**, not Industry Track.

### Why Main Track fits

The work is methodological and research-oriented:

- multi-turn guardrail failure attribution
- dataset construction and validation
- causal token/span supervision
- counterfactual analysis
- attribution-specific evaluation
- benchmark and stress-test design

This is not primarily a deployment/system paper.

### Why Industry Track is less suitable

Industry Track usually expects:

- deployed systems
- production-scale usage
- enterprise operational lessons
- real-world logs/datasets
- measurable product impact

GuardLens currently has strong research methodology and evaluation but no production deployment. So Industry Track would likely weaken the fit unless the paper is rewritten around a deployed guardrail analysis system.

### Current EMNLP chance assessment

After latest deconfounded results:

| State | Estimated competitiveness |
|---|---:|
| Earlier core results only | borderline, about 40-55% |
| Current utility + deconfounded + boundary results | credible / competitive, about 55-70% |
| If human audit lands cleanly | stronger, about 60-75% |

These are rough planning estimates, not guarantees. EMNLP is competitive and reviewer-dependent.

---

## 2. Target contribution claims

The paper should not be framed as “we built the best jailbreak classifier.” The core claim is broader and more defensible:

> We introduce a causal attribution framework for multi-turn LLM guardrail failures, combining interactive adversarial generation, cross-model validation, tiered causal supervision, and specificity-aware attribution evaluation.

### Main contributions

1. **Problem formulation**
   - Defines token/span/turn attribution for multi-turn guardrail failures.
   - Treats causal evidence as potentially distributed across conversation turns rather than localized to a single prompt.

2. **Dataset construction pipeline**
   - Interactive generator-target conversations.
   - Realistic multi-turn adversarial behavior.
   - Benign twins and separately validated benign pool.
   - Cross-model validation with Llama/Qwen/Mistral.
   - Causal span supervision from LLM annotation + counterfactual validation.
   - Transfer tiers and supervision tiers.

3. **GuardLens model**
   - Structured multi-turn model with token attribution and classification.
   - Uses tier-weighted supervision.
   - Learns token-level attribution over multi-turn conversations.

4. **Evaluation methodology**
   - Goes beyond classification accuracy.
   - Includes deletion-based causal metrics, attribution precision, paraphrase stability, transfer tiers, boundary benign specificity, deconfounded artifact stress tests, and attribution utility.

5. **Surface-risk analysis**
   - Shows surface-risk heuristics are powerful but brittle.
   - Raw deletion favors surface-risk.
   - Specificity-aware utility strongly favors GuardLens.

6. **Human audit benchmark**
   - Planned/needed to reduce synthetic-label skepticism.
   - Should be described as excluded from training and model selection.

---

## 3. Novelty framing

### Best novelty framing

The strongest novelty is not a single model component. It is the **framework**:

> GuardLens is a framework for evaluating and learning causal token attribution in multi-turn guardrail failures under distributed conversational causality.

Frame novelty around:

- multi-turn attribution rather than single-turn jailbreak detection
- causal span supervision rather than post-hoc explanation alone
- validated benign controls rather than only adversarial examples
- specificity-aware attribution utility rather than raw deletion alone
- controlled deconfounding to expose surface-risk brittleness
- human audit as independent attribution validation

### What not to overemphasize as novelty

Do not overemphasize:

- gated fusion
- Phase 3 CF optimization
- pivot head
- architecture superiority over all ablations

Why:
- NoFusion and NoCF are not worse in current numbers.
- Phase 3 ran but best attribution checkpoint came from Phase 2.
- Exact pivot accuracy is low.

### Novelty sentence candidates

Use something like:

> Unlike prior jailbreak benchmarks that primarily measure whether a model refuses or complies, GuardLens studies where a multi-turn failure becomes causally attributable in the user’s conversation, using tiered supervision and counterfactual validation to distinguish causal adversarial evidence from benign surface-risk vocabulary.

Or:

> We show that lexical surface-risk heuristics can appear strong under deletion metrics while failing causal specificity, motivating attribution evaluation that jointly measures deletion effectiveness and benign over-attribution.

---

## 4. Methodology wording

### Dataset construction wording

Use precise wording:

> We construct multi-turn adversarial conversations through an interactive generator-target loop. A generator model proposes user turns, observes target-model responses, and adapts subsequent turns to escalate or redirect the conversation. For each adversarial conversation, we also construct benign or safety-preserving counterparts and separately generate a validated benign pool.

Mention models:

- Generator:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `Qwen/Qwen2.5-14B-Instruct`
- Target:
  - `meta-llama/Meta-Llama-3-8B-Instruct`
- Validator / causal analyzer:
  - `mistralai/Mistral-7B-Instruct-v0.3`
- Additional validation:
  - Qwen validation
  - ShieldGemma secondary external evaluator

### Validation wording

> We assign transfer tiers based on whether the adversarial behavior transfers across target and validator models. This separates target-only failures from cross-model failures and avoids treating all generated adversarial conversations as equally reliable.

Transfer tiers:

- `transfer_success`
- `target_only`
- `cross_only`
- `no_jailbreak`
- `benign`

### Supervision wording

> We use hierarchical supervision tiers reflecting the confidence of causal evidence. Counterfactually validated spans receive stronger supervision, LLM-confirmed spans receive moderate supervision, construction-derived spans receive weaker supervision, and incidental spans serve as explicit negative attribution supervision.

Tier weights:

```text
cf_strong = 1.00
cf_weak = 0.70
llm_confirmed = 0.60
construction = 0.40
llm_only = 0.25
incidental = 1.00 as negative supervision
ignore = 0.00
```

### Phase 3 wording

Do not say Phase 3 failed. Say:

> Although the training pipeline includes a Phase 3 counterfactual-consistency stage, the best attribution checkpoint in our main run was selected from Phase 2. We attribute this to the sparsity of counterfactually validated examples: only 29 `cf_weak/cf_strong` records appear in the 1,220-record training split. Since Phase 2 already incorporates these examples through tier-weighted supervision, Phase 3 mainly reweights a sparse signal and did not improve validation attribution metrics. We therefore treat counterfactual validation primarily as conservative quality control and supervision calibration in this work.

### Surface-risk wording

Use this exact conceptual framing:

> Surface-risk is a strong lexical-oracle stress baseline. Because some adversarial turns contain recognizable risk-bearing phrases, surface-risk can be highly effective under raw deletion metrics. However, this does not imply causal specificity: the same heuristic overfires on benign conversations containing safe high-risk vocabulary.

Avoid calling surface-risk a fully fair independent baseline.

---

## 5. Reviewer-risk points

### Risk 1: Synthetic-heavy dataset

Likely reviewer criticism:

> The dataset is generated and may not reflect real-world jailbreaks.

Mitigation:

- interactive target-feedback generation
- real seed integration if present
- cross-model validation
- MHJ external eval
- AdvBench/HarmBench cross-dataset eval
- human audit benchmark
- boundary benign pool
- deconfounded evaluation

Paper wording:

> We do not claim generated conversations exhaust real-world jailbreak behavior. We use cross-model validation, external stress tests, and a human-audited subset to assess whether attribution behavior generalizes beyond generator artifacts.

### Risk 2: Surface-risk beats GuardLens on raw deletion

This is the main technical risk.

Observed:
- Surface-risk DD@15 > GuardLens DD@15 on internal raw deletion.
- Example full utility source:
  - GuardLens DD@15 = 0.511
  - Surface-risk DD@15 = 0.568

Answer:
- Raw deletion rewards compact lexical cue removal.
- Surface-risk is a lexical deletion heuristic.
- GuardLens has much stronger specificity.

Key numbers:
- GuardLens boundary FPR = 0.0072
- Surface-risk boundary FPR@0.5 = 0.3728
- GuardLens utility = 0.504
- Surface-risk boundary utility = 0.195
- SR-injected benign FPR:
  - GuardLens = 0.005
  - Surface-risk = 0.947

Paper framing:
> Surface-risk is deletion-effective but causally non-specific.

### Risk 3: NoCF and NoFusion ablations do not underperform

Observed:

```text
guardlens           F1 0.9943 AttrF1 0.8778
guardlens_no_fusion F1 1.0000 AttrF1 0.8810
guardlens_no_cf     F1 0.9943 AttrF1 0.8841
```

Do not claim:
- gated fusion is necessary
- CF phase improves performance
- every architecture component is validated

Instead:
> The primary gains arise from structured multi-turn modeling, tier-aware supervision, and the dataset/evaluation framework. Fusion and CF-specific Phase 3 did not yield additional gains in this run.

### Risk 4: Exact pivot accuracy low

Observed:
- exact pivot in fixed utility eval: 0.092
- within ±1: 0.241
- within ±2: 0.379
- within ±3: 0.483
- within ±5: 0.678

Answer:
- Multi-turn failures often have distributed causality.
- Exact pivot is too strict.
- Use pivot-window accuracy and causal-turn mass.

Paper wording:
> Exact pivot localization is conservative for long multi-turn interactions. We therefore report windowed pivot localization and causal-turn mass to capture distributed causal regions.

### Risk 5: MHJ external detection weak

Observed:
- MHJ classification F1 = 0.0614
- recall = 0.0317
- But detected-case attribution:
  - GuardLens DD@15 = 0.4949
  - Surface-risk DD@15 = 0.2038
  - Random DD@15 = 0.0347

Wording:
> MHJ remains a challenging external stress test. GuardLens has low detection recall, but among detected failures, its attributions remain more causally effective than surface-risk and random ablations.

Do not claim strong MHJ generalization.

### Risk 6: Random token F1 suspicious

Observed:
- Random token F1 around 0.61 in causal eval, clearly suspicious.

Decision:
- Do not report random token F1.
- Use random only for deletion DD/flip where it is near zero.
- Fix metric if including token-F1 table.

### Risk 7: Token F1 on deconfounded variants invalid

Text perturbations change offsets. Do not report token F1 for neutralized/noisy variants unless spans are remapped.

Use:
- DD@k
- Flip@k
- FPR
- sanity retained-detected count

---

## 6. Related work positioning

### Buckets to cover

1. **Jailbreak and prompt injection benchmarks**
   - single-turn jailbreak datasets
   - multi-turn jailbreak datasets
   - AdvBench, HarmBench, JailbreakBench, MHJ

2. **LLM safety / guardrail evaluation**
   - refusal/compliance measurement
   - safety classifiers
   - LlamaGuard, ShieldGemma
   - limitations of binary safety evaluation

3. **Attribution and interpretability**
   - attention
   - gradients
   - integrated gradients
   - token attribution
   - rationale extraction
   - explanation faithfulness

4. **Counterfactual evaluation**
   - deletion/insertion
   - necessity/sufficiency
   - counterfactual rationales
   - causal mediation style framing

5. **Multi-turn dialogue safety**
   - conversational escalation
   - distributed intent
   - context accumulation

### Positioning against prior work

Prior work usually asks:
> Did the model refuse or comply?

GuardLens asks:
> Which user tokens/spans/turns causally contributed to the guardrail failure?

Prior work often uses:
- classification labels
- refusal rate
- attack success rate

GuardLens adds:
- token-level causal spans
- pivot/distributed turn labels
- tiered supervision
- benign boundary stress
- deconfounded specificity evaluation

### Surface-risk positioning

Surface-risk baselines are not enough because:

- they remove lexical cues effectively
- but they fail on benign high-risk vocabulary
- they cannot distinguish adversarial intent from safety/research context
- they have high boundary FPR

Use the phrase:
> lexical-oracle stress baseline

Not:
> weak baseline

---

## 7. Evaluation expectations

The paper must show more than high classification accuracy.

### Required evaluation groups

1. **Detection baselines**
   - GuardLens
   - GuardLens-NoFusion
   - GuardLens-NoCF
   - TurnLevel
   - ConversationDeBERTa

2. **Core causal attribution**
   - GuardLens
   - attention
   - integrated gradients
   - grad×input
   - surface-risk
   - random

3. **Specificity / boundary**
   - GuardLens boundary FPR
   - surface-risk boundary FPR
   - surface-risk false-lead FPR
   - SR-injected benign FPR

4. **Attribution utility**
   - `Utility = DD@15 - λ·FPR`
   - headline λ=1.0, boundary FPR
   - optional all-benign utility λ ∈ {0.5, 1.0, 2.0}

5. **Deconfounded evaluation**
   - original
   - SR-neutralized
   - SR-neutralized changed-only
   - noise-equalized
   - combined
   - SR-injected benign

6. **Pivot / distributed causality**
   - pivot-window accuracy
   - causal turn mass
   - avoid exact pivot as main metric

7. **Robustness**
   - paraphrase robustness
   - contextual vs lexical pivots

8. **External**
   - AdvBench/HarmBench
   - MHJ
   - ShieldGemma as secondary/appendix

9. **Human audit**
   - 100 conversations
   - 50 double-annotated if possible
   - report agreement and model performance on human labels

---

## 8. Current result numbers to preserve

### Dataset

Final dataset:
- Total: 1,762
- Train/dev/test: 1,220 / 266 / 276
- Test: 87 adversarial, 189 benign

Train:
- 383 malicious
- 837 benign
- cf_weak + cf_strong: 29 records

### Detection

| Model | F1 | Accuracy | AttrF1 | Threshold |
|---|---:|---:|---:|---:|
| GuardLens | 0.9943 | 0.9964 | 0.8778 | 0.56 |
| NoFusion | 1.0000 | 1.0000 | 0.8810 | 0.20 |
| NoCF | 0.9943 | 0.9964 | 0.8841 | 0.20 |
| TurnLevel | 0.7582 | 0.8659 | 0.0000 | 0.55 |
| ConversationDeBERTa | 0.8324 | 0.8949 | 0.0000 | 0.59 |

ConversationDeBERTa:
- F1 = 0.8323699421965317
- accuracy = 0.894927536231884
- precision = 0.8372093023255814
- recall = 0.8275862068965517
- threshold = 0.59

### Core attribution / utility

Headline:
- GuardLens DD@15 = 0.511
- GuardLens boundary FPR = 0.007
- GuardLens utility = 0.504
- Surface-risk DD@15 = 0.568
- Surface-risk boundary FPR = 0.373
- Surface-risk utility = 0.195

All-benign utility:
- λ=0.5:
  - GuardLens 0.507
  - Surface-risk 0.496
- λ=1.0:
  - GuardLens 0.504
  - Surface-risk 0.425
- λ=2.0:
  - GuardLens 0.496
  - Surface-risk 0.282

### Surface-risk FPR

Surface-risk FPR@0.5:
- all_benign = 0.143
- boundary_rejected = 0.373
- boundary_false_lead_benign = 0.576
- false_lead_benign = 0.667
- validated_benign_twin = 0.224

Boundary:
- GuardLens FPR = 0.0072
- 2/279 false positives

SR-injected benign:
- 189 records
- GuardLens FPR = 0.005 = 1/189
- Surface-risk FPR = 0.947 = 179/189

### Deconfounded DD@15

| Variant | GuardLens | Surface-risk | Random |
|---|---:|---:|---:|
| Original | 0.473 | 0.543 | -0.002 |
| SR-neutralized | 0.449 | 0.534 | -0.003 |
| SR-neutralized changed-only | 0.415 | 0.504 | -0.003 |
| Noise-equalized | 0.504 | 0.502 | -0.002 |
| Combined | 0.484 | 0.491 | -0.002 |

Important:
- SR-neutralization does not break surface-risk.
- Noise-equalized makes GuardLens roughly match/slightly beat surface-risk.
- Combined narrows gap to nearly zero.
- Specificity is the strongest distinction.

### Pivot / turn metrics

Causal turn mass:
- mean = 0.310
- median = 0.307
- per-role:
  - adaptation mean = 0.0548, n=352
  - escalation mean = 0.0523, n=454
  - payload mean = 0.0512, n=62
  - setup mean = 0.0484, n=322

Pivot window:
- exact = 0.092 = 8/87
- ±1 = 0.241 = 21/87
- ±2 = 0.379 = 33/87
- ±3 = 0.483 = 42/87
- ±5 = 0.678 = 59/87
- mean distance = 5.1
- median = 4.0
- p75 = 7.0
- p90 = 13.4

### Paraphrase robustness

Contextual:
- turn correlation 0.983
- token correlation 0.958
- top-15 stability 0.692
- pivot stability 0.500

Lexical:
- turn correlation 0.982
- token correlation 0.957
- top-15 stability 0.670
- pivot stability 0.444

### Cross dataset

AdvBench/HarmBench:
- GuardLens:
  - accuracy 0.652
  - precision 0.992
  - recall 0.437
  - F1 0.607
- NoCF:
  - F1 0.556
- ConversationDeBERTa:
  - F1 0.082

### MHJ

- Classification F1 = 0.0614
- Recall = 0.0317
- Detected-case attribution:
  - GuardLens DD@15 = 0.4949
  - Surface-risk DD@15 = 0.2038
  - Random DD@15 = 0.0347
  - GuardLens Flip@15 = 0.6327
  - Surface-risk Flip@15 = 0.2857

### ShieldGemma transfer

Use only secondary/appendix.

| Subset | GuardLens | Surface-risk | Usable n |
|---|---:|---:|---:|
| transfer_success | 0.909 | 0.636 | 11 |
| lexical_pivot | 0.789 | 0.737 | 38 |
| contextual_pivot | 0.789 | 0.737 | 19 |

---

## 9. Figures and tables planned

### Table 1: Dataset statistics

Include:
- total records
- train/dev/test
- malicious/benign
- supervision tiers
- transfer tiers
- families
- benign pool / boundary pool

### Table 2: Detection baselines

Columns:
- Model
- Accuracy
- Precision
- Recall
- F1
- Threshold

Rows:
- GuardLens
- NoFusion
- NoCF
- TurnLevel
- ConversationDeBERTa

Caveat:
- Do not overclaim NoFusion/NoCF.

### Table 3: Core causal attribution

Columns:
- Method
- DD@15
- Flip@15
- Necessity@15
- Trigger size
- Token/span precision if fixed

Rows:
- GuardLens
- Attention
- IG
- Grad×Input
- Surface-risk
- Random

Caveat:
- Avoid random token F1 unless fixed.

### Table 4: Attribution utility / causal specificity

Main table:

| Method | DD@15 | Boundary FPR | Utility |
|---|---:|---:|---:|
| GuardLens | 0.511 | 0.007 | 0.504 |
| Surface-risk | 0.568 | 0.373 | 0.195 |

Optional rows:
- Grad×Input
- IG
- Attention
- Random
But only GuardLens vs Surface-risk has meaningful method-specific FPR.

### Table 5: Deconfounded evaluation

| Variant | GuardLens DD@15 | Surface-risk DD@15 | Gap |
|---|---:|---:|---:|
| Original | 0.473 | 0.543 | -0.070 |
| SR-neutralized | 0.449 | 0.534 | -0.085 |
| Noise-equalized | 0.504 | 0.502 | +0.002 |
| Combined | 0.484 | 0.491 | -0.007 |
| SR-injected benign | FPR 0.005 | FPR 0.947 | huge |

### Table 6: Pivot / distributed causality

| Metric | Result |
|---|---:|
| exact pivot | 0.092 |
| within ±1 | 0.241 |
| within ±2 | 0.379 |
| within ±3 | 0.483 |
| within ±5 | 0.678 |
| causal turn mass mean | 0.310 |

### Table 7: Robustness / external

Could combine:
- paraphrase robustness
- cross-dataset
- MHJ detected-case attribution

### Table 8: Human audit

Pending:
- classification κ
- pivot agreement
- span overlap F1
- GuardLens vs surface-risk on human labels

### Figures

1. **Pipeline diagram**
   - generation → validation → causal span annotation → training → evaluation
2. **Multi-turn attribution example**
   - setup/escalation/payload highlighted
3. **Utility tradeoff plot**
   - x-axis FPR, y-axis DD@15
   - GuardLens better tradeoff than surface-risk
4. **Surface-risk injected benign example**
   - same benign conversation with risk phrase injection
   - surface-risk fires, GuardLens remains safe
5. **Pivot-window curve**
   - window size vs accuracy
6. **Deconfounded DD bar plot**
   - original, neutralized, noisy, combined

---

## 10. Writing constraints

### Style constraints

- Be honest and measured.
- Do not oversell.
- Avoid unsupported claims.
- Use precise caveats.
- Prefer “we find” / “we observe” over absolute claims.
- Avoid em dashes if writing in the user's preferred style.

### Page constraints

Likely 8-page main + references depending on EMNLP/ARR format.
Main tables must be compact.

Prioritize main paper:
1. dataset stats
2. detection baseline table
3. core attribution table
4. attribution utility / specificity table
5. deconfounded table
6. human audit table if done

Push to appendix:
- ShieldGemma transfer
- full threshold sweeps
- all per-tier/per-family breakdowns
- detailed qualitative examples
- long implementation details
- full prompt templates

### Writing tone

The paper should read as:

> rigorous and self-critical

not:

> we beat everything

This matters because reviewers will notice surface-risk and ablation caveats.

---

## 11. Phrases or claims to avoid

Avoid:

1. “GuardLens outperforms all baselines.”
   - False because surface-risk wins raw deletion.

2. “Counterfactual training improves attribution.”
   - Not supported. Phase 3 ran but did not improve best checkpoint.

3. “The fusion module is essential.”
   - Not supported by NoFusion.

4. “NoCF is worse.”
   - Not supported.

5. “Pivot localization is accurate.”
   - Exact pivot is low.

6. “MHJ generalization is strong.”
   - Detection recall is weak.

7. “Surface-risk is a weak baseline.”
   - It is strong on deletion.

8. “Surface-risk only works because exact phrases are present.”
   - Neutralization did not break it. More accurate: surface-risk exploits lexical/structural risk cues and lacks specificity.

9. “Random token F1.”
   - Do not report unless metric is fixed.

10. “Fully causal ground truth.”
   - Better: “counterfactually validated / tiered causal supervision.”

11. “Human-level attribution.”
   - Only after human audit, and still phrase carefully.

### Safer alternative phrases

Use:
- “causal attribution framework”
- “tiered causal supervision”
- “specificity-aware attribution”
- “lexical-oracle stress baseline”
- “deletion-effective but non-specific”
- “distributed conversational causality”
- “validated benign controls”
- “counterfactual quality-control signal”
- “causal-region localization”
- “boundary false-positive rate”

---

## 12. Strongest version of the paper story

The strongest paper story is:

> Multi-turn jailbreaks often emerge through distributed conversational context rather than a single malicious prompt. Existing safety evaluations usually measure only whether a model refuses or complies, and post-hoc attribution baselines can be misled by surface-risk words. We introduce GuardLens, a framework for constructing and evaluating token-level causal attribution in multi-turn guardrail failures. Our pipeline generates interactive adversarial conversations, validates them across models, annotates causal spans with tiered LLM and counterfactual evidence, and constructs separately validated benign and boundary sets. We show that lexical surface-risk heuristics are strong raw deletion baselines but fail causal specificity: they achieve high DD@15 yet falsely fire on 37.3% of boundary benign records and 94.7% of benign records injected with safe high-risk vocabulary. GuardLens achieves a better causal-specificity tradeoff, with DD@15 0.511, boundary FPR 0.007, and attribution utility 0.504 versus 0.195 for surface-risk. It also strongly outperforms turn-level and flat DeBERTa baselines for detection and remains robust under paraphrase and deconfounded stress tests. These results suggest that reliable guardrail-failure attribution requires evaluating not only deletion effectiveness, but also benign specificity and distributed multi-turn causality.

This is the core EMNLP pitch.

---

## 13. Suggested abstract skeleton

Use this as a draft basis, not final text:

> Multi-turn jailbreaks can induce unsafe LLM behavior through gradual contextual escalation, making it difficult to identify which user tokens or turns caused a guardrail failure. We introduce GuardLens, a framework for token-level causal attribution in multi-turn adversarial conversations. GuardLens combines interactive adversarial generation, cross-model behavioral validation, tiered causal supervision, and a multi-turn attribution model trained to identify causal user spans. We evaluate attribution with deletion-based metrics, paraphrase robustness, transfer-tier analysis, boundary benign stress tests, and a new causal-specificity utility that penalizes false attribution on benign conversations. While lexical surface-risk heuristics achieve strong raw deletion performance, they overfire on benign high-risk contexts, producing 37.3% boundary FPR and 94.7% FPR on surface-risk-injected benign examples. GuardLens achieves a stronger attribution-specificity tradeoff, with DD@15 of 0.511, boundary FPR of 0.007, and utility of 0.504 versus 0.195 for surface-risk. Our results show that multi-turn guardrail attribution requires distinguishing causal adversarial evidence from benign surface-risk vocabulary.

---

## 14. Suggested section/subsection structure

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

## 15. Reviewer response prep

### If reviewer says: “Surface-risk beats your model.”

Response:
> Correct on raw deletion. That is why we include surface-risk as a lexical-oracle stress baseline and introduce attribution utility. Surface-risk achieves high deletion by targeting compact lexical cues but has poor specificity, with 37.3% boundary FPR and 94.7% FPR on SR-injected benign examples. GuardLens achieves much stronger utility once false positives are considered.

### If reviewer says: “Synthetic data.”

Response:
> We mitigate this with cross-model validation, external AdvBench/HarmBench and MHJ stress tests, validated benign controls, deconfounded artifact tests, and a human-audited benchmark excluded from training/model selection.

### If reviewer says: “Ablations weak.”

Response:
> The ablations show that the strongest contribution is not an isolated architecture component but the framework: dataset construction, tiered supervision, and specificity-aware evaluation. We report NoCF/NoFusion honestly and do not claim those components independently drive gains.

### If reviewer says: “Pivot accuracy low.”

Response:
> Exact pivot is ill-suited to distributed multi-turn causality. We report windowed pivot localization and causal-turn mass; GuardLens localizes causal regions within ±5 user turns for 67.8% of adversarial test conversations.

### If reviewer says: “MHJ weak.”

Response:
> MHJ is an external stress test. GuardLens has low detection recall there, which we report as a limitation. However, on detected MHJ failures, GuardLens attributions are substantially more causally effective than surface-risk and random.

---

## 16. Final writing recommendation

The final paper should be a **dataset + methodology + evaluation** paper, not primarily an architecture paper.

Winning framing:

> Raw deletion alone is an incomplete attribution metric because it rewards aggressive keyword removal. GuardLens shifts evaluation toward causal specificity, where the goal is to identify adversarial evidence without hallucinating danger from benign high-risk vocabulary.

This directly turns the project’s largest weakness into the paper’s strongest insight.
