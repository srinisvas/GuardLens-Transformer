# GuardLens Project Context Pack

Portable handoff for continuing the GuardLens / NN Token Attribution work in a new ChatGPT or Claude chat.

Last updated after the latest deconfounded evaluation results.

---

## 1. Project goal and current research direction

The project is **GuardLens: token-level and turn-level attribution of multi-turn LLM guardrail failures**.

The intended venue is **EMNLP Main Track / ACL Rolling Review**, not Industry Track, unless the paper is reframed as a deployment/system report. Current recommendation is **EMNLP Main** because the contribution is methodological: dataset construction, causal attribution supervision, and evaluation for multi-turn guardrail failures.

The research direction has evolved from “build a jailbreak detector” to:

> Develop a framework for identifying which user-provided tokens, spans, and conversational regions causally contribute to guardrail failures in multi-turn adversarial interactions.

The core novelty is not just classification. The paper should be framed around:

1. Interactive multi-turn adversarial data generation
2. Cross-model behavioral validation
3. Tiered causal supervision
4. Counterfactual / LLM-confirmed attribution labels
5. Validated benign and boundary stress sets
6. Attribution evaluation beyond raw deletion, especially causal-specificity utility
7. Analysis showing surface-risk heuristics are strong but brittle

The strongest current paper story:

> Surface-risk heuristics can look strong under raw deletion metrics because many adversarial prompts contain compact lexical cues. However, they lack causal specificity and catastrophically overfire on benign or boundary conversations containing safe high-risk vocabulary. GuardLens offers a better attribution-specificity tradeoff: slightly lower raw deletion than surface-risk, but far stronger label alignment, low false positives, paraphrase stability, and robustness under deconfounded stress tests.

---

## 2. Core problem being solved

Modern multi-turn jailbreaks often do not contain one obvious malicious prompt. Instead, adversarial intent can be built through:

- benign setup
- roleplay or fictional framing
- academic framing
- perspective shifts
- authority appeals
- gradual normalization
- context injection
- task decomposition
- lexical payload turns
- distributed escalation over many user turns

The target problem is:

> Given a multi-turn conversation, identify whether it is adversarial and attribute the guardrail failure to specific user tokens/spans/turns that causally contribute to the unsafe request.

The focus is **attribution**, not just detection. Classification is a supporting task.

Important distinction:

- A conversation can contain words like `attack`, `exploit`, `malware`, `jailbreak`, etc. and still be benign.
- Surface-risk cues alone are insufficient because they cause high false positives in research, policy, safety education, or red-team training contexts.
- The model must distinguish **causal adversarial evidence** from **benign surface-risk vocabulary**.

---

## 3. Current methodology / pipeline

### 3.1 Data generation strategy

Current dataset generation is **interactive**, not static.

Final generation architecture:

- **Generator**: `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-14B-Instruct`
- **Target model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Validator / causal analyzer**: `mistralai/Mistral-7B-Instruct-v0.3`
- Additional transfer validation with Qwen and Mistral.
- ShieldGemma used only as secondary external evaluator.

Key decision:

- Static generation and template augmentation were dropped.
- Interactive generation with target feedback is the main data source.
- Augmentation was removed for interactive data because template hard negatives diluted quality and were mostly removed by dedup.

Earlier run showed augmentation problem:

- Raw interactive: 100 records
- Augmented: 330 records
- Post-dedup: 113 records
- Dedup removed 217 records
- Conclusion: augmentation added mostly low-value template filler.

### 3.2 Generation model choice

Originally tested `Qwen/Qwen2.5-14B-Instruct` as generator against Llama-8B target. It was slower and not clearly better.

#### Qwen-7B interactive dry run

- 50 pairs
- 100 records
- 50 malicious / 50 benign
- Llama target jailbreak rate: **19/50 = 38.0%**
- Avg turns malicious: **29.3**
- Total adaptations: **242**
- Stronger than 14B in rate during this sample.

#### Qwen-14B generation

- 200 pairs
- 400 records
- 200 malicious / 200 benign
- Llama target jailbreak rate: **54/200 = 27.0%**
- Avg turns malicious: **23.7**
- Total adaptations: **392**

Conclusion:

- 7B + feedback loop is sufficient and faster.
- 14B adds diversity but not necessarily better jailbreak rate.
- Final data combines 7B and 14B generation.

### 3.3 Validation pipeline

For each generated adversarial record:

- Llama target response comes from generation.
- Mistral validation checks whether conversation jailbreaks another model family.
- Qwen validation was also used for transfer-tier construction earlier.
- Records categorized by transfer behavior:
  - `transfer_success`: jailbreaks target and validator
  - `target_only`: jailbreaks Llama target only
  - `cross_only`: jailbreaks validator but not target
  - `no_jailbreak`
  - `benign`

### 3.4 Causal analysis pipeline

A 4-pass causal analysis was run using Mistral:

1. **Pass 1: LLM span annotation**
   - Annotates adversarial spans in jailbreak records.
   - Needed because template keyword span annotation failed on natural interactive phrasing.

2. **Pass 2: Pivot-turn counterfactual**
   - Replays conversation with pivot removed/neutralized.
   - Classifies turn as `cf_turn_strong`, `cf_turn_weak`, or `distributed_or_unclear`.

3. **Pass 3: Span-level counterfactual**
   - Ablates annotated spans and replays through validator.
   - Marks spans causal or incidental.

4. **Pass 4: Negative-control validation**
   - Tests benign or non-causal spans to identify false-negative controls / surprise causal spans.

Important issue fixed:

- Causal replay initially failed for conversations with consecutive user turns because Mistral chat template expects strict user/assistant alternation.
- Fixed by normalizing/rebuilding replay sequence.

Fresh causal run stats before postprocessing:

- Input: **1,500 records**
- Supervision tiers:
  - construction: **1204**
  - llm_confirmed: **255**
  - cf_weak: **30**
  - cf_strong: **11**
- Pivot-turn CF:
  - none: **487**
  - distributed_or_unclear: **223**
  - cf_turn_weak: **28**
  - cf_turn_strong: **12**
- Spans:
  - **62 causal**
  - **146 incidental**
  - **4333 unvalidated**
- Training eligible: **1041/1500**
- Avg loss weight: **0.583**

---

## 4. Dataset details

### 4.1 Final dataset

Final dataset after malicious postprocess + clean benign pool merge:

- Total records: **1,762**
- Train/dev/test split:
  - Train: **1,220**
  - Dev: **266**
  - Test: **276**

#### Train

- Records: **1,220**
- Labels:
  - malicious: **383**
  - benign: **837**
- Training eligible: **1,220**
- Supervision tiers:
  - `llm_confirmed`: **237**
  - `construction`: **452**
  - `cf_weak`: **22**
  - `cf_strong`: **7**
  - `benign_validated`: **502**
- Pivot kinds:
  - contextual_pivot: **213**
  - none: **844**
  - lexical_pivot: **160**
  - misleading_decoy: **3**
- Top families:
  - interactive_adversarial: **383**
  - interactive_benign_twin: **335**
  - clean_everyday: **206**
  - research_technical: **132**
  - topic_matched_safe: **72**

#### Dev

- Records: **266**
- Labels:
  - malicious: **79**
  - benign: **187**
- Supervision tiers:
  - `llm_confirmed`: **54**
  - `construction`: **100**
  - `cf_strong`: **2**
  - `cf_weak`: **4**
  - `benign_validated`: **106**

#### Test

- Records: **276**
- Labels:
  - malicious: **87**
  - benign: **189**
- Supervision tiers:
  - `llm_confirmed`: **52**
  - `construction`: **105**
  - `cf_weak`: **4**
  - `cf_strong`: **2**
  - `benign_validated`: **113**
- Pivot kinds:
  - contextual_pivot: **48**
  - lexical_pivot: **36**
  - misleading_decoy: **1**
  - none: **191**

### 4.2 Benign pool

Separate benign generation/validation was added to avoid relying on template augmentation.

Files:

- `benign_pool_v11_llama_validated.jsonl`
- `benign_pool_v11_mistral_validated.jsonl`
- Combined into:
  - `benign_clean.jsonl`
  - `benign_boundary.jsonl`

Filtering rule:

- Clean benign kept only if accepted by both Llama and Mistral validation:
  - no jailbreak detected by either validator
  - `benign_status = clean_benign`
  - `validation_status = validated`
  - `training_eligible = True`

Boundary rejected benign:

- If either validator flagged it, record marked:
  - `benign_status = benign_boundary_rejected`
  - `validation_status = rejected`
  - `training_eligible = False`
  - includes `rejected_by`

Boundary rejected records are used for stress testing, not training.

### 4.3 Families / categories

Important families in final dataset:

- `interactive_adversarial`
- `interactive_benign_twin`
- `clean_everyday`
- `research_technical`
- `topic_matched_safe`
- `hard_benign`
- `false_lead_benign`

Important benign statuses:

- `clean_benign`
- `validated_benign_twin`
- `benign_boundary_rejected`

### 4.4 Transfer tiers

Transfer tiers used for evaluation:

- `transfer_success`
- `target_only`
- `cross_only`
- `no_jailbreak`
- `benign`

Merged 7B results before final postprocess:

- Total: **1100**
- 550 malicious / 550 benign
- Transfer tiers:
  - benign: **550**
  - transfer_success: **227**
  - cross_only: **167**
  - no_jailbreak: **132**
  - target_only: **24**
- Jailbreak success malicious only:
  - Llama-8B target: **251/550 = 45.6%**
  - Qwen-7B transfer: **394/550 = 71.6%**
  - Both: **227/550 = 41.3%**
  - Any validator: **418/550 = 76.0%**
- Benign false alarms:
  - rejected: **199/550 = 36.2%**

Mistral CF validation on 7B:

- Total records: **1100**
- Validation status:
  - validated: **727**
  - ambiguous: **158**
  - rejected: **215**
- Mistral jailbreak rate: **392/550 = 71.3%**
- False alarm rate: **215/550 = 39.1%**
- Cross-model transfer from Llama jailbreaks:
  - Llama jailbreaks: **251**
  - Mistral jailbreaks: **392**
  - Both: **221**
  - Transfer rate: **221/251 = 88.0%**

### 4.5 Supervision tiers

Supervision tiers and intended loss weighting:

- `cf_strong`: high-confidence counterfactual causal evidence
- `cf_weak`: weaker counterfactual evidence
- `llm_confirmed`: LLM-confirmed adversarial spans
- `construction`: spans from construction or weaker evidence
- `benign_validated`: validated benign records
- `incidental`: confirmed non-causal spans, should be negative attribution supervision
- `ignore`: excluded / unannotated

Training config span tier weights:

```python
span_tier_weights = {
    "cf_strong": 1.00,
    "cf_weak": 0.70,
    "llm_confirmed": 0.60,
    "construction": 0.40,
    "llm_only": 0.25,
    "incidental": 1.00,
    "ignore": 0.00,
}
```

Important interpretation:

- CF labels are sparse: in train only **29 cf_weak/cf_strong records out of 1,220**.
- Phase 2 already incorporates CF evidence through tier weighting.
- Phase 3 CF-specific training ran but did not improve the selected checkpoint.

---

## 5. Model / architecture details

Primary model: **GuardLens**

Backbone:

- `microsoft/deberta-v3-base`
- Frozen backbone by default
- Trainable parameters around **2.17M**
- Total model around **186M**

Input representation:

- Multi-turn conversation
- `max_turns = 32`
- `max_tokens_per_turn = 192`
- `max_total_tokens = 2048` for flat baseline

Core architecture components:

- Turn-level encoding from DeBERTa
- Cross-turn transformer / attention layers
- Token-level attribution head
- Classification head
- Optional pivot head
- Optional gated fusion between attribution features and classification

Important config values:

```python
backbone_name = "microsoft/deberta-v3-base"
backbone_dim = 768
freeze_backbone = True

cross_turn_layers = 2
cross_turn_heads = 8
cross_turn_dim = 256
cross_turn_dropout = 0.1

cls_hidden_dim = 256
attr_hidden_dim = 128
n_classes = 2

use_pivot_head = True
n_pivot_kinds = 5

max_turns = 32
max_tokens_per_turn = 192
max_total_tokens = 2048

learning_rate = 2e-4
weight_decay = 0.01
warmup_steps = 200
batch_size = 4
gradient_accumulation = 4
max_grad_norm = 1.0
```

Loss terms:

```python
lambda_cls = 0.2
lambda_attr = 1.0
lambda_cf = 0.5
lambda_pivot = 0.3
```

Training phases:

```python
phase1_epochs = 5     # classification bootstrap
phase2_epochs = 15    # attribution/tier supervision
phase3_epochs = 5     # CF consistency
```

Important ablation models:

- `guardlens`
- `guardlens_no_fusion`
- `guardlens_no_cf`
- `turn_level`
- `conversation_deberta`

Critical interpretation:

- NoFusion and NoCF are not worse in current results.
- Do **not** claim fusion or CF phase independently improves detection.
- Current gains are best framed as coming from:
  - dataset construction
  - tiered supervision
  - structured multi-turn modeling
  - attribution-specific evaluation

---

## 6. Training setup

Training command pattern:

```bash
python -m guardlens.train \
  --train-path splits/train.jsonl \
  --dev-path splits/dev.jsonl \
  --test-path splits/test.jsonl \
  --output ./checkpoints \
  --model guardlens
```

Important training fixes already made:

1. Trainer now loads pre-split files directly.
2. No internal re-splitting when `train_path`, `dev_path`, `test_path` are supplied.
3. `max_turns` increased to **32**.
4. `max_tokens_per_turn` increased to **192**.
5. Batch size reduced to **4**, gradient accumulation **4**.
6. `causal_type == "causal"` is primary attribution signal, not just label names.
7. Incidental spans are explicit negative supervision.
8. `loss_weight` / supervision tier weights are used.
9. `pos_weight` auto-computed from training class balance.
10. CF oversampling enabled by default:
    - `oversample_cf = True`
    - `cf_oversample_factor = 3`

Class balance train:

- 383 positive
- 837 negative
- `pos_weight = 2.19`

Training run log `train_all_24224.out`:

For `guardlens`:

- 25 epochs configured.
- Phase 1: epochs 0-4
- Phase 2: epochs 5-19
- Phase 3: epochs 20-24
- Phase 3 **did run**:
  - Ep 20 P3
  - Ep 21 P3
  - Ep 22 P3
  - Ep 23 P3
  - Ep 24 P3
- However, best attribution checkpoint remained from **epoch 13 phase 2**.
- Phase 3 did not improve dev attribution F1 enough to replace Phase 2 checkpoint.

GuardLens test eval from training log:

- Checkpoint: `best_attribution.pt`
- Loaded from epoch 13, phase 2, threshold **0.56**
- Accuracy: **0.9964**
- Precision: **0.9886**
- Recall: **1.0000**
- F1: **0.9943**
- Attr F1: **0.8778**
- Pivot Accuracy: **0.7536** in training summary, but causal eval exact pivot is much lower because metrics differ.

Ablation training summary:

```text
Model                     | F1       Acc      AttrF1   PivAcc   Thresh
guardlens                 | 0.9943   0.9964   0.8778   0.7536   0.56
guardlens_no_fusion       | 1.0000   1.0000   0.8810   0.7536   0.20
guardlens_no_cf           | 0.9943   0.9964   0.8841   0.7645   0.20
turn_level                | 0.7582   0.8659   0.0000   0.0000   0.55
conversation_deberta      | 0.8324   0.8949   0.0000   0.0000   0.59
```

ConversationDeBERTa fix:

- Evaluation originally failed due to wrong collator shape.
- Fixed by using `FlatConversationCollator` when `model_name == "conversation_deberta"`.
- ConversationDeBERTa final:
  - F1: **0.8323699421965317**
  - Accuracy: **0.894927536231884**
  - Precision: **0.8372093023255814**
  - Recall: **0.8275862068965517**
  - Threshold: **0.59**

Important interpretation:

- GuardLens beats turn-level and flat DeBERTa baselines.
- NoFusion and NoCF do not underperform GuardLens, so architecture component claims must be careful.

---

## 7. Evaluation setup

Evaluation stack now includes:

1. **Core causal attribution**
   - Methods: GuardLens, attention, integrated gradients, grad×input, surface-risk, random
   - Metrics: DD@k, flip@k, necessity, sufficiency, trigger size, token F1
   - k values: 5%, 10%, 15%, 20%

2. **Per-supervision-tier attribution**
   - cf_strong, cf_weak, llm_confirmed, construction, benign_validated
   - Report n per tier due to sparse CF.

3. **Per-transfer-tier attribution**
   - transfer_success
   - target_only
   - cross_only
   - contextual_pivot
   - lexical_pivot

4. **Classification baselines**
   - GuardLens
   - NoFusion
   - NoCF
   - TurnLevel
   - ConversationDeBERTa

5. **Surface-risk FPR comparison**
   - Thresholds: 0.3, 0.5, etc.
   - Subsets:
     - all_benign
     - boundary_rejected
     - boundary_false_lead_benign
     - false_lead_benign
     - validated_benign_twin
     - hard_benign
     - research_technical

6. **Boundary stress test**
   - 279 rejected benign records
   - Uses `best_detection.pt`

7. **Implicit/explicit subset analysis**
   - Contextual pivot vs lexical pivot

8. **Paraphrase robustness**
   - full
   - contextual
   - lexical

9. **Attribution precision + minimality curve**

10. **Cross-dataset generalization**
    - AdvBench + HarmBench
    - 3 models:
      - GuardLens
      - GuardLens-NoCF
      - ConversationDeBERTa

11. **Cross-model transfer**
    - ShieldGemma
    - subsets:
      - transfer_success
      - lexical_pivot
      - contextual_pivot
    - threshold sensitivity
    - Use only as secondary evidence.

12. **MHJ external eval**
    - Classification + causal attribution
    - MHJ is external multi-turn test, not training.
    - Keep separate from internal split.

13. **Attribution Utility**
    - `Utility = DD@15 - λ * FPR`
    - Headline uses boundary FPR, λ=1.0.
    - Also compute all-benign utilities with λ = 0.5, 1.0, 2.0.

14. **Deconfounded evaluation**
    Variants:
    - Original
    - SR-neutralized adversarial
    - SR-neutralized changed-only
    - Noise-equalized adversarial
    - SR-injected benign
    - Combined neutralized + noisy

15. **Causal Turn Mass**
    - Attribution mass on causal semantic turn roles / total mass
    - Better than exact pivot for distributed causality.

16. **Pivot Window Accuracy**
    - within 0, 1, 2, 3, 5 user turns
    - Replaces over-harsh exact pivot claim.

---

## 8. Current results and metrics

### 8.1 Detection results

| Model | F1 | Accuracy | Attr F1 | Pivot Acc | Threshold |
|---|---:|---:|---:|---:|---:|
| GuardLens | 0.9943 | 0.9964 | 0.8778 | 0.7536 | 0.56 |
| GuardLens-NoFusion | 1.0000 | 1.0000 | 0.8810 | 0.7536 | 0.20 |
| GuardLens-NoCF | 0.9943 | 0.9964 | 0.8841 | 0.7645 | 0.20 |
| Turn-level | 0.7582 | 0.8659 | 0.0000 | 0.0000 | 0.55 |
| ConversationDeBERTa | 0.8324 | 0.8949 | 0.0000 | 0.0000 | 0.59 |

Important:

- GuardLens beats turn-level and flat DeBERTa.
- NoFusion/NoCF are not worse, so do not claim they are necessary.

### 8.2 Core causal attribution

Core causal attribution, full test, DD@15:

| Method | DD@15 | Flip@15 | Token F1 |
|---|---:|---:|---:|
| GuardLens | about 0.511 in full causal summary / 0.473 in deconfounded original self-run | about 0.494 | 0.8778 |
| Surface-risk | about 0.568 full / 0.543 deconfounded original | about 0.563 | 0.0583 |
| Attention | 0.058 | 0.046 | interpret carefully |
| Integrated Gradients | 0.297 | 0.276 | 0.002 |
| Grad×Input | 0.361 | 0.333 | 0.003 |
| Random | ~0 or negative | 0 | suspicious token F1 around 0.61, do not report |

Surface-risk beats GuardLens on raw deletion, but has terrible FPR on benign/boundary examples.

### 8.3 Attribution utility

Latest deconfounded/utility run `eval_deconfound_24495.out`:

Headline utility with boundary FPR, λ=1.0:

| Method | DD@15 | Boundary FPR | Utility |
|---|---:|---:|---:|
| GuardLens | **0.511** | **0.007** | **0.504** |
| Surface-risk | 0.568 | 0.373 | 0.195 |

All-benign utility:

λ=0.5:
- GuardLens: DD 0.511, FPR 0.007, Utility **0.507**
- Surface-risk: DD 0.568, FPR 0.143, Utility **0.496**
- Grad×Input: 0.361
- IG: 0.297
- Attention: 0.058
- Random: -0.003

λ=1.0:
- GuardLens: **0.504**
- Surface-risk: **0.425**
- Grad×Input: 0.361
- IG: 0.297

λ=2.0:
- GuardLens: **0.496**
- Surface-risk: **0.282**

This is one of the strongest results. Use in main paper.

### 8.4 Boundary stress test

Boundary rejected benign set:

- **279 records**
- GuardLens:
  - Accuracy: **0.9928**
  - FPR: **0.0072**
  - False positives: **2/279**
- Earlier boundary run had FPR 0.043, later improved with best_detection checkpoint to 0.0072.

Surface-risk boundary FPR at threshold 0.5:

- **0.3728**

### 8.5 Surface-risk FPR comparison

Surface-risk FPR@0.5:

- all benign: **0.143**
- boundary rejected: **0.373**
- boundary false-lead benign: **0.576**
- false-lead benign: **0.667**
- validated benign twin: **0.224**

New SR-injected benign:

- injected benign records: **189**
- GuardLens FPR: **0.005 = 1/189**
- Surface-risk FPR: **0.947 = 179/189**

This is extremely strong evidence:

> Surface-risk collapses under safe high-risk vocabulary injection; GuardLens remains specific.

### 8.6 Deconfounded evaluation

Latest run:

Original adversarial surface risk:

- 87 adversarial test records
- mean SR: **0.750**
- max: **1.000**
- >0.3: **80/87**

SR-neutralized:

- **81/87** records had SR phrases replaced
- Neutralized SR mean: **0.000**
- >0.3: **0/87**
- Sanity: **87/87 neutralized records still detected adversarial**

Deconfounded DD@15 table:

| Variant | GuardLens DD@15 | Surface-risk DD@15 | Random DD@15 |
|---|---:|---:|---:|
| Original | 0.473 | **0.543** | -0.002 |
| SR-neutralized | 0.449 | **0.534** | -0.003 |
| SR-neutralized changed-only | 0.415 | **0.504** | -0.003 |
| Noise-equalized | **0.504** | 0.502 | -0.002 |
| Combined | 0.484 | **0.491** | -0.002 |

Interpretation:

- Surface-risk still wins raw deletion on original and neutralized data.
- Neutralization alone does **not** destroy surface-risk, so the surface-risk advantage is not just exact phrase matching.
- Under noise-equalized data, GuardLens matches/slightly beats surface-risk.
- Under combined deconfounding, the gap becomes tiny.
- Most important: SR-injected benign shows surface-risk is catastrophically non-specific.

### 8.7 Causal turn mass and pivot window

Latest fixed results:

Causal turn mass:

- Records: **87**
- Mean: **0.310**
- Median: **0.307**
- Per-role attribution mass:
  - adaptation: mean **0.0548**, n=352
  - escalation: mean **0.0523**, n=454
  - payload: mean **0.0512**, n=62
  - setup: mean **0.0484**, n=322

Pivot window accuracy:

- Records with pivot: **87**
- within_0: **0.092 = 8/87**
- within_1: **0.241 = 21/87**
- within_2: **0.379 = 33/87**
- within_3: **0.483 = 42/87**
- within_5: **0.678 = 59/87**
- Distance:
  - mean **5.1**
  - median **4.0**
  - p75 **7.0**
  - p90 **13.4**

Interpretation:

- Exact pivot remains low.
- Windowed pivot supports distributed-causality framing.
- Within ±5 user turns = 67.8% is usable.
- Causal turn mass is moderate, not headline-dominant.

### 8.8 Paraphrase robustness

Earlier result:

- Contextual:
  - Turn rank correlation: **0.983**
  - Token rank correlation: **0.958**
  - Top-15 stability: **0.692**
  - Pivot stability: **0.500**
- Lexical:
  - Turn rank correlation: **0.982**
  - Token rank correlation: **0.957**
  - Top-15 stability: **0.670**
  - Pivot stability: **0.444**

Interpretation:

- Attribution stable under paraphrase.
- Pivot stability moderate, not headline.

### 8.9 Cross-dataset generalization

AdvBench + HarmBench extension:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| GuardLens | 0.652 | 0.992 | 0.437 | 0.607 |
| GuardLens-NoCF | 0.622 | 0.992 | 0.387 | 0.556 |
| ConversationDeBERTa | 0.174 | 0.129 | 0.060 | 0.082 |

Interpretation:

- GuardLens is conservative OOD detector.
- High precision, lower recall.
- GuardLens > NoCF and flat baseline.
- Useful but not main headline.

### 8.10 ShieldGemma transfer

Stable transfer results:

| Subset | GuardLens | Surface-risk | Random | Usable n |
|---|---:|---:|---:|---:|
| transfer_success | 0.909 | 0.636 | 0.909 | 11 |
| lexical_pivot | 0.789 | 0.737 | 0.763 | 38 |
| contextual_pivot | 0.789 | 0.737 | 0.789 | 19 |
| threshold 0.3 | 0.714 | 0.524 | 0.667 | 21 |
| threshold 0.4 | 0.917 | 0.500 | 0.833 | 12 |
| threshold 0.5 | 0.909 | 0.636 | 0.909 | 11 |

Interpretation:

- Directionally useful.
- GuardLens beats surface-risk in ShieldGemma transfer.
- But denominator is small and random sometimes ties.
- Use as appendix / secondary only.

### 8.11 MHJ external eval

MHJ classification:

- F1: **0.0614**
- Recall: **0.0317**
- Weak detection generalization.

MHJ detected-case causal attribution:

- Among detected adversarial cases:
  - GuardLens DD@15: **0.4949**
  - Surface-risk DD@15: **0.2038**
  - Random DD@15: **0.0347**
  - GuardLens Flip@10: **0.5714**
  - GuardLens Flip@15: **0.6327**
  - Surface-risk Flip@15: **0.2857**
  - Random Flip@15: **0.0612**

Interpretation:

- Do not claim MHJ detection is strong.
- Claim: when GuardLens detects MHJ failures, its attributions are more causally effective than surface-risk/random.

---

## 9. Known issues, weaknesses, reviewer risks

### 9.1 Synthetic-heavy dataset

Biggest reviewer risk.

Mitigations:

- human audit benchmark planned
- MHJ external eval
- AdvBench/HarmBench cross-dataset eval
- deconfounded evaluation
- validated benign pool
- boundary stress test
- paraphrase robustness

### 9.2 Surface-risk beats GuardLens on raw deletion

Surface-risk DD@15 > GuardLens DD@15 on internal test.

Explanation:

- raw deletion rewards removing compact lexical cues
- surface-risk acts like a lexical-oracle stress baseline
- dataset contains many high-surface-risk adversarial phrases
- but surface-risk has terrible specificity

Mitigation:

- Attribution Utility
- boundary FPR
- SR-injected benign FPR
- deconfounded analysis
- label alignment/token F1

Do not claim GuardLens beats surface-risk on raw deletion.

### 9.3 Phase 3 CF training did not improve

Phase 3 ran but best attribution checkpoint remained Phase 2.

Reason:

- CF records sparse:
  - train has 29 cf_weak/cf_strong records out of 1220.
- Phase 2 already uses CF evidence via tier weighting.
- Phase 3 mostly reweights a sparse signal.

Paper wording:

> Phase 3 executed but did not improve the selected attribution checkpoint over Phase 2. At the current scale, counterfactual validation contributes more as quality control and supervision calibration than as an independent training objective. Larger-scale CF annotation may make CF-specific optimization more effective.

### 9.4 NoCF / NoFusion are not worse

Ablations:

- NoFusion F1 and AttrF1 slightly better than GuardLens
- NoCF AttrF1 slightly better

Do not overclaim architecture components.
Frame GuardLens as one instantiation of the dataset/evaluation framework.

### 9.5 Exact pivot accuracy is low

Causal eval exact pivot:

- around 5.7% earlier
- fixed pivot-window exact: **9.2%**

Windowed:

- ±5 = **67.8%**

Explanation:

- multi-turn causality is distributed
- exact pivot is too strict
- pivot labels can be noisy
- model attribution mass may spread across setup/escalation/payload

Use pivot-window and causal turn mass instead of exact pivot.

### 9.6 Random token F1 suspicious

Random token F1 around 0.61 is suspicious.
Likely metric issue due to:

- counting true negatives
- ignored tokens
- imbalance in token labels
- random selecting mostly non-causal tokens

Decision:

- Do **not** report random token F1.
- Random DD/flip is valid near zero and can be reported for deletion only.

### 9.7 MHJ detection weak

MHJ classification recall is low.
Frame as external stress test, not success story.
Use detected-case attribution only.

### 9.8 ShieldGemma underpowered

ShieldGemma dangerous-content policy marks few originals as unsafe.
Use as secondary evidence only.
Do not headline it.

### 9.9 Token F1 on deconfounded variants

Span offsets may be invalid after neutralization/noise.
Use DD/flip/FPR for deconfounded variants.
Treat token F1 there as secondary / not reportable.

---

## 10. Decisions already made

1. Target **EMNLP Main Track**.
2. Do not target Industry Track unless final paper becomes deployment/system-focused.
3. Do not generate 1000 more generic pairs.
4. Use existing 1,762-record dataset.
5. Keep MHJ external only, not in train/dev/test split.
6. Do not re-split current dataset.
7. Remove interactive augmentation permanently.
8. Generate benign pool separately and validate it.
9. Surface-risk should be framed as a strong lexical-oracle / stress baseline, not a fair independent learned method.
10. Do not claim GuardLens dominates surface-risk raw deletion.
11. Use causal-specificity utility as headline metric.
12. Use boundary FPR and SR-injected benign FPR to explain surface-risk brittleness.
13. Use pivot-window and causal-turn-mass instead of exact pivot.
14. Do not rely on NoCF/NoFusion ablations for novelty.
15. Do not report random token F1.
16. Run human audit benchmark to mitigate synthetic-label risk.
17. Use Deconfounded evaluation and Attribution Utility in main paper.
18. Use ShieldGemma transfer only as secondary/appendix.
19. Use MHJ as external stress test, with honest caveat about low recall.
20. Preserve exact numbers in writing, including caveats.

---

## 11. Open questions still unresolved

1. Human audit benchmark results:
   - not yet completed
   - need 100 conversations
   - ideally 50 double-annotated
   - report classification Cohen’s κ
   - pivot exact/±1/distributed agreement
   - span overlap F1

2. Span-level metrics:
   - need causal span precision/recall/F1
   - incidental span FPR
   - top-k span precision
   - token AUPRC
   - random token F1 issue must not pollute reporting

3. Whether to include ShieldGemma in main or appendix:
   - likely appendix only.

4. Whether to include NoFusion/NoCF table:
   - include honestly, but do not claim component superiority.

5. Exact final paper title:
   - candidate direction:
     - “Beyond Surface Risk: Causal Token Attribution for Multi-Turn Guardrail Failures”
     - “GuardLens: Causal Token Attribution for Multi-Turn LLM Guardrail Failures”
   - Avoid title implying GuardLens beats all baselines.

6. Final table ordering:
   - should lead with dataset/evaluation and specificity rather than architecture.

7. Whether to include utility equation:
   - yes:
     - `Utility = DD@15 - λ · FPR`
     - headline λ=1.0 boundary FPR
     - possibly also multiplicative `DD@15 × (1 - FPR)` as sensitivity/appendix.

---

## 12. Important terminology and definitions

### GuardLens

The proposed framework/model for token-level and turn-level attribution of multi-turn guardrail failures.

### Adversarial conversation

A conversation in which the user attempts to elicit harmful, unsafe, restricted, or policy-violating content, including via roleplay, academic framing, fictional framing, evasion, concealment, operational guidance, or gradual escalation.

### Benign conversation

Safe conversation, even if it contains sensitive terms in defensive, educational, policy, or high-level contexts.

### Pivot turn

First user turn where adversarial intent becomes actionable.
Can be:

- exact 0-indexed user turn
- `DISTRIBUTED`
- blank if benign

### Distributed causality

No single turn fully causes the failure. Setup, escalation, roleplay, and payload jointly create adversarial intent.

### Causal span

Specific user-provided word/phrase that contributes to adversarial intent and whose removal/neutralization would make the conversation safer.

### Incidental span

A span that appears suspicious but is not causally responsible, often confirmed by negative control.

### DD@k / deviation drop

Drop in adversarial probability after removing the top-k% attributed tokens.

### Flip@k

Fraction of adversarial examples whose prediction flips to benign after top-k% attributed tokens are removed.

### Necessity

Whether removal of attributed tokens eliminates/decreases adversarial classification.

### Sufficiency

Whether retaining only attributed tokens is sufficient to preserve adversarial classification.

### Boundary FPR

False positive rate on rejected/boundary benign conversations.

### Attribution Utility

Specificity-aware attribution metric:

```text
Utility = DD@15 - λ · FPR
```

Headline:

- λ=1.0
- FPR = boundary FPR

### Causal Turn Mass

Fraction of attribution mass falling on causal semantic roles:

- pivot
- escalation
- payload
- adaptation / adversarial transition

### Pivot Window Accuracy

Whether predicted top-attributed user turn falls within ±W user turns of the annotated pivot.

### Surface-risk baseline

Hardcoded lexical heuristic based on risky phrases. Should be interpreted as a strong lexical-oracle stress baseline, not a fully independent learned attribution method.

---

## 13. File / code names and what each file does

### Dataset generation / validation files

#### `build_semantic_datasetv11.py`

Main interactive dataset generation script.
Uses generator-target feedback loop.
Important constants:

- `_JUDGE_SYSTEM_PROMPT`: exists
- `_BEHAVIOR_TO_COMPLIANCE`: exists
- `_JUDGE_SYSTEM`: does not exist

#### `launch_gen.slurm`
Older generation SLURM.

#### `launch_gen (1).slurm`
Modified generation SLURM.

#### `launch_val.slurm`
Validation SLURM.

#### `merge_validations.py`
Merges interactive validation outputs.
Used for `merged_7b.jsonl`.

#### `postprocess_causal.py`
Postprocesses causal malicious data.
Input: `causal_analyzed_all_final.jsonl`
Output: `malicious_final.jsonl`

#### `split_dataset.py`
Creates train/dev/test and human benchmark splits.
Important:

- supports `--mhj-input`, but MHJ was not included in final internal split.
- Current split created without MHJ:
  - External test: 0
- Current decision: do not re-split with MHJ; keep MHJ separate external eval.

#### `mhj_loader.py`
Processes MHJ dataset into GuardLens schema.
Needs to output: `mhj_external_test.jsonl`
Use MHJ as external multi-turn eval only.

### Data files

#### `semantic_multiturn_v11_interactive_raw.jsonl`
Raw interactive generated data.

#### `semantic_multiturn_v11_interactive_augmented.jsonl`
Deprecated/avoid. Augmentation diluted interactive data.

#### `semantic_multiturn_v11_interactive_augmented_dedup.jsonl`
Deprecated final from augmented path, 113 records. Not main dataset.

#### `semantic_multiturn_v11_interactive_cf_validated.jsonl`
Mistral CF validation output for 7B data.

#### `semantic_multiturn_v11_interactive_14b_raw.jsonl`
14B generated raw data:

- 400 records
- 200 malicious / 200 benign
- 54/200 Llama jailbreaks

#### `semantic_multiturn_v11_interactive_14b_validated.jsonl`
14B data validated with Mistral.

#### `combined_all.jsonl`
Combined 7B + 14B records before final causal postprocessing.
Input to causal analysis.

#### `causal_analyzed_all_final.jsonl`
Final causal-analyzed 1,500 records before postprocessing.

#### `malicious_final.jsonl`
Postprocessed malicious data.

#### `benign_pool_v11_llama_validated.jsonl`
Benign pool validated by Llama.

#### `benign_pool_v11_mistral_validated.jsonl`
Benign pool validated by Mistral.

#### `benign_clean.jsonl`
Clean benign accepted by both validators.

#### `benign_boundary.jsonl`
Rejected/boundary benign for stress testing.

#### `final_dataset.jsonl`
Final merged dataset: malicious_final + benign_clean, 1,762 records.

#### `splits/train.jsonl`
Train split: 1,220 records.

#### `splits/dev.jsonl`
Dev split: 266 records.

#### `splits/test.jsonl`
Test split: 276 records.

### Training files

#### `guardlens/train.py`
Training entry point.
Supports:

- `--train-path`
- `--dev-path`
- `--test-path`
- fallback `--data`
- `--model`
- `--batch-size`
- `--max-turns`
- `--max-tokens`
- `--no-pivot-head`
- `--no-oversample`
- `--no-threshold-tune`

#### `guardlens/config.py`
Model/training config.
Important values listed above.
Updated for v11:

- `max_turns=32`
- `max_tokens_per_turn=192`
- `max_total_tokens=2048`
- causal labels include `CONTEXT_BRIDGE`
- uses tier weights
- pos_weight auto

#### `guardlens/data/dataset.py`
Dataset + collator.
Important classes:

- `GuardLensDataset`
- `GuardLensCollator`
- `FlatConversationCollator`

Known required behavior:

- For v11, attribution labels must use `span["causal_type"] == "causal"` as primary signal.
- Incidental spans must be explicit negatives.
- ConversationDeBERTa must use `FlatConversationCollator`.

#### `guardlens/training/trainer.py`
Training loop and evaluate function.
Important:

- Must support pre-split loading.
- Must pass sample/tier weights.
- Must support flat DeBERTa batches.
- Must save:
  - `best_detection.pt`
  - `best_attribution.pt`
  - `best_phase2.pt` sometimes

#### `guardlens/training/loss.py`
Loss computation.
Important:

- BCE classification loss with pos_weight.
- Attribution loss with tier weights.
- Pivot loss.
- CF loss phase.

#### `guardlens/models/*`
Model registry includes:

- `guardlens`
- `guardlens_no_fusion`
- `guardlens_no_cf`
- `turn_level`
- `conversation_deberta`

#### `train_all.slurm`
Full training suite.
Latest successful run: `train_all_24224.out`.

### Evaluation files

#### `guardlens/evaluate.py`
General classification evaluation entrypoint.
Fix made:

- If `model_name == "conversation_deberta"`, use `FlatConversationCollator`.

Need:

- should write clean JSON with `--output` rather than mixing logs into stdout.

#### `eval_causal.py`
Runs causal attribution evaluation.
Added:

- per-tier eval
- transfer-tier eval
- pivot-kind eval
- LaTeX output

Known issue:

- exact pivot metric is harsh.
- Use pivot-window utility file for better pivot framing.

#### `causal_eval.py`
Core causal deletion engine.
Methods:

- GuardLens attribution
- attention
- integrated gradients
- grad×input
- surface-risk
- random

Known:

- Internal self-eval uses model masking.
- External eval uses separate script.

#### `eval_external.py`
External evaluator deletion test.
Uses PAD-token removal for external models.
Keep distinction:

- `causal_eval.py` = internal representation/self-eval
- `eval_external.py` = external token-removal eval

#### `eval_cross_dataset.py`
Extends AdvBench/HarmBench single-turn seeds into multi-turn conversations.
Do not use for MHJ because MHJ is already multi-turn.

#### `eval_cross_model_transfer.py`
ShieldGemma/LlamaGuard transfer.
Important fixes:

- use unique variants cache per subset
- save actual threshold, not hardcoded 0.5
- evaluator type must match model
- ShieldGemma results are secondary

#### `eval_boundary_stress.py`
Boundary stress test on rejected benign records.
Strong result: GuardLens FPR 0.0072 on 279 boundary records.

#### `eval_surface_risk_fpr.py`
Surface-risk FPR comparison.
Important result:

- surface-risk boundary FPR@0.5 = 0.3728
- all benign FPR@0.5 = 0.1429

#### `eval_implicit_explicit.py`
Contextual vs lexical pivot analysis.
Updated from old implicit/explicit naming.

#### `eval_paraphrase.py`
Paraphrase robustness.
Important:

- subset names must be `contextual_pivot`, `lexical_pivot`
- not old `implicit`, `explicit`.

#### `eval_attribution_precision.py`
Hard-negative attribution precision / minimality.
Must use v11 family/status names.

#### `eval_deconfounded.py`
Creates deconfounded test variants.

Variants:

- SR-neutralized adversarial
- Noise-equalized adversarial
- SR-injected benign
- Combined
- SR-neutralized changed-only

Important functions:

- `surface_risk_score`
- `neutralize_surface_risk`
- `equalize_noise`
- `inject_surface_risk`
- `combined_deconfound`
- `evaluate_variant`

Known caveats:

- Do not report token F1 on transformed variants due to span offset mismatch.
- Use DD/flip/FPR.

Latest results:

- SR-injected benign FPR:
  - GuardLens 0.005
  - Surface-risk 0.947

#### `eval_attribution_utility.py`
Post-hoc utility + turn mass + pivot-window metrics.

Metrics:

- Utility = DD@15 - λ*FPR
- Causal turn mass
- Pivot-window accuracy

Important fixes:

- Get correct attribution output key.
- Use boundary FPR headline.
- Avoid double-counting causal turn mass.
- Map raw turn/user-turn index correctly.

Latest results:

- GuardLens utility 0.504
- Surface-risk boundary utility 0.195
- causal turn mass mean 0.310
- pivot within ±5 = 0.678

#### `eval_deconfound.slurm`
Runs:

1. attribution utility
2. causal turn mass
3. pivot window
4. deconfounded variants

Important:

- Uses one GPU.
- Prerequisites:
  - `causal_eval_results.json`
  - `boundary_stress.json`
  - `surface_risk_fpr.json`

#### `eval_full.slurm`
Master/core eval script.
Uses:

- `best_attribution.pt` for causal
- `best_detection.pt` for classification
- outputs core results

Known:

- should fail if main checkpoint missing.
- should avoid JSON corruption by using clean `--output`.

### Human annotation files

#### `guardlens_human_annotation_guide_updated.md`
Human annotation guide.

Annotators fill:

- classification: `ADVERSARIAL`, `BENIGN`, `UNCERTAIN`
- pivot_turn: 0-indexed user turn ID, `DISTRIBUTED`, or blank
- causal_spans
- confidence: `HIGH`, `MEDIUM`, `LOW`
- notes

Important guide requirements:

- Blind annotation
- Only annotate user turns
- Sensitive words alone do not make conversation adversarial
- Use DISTRIBUTED for multi-turn causality
- Target agreement:
  - classification Cohen’s κ ≥ 0.70
  - pivot/span lower acceptable but report exact/±1/span-overlap

### Context packs created

Already created earlier:

- `GuardLens_Project_Context_Pack.md`
- `GuardLens_Codebase_Context_Pack.md`
- `GuardLens_Paper_Writing_Context_Pack.md`

Current file is the updated project context pack including latest deconfounded results.

---

## 14. Best next steps

### Highest priority before writing

1. **Run human audit benchmark**
   - 100 conversations
   - 50 adversarial / 50 hard benign
   - ideally 50 double-annotated
   - report κ, pivot agreement, span overlap F1
   - exclude from training/model selection

2. **Finalize span-level metrics**
   - span precision/recall/F1
   - incidental FPR
   - top-k span precision
   - AUPRC over non-ignored tokens

3. **Clean final tables**
   - Detection table
   - Core attribution table
   - Utility/specificity table
   - Deconfounded table
   - Boundary stress table
   - External/MHJ table
   - Human audit table

4. **Write paper with careful claims**
   - emphasize framework/dataset/evaluation
   - not “architecture beats everything”

5. **Remove or caveat weak results**
   - random token F1
   - exact pivot
   - ShieldGemma main
   - MHJ detection recall

### Optional but useful

6. Add bootstrap confidence intervals for major metrics:
   - DD@15
   - utility
   - boundary FPR
   - SR-injected benign FPR
   - paraphrase stability

7. Add error taxonomy:
   - false positives
   - false negatives
   - surface-risk failures
   - distributed pivot failures

8. Add figure:
   - Utility tradeoff scatter:
     - x-axis FPR
     - y-axis DD@15
     - GuardLens high DD, very low FPR
     - Surface-risk high DD, high FPR

---

## 15. Writing / paper strategy

### Intended venue

EMNLP Main Track / ACL Rolling Review.

Current estimate after deconfounded results:

- Without human audit: ~55-70% competitive range depending on writing/reviewer fit.
- With clean human audit: ~60-75% plausible, still not guaranteed.

Do not choose Industry Track unless paper is reframed around deployment/system operations.

### Strongest contribution framing

Do not frame as:

> GuardLens is the best attribution model.

Frame as:

> GuardLens introduces a framework for causal attribution of multi-turn guardrail failures, with interactive data generation, tiered causal supervision, validated benign controls, and specificity-aware evaluation showing that lexical heuristics are strong but brittle.

### Main claims to make

Supported claims:

1. GuardLens strongly outperforms flat and turn-only baselines on internal detection.
2. GuardLens produces label-aligned token attribution.
3. GuardLens beats attention, IG, Grad×Input, and random on causal attribution/deletion.
4. Surface-risk is a strong lexical deletion heuristic.
5. Surface-risk has poor specificity on benign/boundary/surface-injected examples.
6. GuardLens has far better attribution utility when false positives are penalized.
7. GuardLens is robust under paraphrase and deconfounded noise/neutralization stress.
8. Exact pivot is harsh; windowed pivot shows causal-region localization.

Claims to avoid:

1. GuardLens beats surface-risk on raw deletion.
2. CF Phase 3 improves attribution.
3. NoCF/NoFusion ablations prove those modules are essential.
4. MHJ detection generalizes strongly.
5. Exact pivot localization is strong.
6. Surface-risk neutralization completely breaks surface-risk.
7. Random token F1 is meaningful.

### Suggested paper structure

1. Introduction
2. Related Work
   - jailbreak detection
   - prompt attribution/explainability
   - multi-turn safety
   - counterfactual evaluation
3. Problem Setup
   - multi-turn causal attribution
   - definitions: pivot, causal span, distributed causality
4. Dataset Construction
   - data sources
   - interactive generation
   - target/validator setup
   - transfer tiers
   - benign pool
   - causal analysis
   - postprocessing
   - statistics
5. GuardLens Model
   - multi-turn encoding
   - attribution head
   - detection head
   - tiered supervision
   - training phases
6. Evaluation
   - detection
   - causal attribution
   - utility/specificity
   - deconfounded stress
   - paraphrase robustness
   - external/MHJ
   - human audit
7. Results
8. Analysis and Limitations
9. Conclusion

### Key equations

Attribution Utility:

```text
Utility_k(m) = DD_k(m) - λ · FPR(m)
```

Headline:

```text
Utility_15 = DD@15 - BoundaryFPR
```

Alternative optional multiplicative:

```text
SpecificityAdjustedDD = DD@15 × (1 - BoundaryFPR)
```

Causal Turn Mass:

```text
CTM = attribution mass on causal turn roles / total attribution mass
```

Pivot-window:

```text
Acc_w = 1[ |predicted_user_turn - annotated_pivot_user_turn| ≤ w ]
```

### Planned tables / figures

Main paper tables:

1. Dataset statistics
2. Detection baselines
3. Core causal attribution
4. Attribution utility / causal-specificity
5. Deconfounded evaluation
6. Boundary stress / surface-risk FPR
7. Human audit agreement and model performance
8. External generalization / MHJ stress test

Figures:

1. Pipeline diagram:
   - generation → validation → causal analysis → training → evaluation
2. Utility tradeoff plot:
   - DD@15 vs FPR
3. Multi-turn attribution example:
   - setup/escalation/payload highlighted
4. Pivot-window / causal-turn-mass chart
5. Surface-risk injected benign contrast

---

## 16. Anything repeatedly corrected or clarified

1. User wants crisp, concise answers unless explicitly asking for details.
2. Avoid em dashes.
3. Do not over-agree if an idea is weak.
4. Do not call the dataset “complete solution”; keep caveats.
5. Augmentation should be removed for interactive generation.
6. More synthetic data is not the bottleneck anymore.
7. The work should focus on attribution, not just classification.
8. Use `meta-llama/Meta-Llama-3-8B-Instruct` as the Llama target in descriptions.
9. `Qwen/Qwen2.5-7B-Instruct` interactive generation is viable due to feedback loop.
10. 14B generator was slower and not clearly better.
11. MHJ is already multi-turn, so do not feed it into single-turn extension pipeline.
12. Keep MHJ external, not in training split.
13. If surface-risk wins raw deletion, do not hide it. Explain it as deletion vs specificity.
14. Surface-risk should be framed as a lexical-oracle stress baseline.
15. Boundary stress and SR-injected benign are now central results.
16. Phase 3 ran, but did not improve selected checkpoint.
17. NoCF and NoFusion did not underperform, so do not overclaim them.
18. Exact pivot accuracy is low; use pivot-window and causal-turn-mass.
19. Random token F1 is suspicious; do not report.
20. Human annotation guide is ready and should be used to strengthen paper credibility.
21. The final target remains EMNLP Main if human audit and writing are strong.
22. Paper should be honest and nuanced, not overstated.

---

## Current one-paragraph summary for a new collaborator

GuardLens is a multi-turn guardrail-failure attribution project targeting EMNLP Main. The final v11 dataset has 1,762 records split 1,220/266/276, built from interactive Qwen→Llama generation, Mistral/Qwen validation, LLM span annotation, counterfactual causal analysis, and a separately validated benign pool. GuardLens uses frozen DeBERTa-v3-base with multi-turn encoding and token attribution/classification heads. It achieves strong internal detection (F1 0.994) and strong token attribution alignment (Attr F1 0.878), beating turn-level and flat DeBERTa baselines. Surface-risk beats GuardLens on raw DD@15 deletion, but has poor specificity: boundary FPR 0.373 vs GuardLens 0.007, and SR-injected benign FPR 0.947 vs GuardLens 0.005. The new headline metric is Attribution Utility = DD@15 − BoundaryFPR, where GuardLens scores 0.504 vs surface-risk 0.195. Deconfounded evaluations show GuardLens remains robust under neutralization/noise and matches surface-risk under noise-equalized conditions. Pivot exact accuracy is weak, but windowed pivot reaches 67.8% within ±5 user turns. Main remaining task is a human audit benchmark to reduce synthetic-label reviewer risk.
