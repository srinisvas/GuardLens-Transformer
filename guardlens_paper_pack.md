# GuardLens Paper Writing Context Pack
**Last updated:** May 20, 2026
**Working title:** *GuardLens: Tiered Causal Attribution for Multi-Turn Adversarial Prompt Detection*

---

## 1. Intended Venue

**Primary:** EMNLP 2026 (ARR submission)
**Backup:** ACL 2026 Rolling Review, NAACL 2026
**Format:** Long paper (8 pages + references + appendix)
**Review criteria:** Novelty, soundness, significance, clarity, reproducibility

---

## 2. Target Contribution Claims

### Primary contributions (must defend):
1. **Dataset and supervision framework:** Interactive multi-turn adversarial generation with cross-model behavioral validation and tiered causal supervision (construction → llm_confirmed → cf_weak → cf_strong)
2. **Attribution methodology:** Hierarchical token→turn→conversation attribution that separates causal adversarial evidence from surface-form shortcuts
3. **Evaluation methodology:** Deconfounded evaluation framework that exposes construction artifacts in attribution baselines, including Attribution Utility (DD−λ×FPR), causal turn mass, and controlled lexical neutralization

### Secondary contributions (support but don't headline):
4. Separate validated benign pool with boundary stress testing
5. Cross-model transfer evaluation via ShieldGemma-9B
6. External generalization to MHJ real-world multi-turn jailbreaks

### What this paper is NOT claiming:
- NOT claiming state-of-the-art jailbreak detection (classification is table stakes, not the contribution)
- NOT claiming GuardLens beats surface risk on raw Deviation Drop (it doesn't on internal data)
- NOT claiming pivot-turn localization is solved (it's auxiliary)
- NOT claiming the dataset is large-scale (1,762 records is modest but carefully curated)

---

## 3. Novelty Framing

### What exists:
- Single-turn jailbreak detection (LlamaGuard, ShieldGemma, PromptGuard)
- Binary multi-turn classification (WildGuard, some RLHF-based approaches)
- Post-hoc XAI attribution (IG, attention, SHAP) applied to safety classifiers
- Jailbreak datasets (AdvBench, HarmBench, JailbreakBench, MHJ) — mostly single-turn or human-crafted

### What's missing (our gap):
- **No existing work provides token-level causal attribution for multi-turn guardrail failures.** Existing attribution methods treat conversations as flat sequences and don't account for cross-turn causal structure. Surface-form keyword matching is competitive on curated benchmarks but fails on implicit/contextual attacks and produces high false positive rates on benign conversations with adversarial vocabulary.

### Our novelty:
1. **Interactive adversarial generation** — LLM-vs-LLM attack loop that produces naturalistic multi-turn jailbreaks (45.6% success rate on Llama-8B)
2. **Three-family zero-circularity validation** — Qwen generates, Llama targets, Mistral validates. No model evaluates its own outputs.
3. **Tiered causal supervision** — conservative counterfactual promotion where only spans surviving replay-based intervention receive CF labels
4. **Deconfounded attribution evaluation** — controlled experiments (SR-neutralization, noise equalization, vocabulary injection) that isolate construction artifacts from genuine causal understanding

---

## 4. Methodology Wording

### Dataset generation — key phrases:
- "Interactive adversarial generation with adaptive feedback" (not "template-based" or "static")
- "Cross-model behavioral validation" (not just "validation")
- "Three-family zero-circularity design" — Qwen generates, Llama targets, Mistral validates
- "Tiered causal supervision with conservative counterfactual promotion"
- "Separate validated benign pool with dual-model rejection filtering"

### Model — key phrases:
- "Hierarchical multi-turn attribution" — turn-level encoding → cross-turn fusion → token-level attribution
- "Lightweight attribution head on frozen backbone" — 2.2M trainable parameters on DeBERTa-v3-base
- "Joint classification, attribution, and pivot localization"
- "Three-phase curriculum: classification → attribution → counterfactual fine-tuning"

### Evaluation — key phrases:
- "Model-grounded causal metrics" — Deviation Drop, Flip Rate measure actual model behavior change
- "Attribution Utility" — combines deletion power with specificity cost
- "Deconfounded evaluation" — controlled interventions isolating construction artifacts
- "External multi-turn generalization" — MHJ benchmark with zero training exposure

### Phrases to use carefully:
- "Surface risk baseline" — always clarify it's a deterministic keyword matcher, not a learned model
- "Construction artifact" — the SR lexicon guided generation, so the baseline reads the construction signal
- "Causal" vs "correlated" — GuardLens identifies causally validated tokens; surface risk finds correlated keywords

---

## 5. Reviewer-Risk Points and Defenses

### Risk 1: "Surface risk outperforms GuardLens on Deviation Drop"
**The concern:** DD@15% surface_risk=0.568 vs GuardLens=0.511 on internal test.
**Defense (multi-pronged):**
- Construction artifact: SR lexicon guided generation, creating circularity. SR reads the "answer key."
- MHJ reversal: on external data, GuardLens DD=0.495 vs SR=0.204 (2.4× improvement)
- Deconfounded: noise equalization eliminates the gap (GL=0.504 vs SR=0.502)
- SR-injected benign FPR: SR fires on 94.7% of benign records with injected keywords; GuardLens fires on 0.5%
- Attribution Utility at λ=1.0: GuardLens=0.504 vs SR=0.195 (boundary FPR penalty)
- SR plateaus: identical flip rate at k=5% through k=20% (finds keywords and nothing else)
**Wording:** "Surface risk achieves competitive Deviation Drop on internal test data due to construction circularity — the evaluation baseline effectively reads the generation signal. Three lines of evidence expose this as a shortcut rather than causal understanding: [MHJ, deconfounding, FPR]."

### Risk 2: "MHJ recall is only 9.1%"
**Defense:** Classification doesn't generalize (expected — different attack distributions). Attribution does: GuardLens DD=0.495 vs SR=0.204 on detected subset. Frame as: "We separately evaluate classification generalization (limited, 9.1% recall) and attribution generalization (strong, 2.4× improvement over surface risk). The attribution mechanism transfers even when classification does not."

### Risk 3: "Dataset is synthetic and small (1,762 records)"
**Defense:** Quality over quantity. Each adversarial record was interactively generated against a live target, cross-model validated, and causally analyzed. 549 malicious records with validated jailbreaks across 10 strategies. Separate benign pool with boundary stress testing. Human annotation benchmark (in progress) validates synthetic labels.

### Risk 4: "Pivot accuracy is weak"
**Defense:** Exact pivot accuracy (9.2%) reflects distributed causality in 28-turn conversations. Pivot window ±3 = 48.3%, ±5 = 67.8%. Causal turn mass = 31% on causal region. Reframe: "In long multi-turn conversations, adversarial intent is often distributed across setup, escalation, and payload turns rather than concentrated at a single pivot. We therefore evaluate attribution mass distribution over the causal region rather than exact pivot localization."

### Risk 5: "Phase 3 CF training didn't help"
**Defense:** CF signal was too sparse (29 records). The tier-weighted attribution loss in Phase 2 already captures the signal. "Conservative counterfactual validation provides dataset quality assurance — ensuring only causally validated spans receive high supervision weights — rather than direct training signal at this annotation scale."

### Risk 6: "GuardLens ≈ GuardLens-NoCF"
**Defense:** Same root cause as Risk 5. Both models trained through identical Phase 2 with same tier weights. CF fine-tuning adds phase 3 oversampling which didn't improve. This is a data scale limitation, not a methodological failure.

### Risk 7: "Why not use a larger model / LlamaGuard / GPT-4?"
**Defense:** The contribution is the attribution framework, not the base model. 2.2M trainable parameters on a frozen DeBERTa backbone demonstrates that lightweight attribution heads can learn causal structure. Using a larger model would conflate model capacity with attribution methodology. External evaluator (ShieldGemma-9B) tests cross-model transfer.

### Risk 8: "Token F1 may be inflated"
**Defense:** Random baseline gets ~0.62 token F1 due to majority null-class matching. GuardLens at 0.878 is substantially above this floor. The primary metrics (DD, Flip Rate) are model-grounded and not subject to this artifact. Token F1 is reported as secondary.

### Risk 9: "Test set is small (87 adversarial, 189 benign)"
**Defense:** This is the frozen test split. Transfer eval uses full dataset (549 malicious). MHJ adds 537 external records. Cross-dataset adds 837 AdvBench+HarmBench seeds. Total evaluation coverage is substantial even if the primary test split is modest.

---

## 6. Related Work Positioning

### Jailbreak detection:
- LlamaGuard (Inan et al., 2023), ShieldGemma (Team et al., 2024), PromptGuard — single-turn or binary, no attribution
- WildGuard (Han et al., 2024) — multi-turn but binary classification only
- Position: "These systems detect jailbreaks but do not explain *which tokens* cause the failure or *why* a conversation is adversarial."

### Multi-turn adversarial attacks:
- Crescendo (Russinovich et al., 2024), MHJ (Li et al., 2024), TAP (Mehrotra et al., 2024) — attack methods
- Position: "We use interactive adversarial generation as a dataset construction method, not as an attack contribution."

### Attribution / Explainability:
- Integrated Gradients (Sundararajan et al., 2017), Attention-based (Jain & Wallace, 2019), SHAP (Lundberg & Lee, 2017)
- Position: "Standard attribution methods treat inputs as flat sequences and don't account for multi-turn conversational structure. We show they underperform on conversations where adversarial intent is distributed across turns."

### Jailbreak datasets:
- AdvBench (Zou et al., 2023), HarmBench (Mazeika et al., 2024), JailbreakBench, ToxicChat
- Position: "Existing benchmarks are predominantly single-turn. We construct a multi-turn dataset with per-token causal annotations, enabling attribution-level evaluation beyond binary classification."

### Causal evaluation:
- Deletion-based faithfulness (DeYoung et al., 2020), Counterfactual explanations, ERASER benchmark
- Position: "We adapt deletion-based faithfulness metrics to multi-turn conversations and add tier-stratified evaluation that connects annotation confidence to attribution quality."

---

## 7. Evaluation Expectations (What Reviewers Want to See)

### Must-have tables:
- Classification comparison (5 models)
- Causal attribution comparison (6 methods, DD/Flip/Nec/Suf/Token F1)
- Attribution Utility (DD − λ×FPR tradeoff)
- Surface risk FPR comparison across benign subsets
- External generalization (MHJ)

### Should-have tables:
- Deconfounded evaluation (SR-neutralized, noise-equalized, combined)
- Per-supervision-tier attribution quality
- Pivot window accuracy
- Paraphrase robustness
- Cross-dataset generalization (AdvBench + HarmBench)

### Nice-to-have (appendix):
- ShieldGemma transfer with threshold sensitivity
- Attribution precision + minimality curve
- Per-transfer-tier breakdown
- Human annotation IAA (when available)
- Dataset statistics and examples

---

## 8. Planned Figures and Tables

### Main paper (8 pages):

**Table 1: Dataset Statistics**
Records, splits, supervision tier distribution, pivot kind distribution, avg turns

**Table 2: Classification Results**
| Model | F1 | Acc | P | R |
5 models, test set

**Table 3: Causal Attribution Comparison**
| Method | DD@15% | Flip@15% | Token F1 | Trigger Size |
6 methods (GuardLens, SR, IG, Grad×Input, Attention, Random)
Caption emphasizes: SR has higher DD but 0.058 Token F1; GuardLens has 0.878

**Table 4: Attribution Utility**
| Method | DD@15% | Boundary FPR | Utility (λ=1.0) |
Only GuardLens vs Surface Risk. Headline: 0.504 vs 0.195

**Table 5: Surface Risk Specificity Failure**
| Subset | SR FPR@0.5 | GuardLens FPR |
all_benign, false_lead, boundary, hard_benign
Plus: SR-injected benign: SR 94.7% vs GL 0.5%

**Table 6: Deconfounded Evaluation**
| Variant | GuardLens DD | Surface Risk DD |
original, noise-equalized, combined
Key finding: noise equalization eliminates SR advantage

**Table 7: External Generalization (MHJ)**
| Metric | GuardLens | Surface Risk | Random |
DD, Flip, Necessity on detected subset (n=49)
With caveat about 9.1% classification recall

**Figure 1: Minimality Sensitivity Curve**
Flip rate vs k% for guardlens, surface_risk, random
Shows SR plateau at 56.3% while GL keeps climbing

**Figure 2: Pipeline Overview**
Interactive generation → cross-model validation → causal analysis → tiered supervision → training → evaluation

**Figure 3: Deconfounded Comparison (bar chart)**
Side-by-side DD@15% for original vs noise-equalized vs combined

### Appendix:

**Table A1: Per-Supervision-Tier Attribution**
cf_weak, construction, llm_confirmed — validates tier system

**Table A2: Pivot Window Accuracy**
±0 through ±5 with distance statistics

**Table A3: Paraphrase Robustness**
Turn ρ, Token ρ, by pivot kind

**Table A4: Cross-Dataset Generalization**
AdvBench/HarmBench per-source breakdown

**Table A5: ShieldGemma Transfer**
Per-subset, per-threshold results

**Table A6: Causal Turn Mass**
Per-role attribution distribution

**Table A7: Attribution Precision**
FPAR per subset, hard-neg vs adversarial

**Table A8: Human Annotation Agreement** (when available)
IAA, span P/R/F1, per-annotator consistency

---

## 9. Writing Constraints

### Length:
- 8 pages main content (EMNLP format)
- Unlimited appendix
- Aim for 7.5 pages to leave room for reviewer-requested additions

### Section structure (recommended):
1. **Introduction** (1 page) — problem, gap, contribution, key result
2. **Related Work** (0.75 page) — jailbreak detection, attribution, multi-turn attacks
3. **Dataset Construction** (1.5 pages) — interactive generation, validation, causal analysis, tier system
4. **Model** (1 page) — architecture, training, three-phase curriculum
5. **Evaluation** (2.5 pages) — metrics, baselines, main results, deconfounding, external generalization
6. **Analysis** (0.75 page) — surface risk artifact, specificity, limitations
7. **Conclusion** (0.5 page)

### Formatting:
- Use `\small` tables if space is tight
- Figures should be self-contained with full captions
- Bold best results, underline second-best
- Report all numbers to 3 decimal places for metrics, 1 decimal for percentages

### Ethics:
- Include ethics statement about responsible disclosure of jailbreak techniques
- Note that all adversarial examples target open-weight models (Llama, Qwen, Mistral)
- Dataset will be released with appropriate access controls

---

## 10. Phrases and Claims to Avoid

### Never say:
- "We outperform surface risk on deletion metrics" — you don't on internal data
- "Our model achieves state-of-the-art jailbreak detection" — classification isn't the contribution
- "Pivot localization is accurate" — it's not, use window accuracy
- "Phase 3 CF training significantly improves attribution" — it doesn't measurably
- "The dataset is large-scale" — 1,762 records is modest
- "GuardLens solves multi-turn attribution" — too strong
- "Surface risk is a weak baseline" — it's deceptively strong, which is the point

### Use carefully:
- "Causal" — only for spans validated through counterfactual intervention. Construction-tier spans are "annotated" not "causally validated"
- "Generalization" — qualify: classification doesn't generalize to MHJ, attribution does
- "Robust" — only for paraphrase robustness (ρ=0.983), not for general robustness claims
- "Zero-circularity" — technically the same backbone (DeBERTa) is used for all GuardLens variants; zero-circularity applies to the dataset validation (Qwen/Llama/Mistral)

### Preferred framings:
- "Surface risk is a strong but brittle shortcut" — not weak, but limited
- "Attribution Utility captures the precision-deletion tradeoff" — not just deletion power
- "The deconfounded evaluation isolates construction artifacts" — not "proves GuardLens is better"
- "Attribution transfers to unseen attack patterns even when classification does not" — honest about limitations

---

## 11. Strongest Version of the Paper Story

### One-paragraph pitch:
Multi-turn jailbreak attacks exploit conversational dynamics — gradually building context, shifting perspectives, and embedding harmful requests within benign-looking dialogue. Detecting these attacks is necessary but insufficient; understanding *which tokens* and *which turns* cause guardrail failure is essential for building robust defenses. We present GuardLens, a framework combining interactive adversarial dataset generation, tiered causal supervision, and hierarchical token-level attribution. A naive surface-risk keyword baseline achieves competitive deletion metrics on our benchmark by exploiting construction artifacts — a finding we expose through controlled deconfounding experiments that neutralize planted keywords and equalize grammar signals. When evaluated on external multi-turn jailbreaks (MHJ) where construction artifacts are absent, GuardLens attribution outperforms surface risk by 2.4× (DD@15%: 0.495 vs 0.204). On benign conversations with injected adversarial vocabulary, surface risk fires on 94.7% of records while GuardLens maintains 0.5% false positive rate. These results demonstrate that learned attribution captures genuine causal structure beyond surface shortcuts, providing interpretable evidence for why multi-turn conversations succeed or fail at bypassing safety guardrails.

### The paper's "aha moment" (what makes reviewers excited):
The surface risk baseline winning on DD is not a failure — it's the setup for the paper's central insight. Surface risk *should* win on a dataset where the same keyword lexicon guided construction. The interesting question is: does the learned model go beyond keywords? The deconfounded evaluation answers yes:
1. Neutralize the keywords → SR barely changes, GL barely changes (both found more than keywords)
2. Add noise to break grammar signal → GL improves, SR drops, gap closes to zero
3. Inject keywords into benign → SR fires 95%, GL fires 0.5%
4. Test on external MHJ → GL wins 2.4×

This progression tells a complete story about what attribution methods actually learn.

### How each result serves the narrative:

| Result | Narrative role |
|---|---|
| Classification F1 ≈ 0.99 | "Classification is solved for this distribution — the challenge is attribution" |
| SR DD > GL DD on internal | "Sets up the puzzle — is keyword deletion sufficient?" |
| Token F1: GL=0.878 vs SR=0.058 | "SR finds tokens that flip the model but NOT the causally validated ones" |
| SR FPR 66.7% on false-lead | "Keywords without context cause massive false positives" |
| Noise equalization: GL=0.504, SR=0.502 | "Remove the grammar artifact and they're equal" |
| SR-injected benign: SR 94.7%, GL 0.5% | "Keywords in safe contexts fool SR completely" |
| MHJ: GL 2.4× better | "On real attacks, learned attribution wins decisively" |
| Attribution Utility: GL 0.504 vs SR 0.195 | "When you penalize false positives, GL wins by 2.6×" |
| Paraphrase ρ=0.983 | "Attribution tracks meaning, not surface form" |
| Boundary FPR 0.72% | "The model is specific, not just sensitive" |
| Per-tier: cf_weak DD=0.744 > construction=0.583 | "Higher-confidence labels produce better attribution" |
| Pivot ±3 = 48.3% | "In 28-turn conversations, the model localizes the causal region" |

### The paper's honest limitations (report prominently):
1. Classification does not generalize to MHJ (9.1% recall)
2. Surface risk outperforms on internal DD due to construction circularity
3. Pivot exact accuracy is weak; distributed causality is the norm
4. Phase 3 CF training didn't improve over Phase 2 at current annotation scale
5. Test set is small (87 adversarial in frozen split)
6. ShieldGemma transfer is underpowered (90% refusal rate)
7. All training data is synthetic (human benchmark pending)

### The paper's strongest claims (defend vigorously):
1. Attribution Utility (DD−FPR) favors GuardLens over all baselines
2. On external MHJ data, GuardLens attribution is 2.4× better than surface risk
3. Surface risk FPR on benign records with adversarial vocabulary is 95% vs GuardLens 0.5%
4. Deconfounded evaluation (noise equalization) eliminates surface risk's internal advantage
5. Paraphrase robustness (ρ=0.983) proves attribution tracks semantics, not surface form
6. Per-tier attribution quality validates the supervision tier system
7. Boundary stress test (0.72% FPR) proves high specificity

---

## 12. Abstract Draft (iterate from this)

> Multi-turn adversarial attacks exploit conversational dynamics to bypass LLM safety guardrails through gradual context manipulation rather than single explicit prompts. While detecting such attacks is important, understanding *which tokens and turns* are causally responsible for guardrail failure is essential for building interpretable, robust defenses. We present GuardLens, a framework for token-level causal attribution in multi-turn adversarial conversations. Our approach combines (1) interactive LLM-vs-LLM adversarial generation producing naturalistic jailbreak conversations, (2) cross-model behavioral validation ensuring zero-circularity between generation and evaluation, and (3) tiered causal supervision through counterfactual span analysis. We train a hierarchical attribution model on 1,762 conversations with per-token causal labels and evaluate against five attribution baselines. A deterministic keyword baseline achieves competitive deletion metrics on our benchmark by exploiting construction artifacts — a phenomenon we expose through controlled deconfounding experiments. On external multi-turn jailbreaks (MHJ), GuardLens outperforms the keyword baseline by 2.4× on Deviation Drop while maintaining 0.5% false positive rate on benign conversations with injected adversarial vocabulary (vs. 94.7% for keywords). Our Attribution Utility metric, which penalizes false positives, shows GuardLens achieves 2.6× higher utility than the keyword baseline. These results demonstrate that learned attribution identifies genuine causal mechanisms beyond surface shortcuts in multi-turn guardrail failures.

---

## 13. Key Numbers for Quick Reference

```
Dataset: 1,762 total, 549 malicious, 1,213 benign
Splits: 1,220 / 266 / 276 (train/dev/test)
Test: 87 adversarial, 189 benign

Classification F1: 0.988 (GuardLens), 0.832 (ConvDeBERTa), 0.758 (Turn-Level)
Token F1: 0.878 (GuardLens), 0.058 (Surface Risk)
DD@15%: 0.511 (GuardLens), 0.568 (Surface Risk)
Attribution Utility: 0.504 (GuardLens), 0.195 (Surface Risk)

Boundary FPR: 0.72% (GuardLens), 37.3% (Surface Risk @0.5)
False-lead FPR: 3.0% (GuardLens), 66.7% (Surface Risk)
SR-injected benign FPR: 0.5% (GuardLens), 94.7% (Surface Risk)

MHJ DD@15%: 0.495 (GuardLens), 0.204 (Surface Risk) — 2.4× improvement
MHJ Recall: 9.1% (49/537)

Deconfounded (noise-equalized): GL=0.504, SR=0.502 (gap eliminated)
Paraphrase ρ: 0.983 (turn), 0.957 (token)
Pivot ±3: 48.3%, ±5: 67.8%
Causal turn mass: 31%

Per-tier DD@15%: cf_weak=0.744, construction=0.583, llm_confirmed=0.378
```
