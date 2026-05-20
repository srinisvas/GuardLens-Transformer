# GuardLens Project Context Pack
**Last updated:** May 20, 2026
**Author:** Srinivasan (university researcher)
**Target venue:** EMNLP 2026 (ARR submission)

---

## 1. Project Goal and Current Research Direction

**GuardLens** is a framework for **token-level causal attribution in multi-turn adversarial prompt detection**. The core research question: when a multi-turn conversation causes an LLM to violate its safety guardrails, which specific tokens across which turns are *causally responsible* — not just correlated — for the failure?

The paper contribution is threefold:
1. A **dataset generation pipeline** with interactive adversarial generation, cross-model behavioral validation, and tiered causal supervision
2. A **hierarchical DeBERTa-based model** (GuardLens) that jointly performs conversation classification, per-token attribution, and pivot turn localization
3. An **evaluation framework** demonstrating that surface-form keyword baselines are strong but brittle shortcuts, while learned attribution tracks genuine causal mechanisms

---

## 2. Core Problem Being Solved

Existing jailbreak detection treats conversations as binary (safe/unsafe). This misses *why* a conversation is adversarial and *which parts* cause the failure. Attribution methods from XAI (Integrated Gradients, attention, Grad×Input) don't account for multi-turn conversational structure. A naive surface-risk keyword baseline (matching words like "bypass", "exploit", "jailbreak") achieves surprisingly high Deviation Drop but fails on:
- Implicit/contextual attacks with zero keyword overlap
- Benign conversations using adversarial vocabulary safely (cybersecurity research, policy analysis)
- External unseen attack patterns (MHJ real-world jailbreaks)

GuardLens addresses this by learning hierarchical token→turn→conversation attribution with tiered supervision from counterfactual validation.

---

## 3. Current Methodology/Pipeline

### Data Generation (GuardLens-DataGen-V2 project)
1. **Interactive adversarial generation**: Qwen-7B/14B (generator) crafts adaptive multi-turn attacks against live Llama-8B (target). 10 attack strategies × 10 adaptation tactics. Feedback loop where generator adapts based on target responses.
2. **Cross-model validation**: Mistral-7B-Instruct-v0.3 independently validates jailbreaks (three-family zero-circularity: Qwen generates, Llama targets, Mistral validates).
3. **4-pass causal analysis**: (a) LLM span annotation, (b) pivot-turn counterfactual (whole-turn ablation), (c) span-level counterfactual, (d) negative-control validation.
4. **Post-processing**: relabeling causal BENIGN_CONTEXT spans → CONTEXT_BRIDGE, upgrading distributed-causal records, tier/weight consistency validation.
5. **Separate benign pool generation**: 1,000 benign conversations across 5 categories, dual-validated by Llama + Mistral, 721 accepted.
6. **Final assembly**: 1,762 records merged, train/dev/test split (1,220/266/276).

### Training (GuardLens-Transformer project)
- 3-phase training: Phase 1 (classification, epochs 0-4), Phase 2 (+ attribution, epochs 5-19), Phase 3 (+ CF fine-tuning, epochs 20-24)
- 5 model variants: GuardLens, GuardLens-NoFusion, GuardLens-NoCF, Turn-Level, ConversationDeBERTa
- Backbone: DeBERTa-v3-base (frozen), ~2.2M trainable parameters

### Evaluation
- 16+ evaluation scripts covering classification, causal attribution, cross-dataset generalization, paraphrase robustness, cross-model transfer, boundary stress testing, external MHJ evaluation, deconfounded variants, and attribution utility metrics.

---

## 4. Dataset Details

### Final Dataset: 1,762 records
- **Malicious:** 549 (31.2%) — all jailbroke at least one model
- **Benign:** 1,213 (68.8%) — 721 clean_benign + 492 validated_benign_twin
- **Splits:** Train 1,220 / Dev 266 / Test 276

### Record Schema (v11 fields)
```
conversation_id, pair_id, label (0/1), family, subtype, difficulty, turns [],
pivot_turn_id, pivot_kind (contextual_pivot | lexical_pivot | distributed | none),
transfer_tier (transfer_success | target_only | cross_only | benign),
supervision_tier (cf_strong | cf_weak | llm_confirmed | construction | benign_validated),
loss_weight (0.0–1.0), benign_status (clean_benign | validated_benign_twin),
validation_status, source_dataset, is_external_test, training_eligible
```

### Supervision Tier Weights
| Tier | Weight | Count (train) |
|---|---|---|
| cf_strong | 1.00 | 7 |
| cf_weak | 0.70 | 22 |
| llm_confirmed | 0.60 | 237 |
| construction | 0.40 | 452 |
| benign_validated | 1.00 | 502 |

### Key Dataset Properties
- Average conversation length: ~28 turns (malicious)
- 10 attack strategies, top performers: perspective_shift (97% jailbreak rate), academic_framing (96%)
- Pivot kinds: 305 contextual_pivot, 230 lexical_pivot, 10 none, 4 misleading_decoy
- Transfer tiers: 268 transfer_success, 244 cross_only, 37 target_only

### Boundary/Stress Data
- `benign_boundary.jsonl`: 279 records rejected during dual-model validation (hardest negatives)
- MHJ external test: 537 records from Multi-turn Human Jailbreak dataset, `is_external_test=True`

### File Locations (HPC)
```
$HOME/staging/dataset_gen_output/
  final_dataset.jsonl          (1,762 records)
  splits/train.jsonl           (1,220)
  splits/dev.jsonl             (266)
  splits/test.jsonl            (276)
  benign_boundary.jsonl        (279)
  mhj_external_test.jsonl      (537)
```

---

## 5. Model/Architecture Details

### GuardLens (main model)
- **Backbone:** DeBERTa-v3-base (frozen, 183.8M params)
- **Trainable:** 2.17M params (classification head, attribution head, turn fusion, pivot head)
- **Input shape:** [B, T, S] where T=max_turns=32, S=max_tokens_per_turn=192
- **Turn encoder:** Per-turn DeBERTa encoding → [B, T, S, D]
- **Cross-turn fusion:** TransformerEncoder (2 layers, 4 heads) on turn-level representations
- **Heads:**
  - Classification: pooled → linear → logit
  - Attribution: per-token sigmoid scores (attr_logits → attr_probs)
  - Pivot: per-turn logit + no-pivot embedding → [B, T+1]
- **Key output dict keys:** `cls_logits`, `attr_logits`, `attr_probs`, `pivot_logits`

### Baselines
| Model | Params | Description |
|---|---|---|
| GuardLens-NoFusion | 2.04M | No cross-turn transformer fusion |
| GuardLens-NoCF | 2.17M | Same architecture, no phase 3 CF training |
| Turn-Level | 197K | Independent turn classification, no cross-turn reasoning |
| ConversationDeBERTa | 197K | Flat concatenation, FlatConversationCollator, [B, L] input |

---

## 6. Training Setup

### Configuration
```python
max_turns = 32
max_tokens_per_turn = 192
max_total_tokens = 2048
batch_size = 4
gradient_accumulation = 4
learning_rate = 2e-4
weight_decay = 0.01
backbone = "microsoft/deberta-v3-base" (frozen)
```

### Three-Phase Training
- **Phase 1 (epochs 0-4):** Classification loss only
- **Phase 2 (epochs 5-19):** + Attribution loss (tier-weighted) + pivot loss
- **Phase 3 (epochs 20-24):** + CF loss with CF-tier oversampling (3× for cf_strong/weak, 1.5× for llm_confirmed)

### Key Training Outcomes
- Phase 3 ran but did NOT improve attribution over Phase 2 (best attrF1=0.887 at epoch 13 phase 2 vs 0.881 peak in phase 3)
- CF signal too sparse: 29 cf_strong/weak records out of 1,220 training records
- GuardLens ≈ GuardLens-NoCF because both effectively trained through same phase 2

### Checkpoints
```
$HOME/work/results/guardlens_v11/checkpoints/
  guardlens/best_detection.pt       (epoch 8, F1=1.000, threshold=varies)
  guardlens/best_attribution.pt     (epoch 13, attrF1=0.887, threshold=0.56)
  guardlens/best.pt                 (same as best_attribution)
  guardlens_no_fusion/best.pt       (epoch 12, attrF1=0.888)
  guardlens_no_cf/best.pt           (epoch 10, attrF1=0.888)
  turn_level/best.pt                (epoch 10, F1=0.761)
  conversation_deberta/best.pt      (epoch 12, F1=0.843)
```

### SLURM Environment
- University HPC, A100 80GB GPUs, 4 total
- SLURM account: `V_cs_hat_capstone_mkhan74`, partition: `defq`
- Conda env: `$HOME/work/conda_envs/guardlens_train`
- Models: `$HOME/work/hf_models/hub` (offline mode)

---

## 7. Evaluation Setup

### Evaluation uses two checkpoints:
- `best_detection.pt` → classification metrics (F1, accuracy, per-tier accuracy)
- `best_attribution.pt` → causal attribution metrics (DD, flip rate, token F1, pivot)

### Evaluation Scripts (in `guardlens/evaluation/`)
| Script | Purpose |
|---|---|
| `evaluate.py` | Classification eval, `--output` for clean JSON, `FlatConversationCollator` for ConvDeBERTa |
| `eval_causal.py` | Core causal attribution (DD, flip, necessity, sufficiency) + per-tier + per-transfer breakdowns |
| `causal_eval.py` | Framework: `run_causal_evaluation()`, attribution methods, masking |
| `eval_utils.py` | Shared utilities: `load_test_data()`, `partition_test_set_v11()`, LaTeX output |
| `eval_implicit_explicit.py` | Contextual vs lexical pivot subset analysis |
| `eval_attribution_precision.py` | Hard-negative attribution precision + minimality curve |
| `eval_cross_dataset.py` | AdvBench + HarmBench seed extension eval |
| `eval_cross_model_transfer.py` | ShieldGemma-9B transfer eval, `--threshold`, `--external-model`, `--variants-cache` |
| `eval_external.py` | Self-eval vs external model eval comparison |
| `eval_paraphrase.py` | Paraphrase robustness (Spearman ρ) |
| `eval_boundary_stress.py` | Classification on rejected boundary benign records |
| `eval_surface_risk_fpr.py` | Surface risk FPR across benign subsets (no GPU) |
| `eval_attribution_utility.py` | Attribution Utility score, causal turn mass, pivot window accuracy |
| `eval_deconfounded.py` | Deconfounded test variants (SR-neutralized, noise-equalized, SR-injected, combined) |
| `mhj_loader.py` | Converts MHJ data to v11 schema |

### SLURM Scripts (project root)
| Script | Purpose |
|---|---|
| `eval_full.slurm` | Master eval: SR FPR + classification (5 models) + causal (6 methods) + boundary + MHJ + subset analysis |
| `eval_core.slurm` | Core causal + classification only |
| `eval_deconfound.slurm` | Utility metrics + deconfounded variants |
| `eval_transfer_stable.slurm` | Stabilized ShieldGemma transfer with subset filtering + threshold sensitivity |
| `eval_boundary.slurm` | Boundary stress test standalone |
| `eval_mhj.slurm` | MHJ processing + evaluation |
| `eval_phase3.slurm` | Phase 3 checkpoint comparison |
| `train_phase3.slurm` | Resume phase 3 CF training from phase 2 checkpoint |
| `submit_eval_pipeline.sh` | Chain all eval stages with SLURM dependencies |

---

## 8. Current Results and Metrics

### Classification (best_detection.pt, test set n=276)
| Model | F1 | Acc | P | R | AttrF1 | PivotMal | Thr |
|---|---|---|---|---|---|---|---|
| GuardLens | 0.988 | 0.993 | 0.989 | 1.000 | 0.873 | 0.253 | 0.20 |
| GuardLens-NoFusion | 0.994 | 0.996 | 1.000 | 1.000 | 0.872 | 0.138 | 0.20 |
| GuardLens-NoCF | 1.000 | 1.000 | 1.000 | 1.000 | 0.882 | 0.241 | 0.20 |
| ConvDeBERTa | 0.832 | 0.895 | 0.837 | 0.828 | — | — | 0.59 |
| Turn-Level | 0.758 | 0.866 | 0.879 | 0.667 | — | — | 0.55 |

### Core Causal Attribution (best_attribution.pt, 87 adversarial test records)
| Method | DD@15% | Flip@15% | Nec@15% | Suf@15% | TrigSize | Token F1 |
|---|---|---|---|---|---|---|
| **GuardLens** | **0.511** | **0.494** | **0.537** | **1.000** | **310** | **0.878** |
| Surface Risk | 0.568 | 0.563 | 0.591 | 1.000 | 136 | 0.058 |
| Grad×Input | 0.361 | 0.333 | 0.358 | 1.000 | 517 | 0.003 |
| Int. Gradients | 0.297 | 0.276 | 0.292 | 1.000 | 514 | 0.002 |
| Attention | 0.058 | 0.046 | 0.059 | 1.000 | 1761 | 0.724 |
| Random | 0.001 | 0.000 | 0.000 | 0.985 | 1798 | 0.598 |

**Key observation:** Surface risk outperforms GuardLens on DD/Flip due to construction artifact (surface risk lexicon guided dataset generation). This is addressed by deconfounded evaluation and MHJ external results.

### Per-Supervision-Tier Attribution (GuardLens DD@15%)
| Tier | DD@15% | Flip@15% | n |
|---|---|---|---|
| cf_weak | 0.744 | 0.750 | 4 |
| construction | 0.583 | 0.586 | 29 |
| llm_confirmed | 0.378 | 0.404 | 52 |
| benign_validated | 0.000 | 0.000 | 113 |

### Attribution Utility (headline metric)
| Method | DD@15% | Boundary FPR | Utility (λ=1.0) |
|---|---|---|---|
| **GuardLens** | 0.511 | **0.007** | **0.504** |
| Surface Risk | 0.568 | 0.373 | 0.195 |

GuardLens utility is **2.6× higher** than surface risk.

### Surface Risk FPR Comparison (test + boundary)
| Subset | SR FPR@0.5 | GuardLens FPR |
|---|---|---|
| all_benign | 14.3% | ~0.5% |
| false_lead_benign | 66.7% | 3.0% |
| boundary_rejected | 37.3% | 0.72% |
| hard_benign | 18.2% | 0.0% |

### Boundary Stress Test (best_detection.pt, 279 boundary records)
- Accuracy: 99.28%, FPR: 0.72% (2/279 FP)
- Mean P(adv): 0.0045, P95: 0.0003, Max: 0.8165
- Per-family: false_lead_benign FPR=3.0%, all others 0.0%

### Deconfounded Evaluation (DD@15%)
| Variant | GuardLens | Surface Risk | Random |
|---|---|---|---|
| Original | 0.473 | 0.543 | -0.002 |
| SR-neutralized | 0.449 | 0.534 | -0.003 |
| SR-neutralized (changed only) | 0.415 | 0.504 | -0.003 |
| **Noise-equalized** | **0.504** | **0.502** | -0.002 |
| **Combined** | **0.484** | **0.491** | -0.002 |

**Key finding:** Noise equalization (adding typos to pristine adversarial turns) eliminates surface risk's advantage. GuardLens DD=0.504 vs SR DD=0.502.

### SR-Injected Benign FPR
- **GuardLens: 0.5% (1/189)**
- **Surface Risk: 94.7% (179/189)**

### MHJ External Evaluation (537 records, malicious-only)
- Classification recall: 9.1% (49/537 detected) — expected distribution gap
- On detected subset (n=49):
  - GuardLens DD@15%=0.495, Flip@15%=63.3%
  - Surface Risk DD@15%=0.204, Flip@15%=28.6%
  - **GuardLens beats surface risk by 2.4× on external data**

### Paraphrase Robustness
- Turn-level Spearman ρ: 0.983 ± 0.012
- Token-level Spearman ρ: 0.957 ± 0.009
- Contextual pivot ρ ≈ lexical pivot ρ (both ~0.983)

### Attribution Precision
- Adversarial FPAR: 0.339, Hard-negative FPAR: 0.000
- Minimality: GuardLens inflects at k=0.18 (slope 1.38); surface risk inflects at k=0.08 (slope 7.82) but plateaus at 56.3%

### Cross-Dataset Generalization (AdvBench + HarmBench, 489 records)
- GuardLens: F1=0.607, P=0.992, R=0.437
- HarmBench (52.8%) > AdvBench (37.0%) recall

### ShieldGemma Transfer (stabilized, final_dataset.jsonl)
| Subset | n | GuardLens | Surface Risk | GL-SR |
|---|---|---|---|---|
| transfer_success | 11 | 0.909 | 0.636 | +0.273 |
| lexical_pivot | 38 | 0.789 | 0.737 | +0.053 |
| contextual_pivot | 19 | 0.789 | 0.737 | +0.053 |

Threshold sensitivity (transfer_success): GL consistently beats SR across θ=0.3/0.4/0.5.

### Causal Turn Mass
- Mean: 0.310 (31% of attribution mass on causal turns)
- Per-role: adaptation=0.055, escalation=0.052, payload=0.051, setup=0.048

### Pivot Window Accuracy
| Window | Accuracy |
|---|---|
| ±0 (exact) | 9.2% |
| ±1 | 24.1% |
| ±2 | 37.9% |
| ±3 | 48.3% |
| ±5 | 67.8% |

Median distance: 4.0 turns. In 28-turn conversations, this is reasonable.

---

## 9. Known Issues, Weaknesses, Reviewer Risks

### Critical weaknesses to address in paper:
1. **Surface risk outperforms on DD@15%** — construction artifact. The SR lexicon guided generation, so the baseline reads the "answer key." Addressed by: MHJ (2.4× reversal), deconfounded evaluation (noise equalization eliminates advantage), FPR comparison (SR FPR=94.7% on injected benign vs GL 0.5%).
2. **MHJ low recall (9.1%)** — classification doesn't generalize to unseen attack patterns. Attribution does (DD=0.495 vs SR=0.204). Frame as: classification is distribution-dependent, attribution mechanism transfers.
3. **Pivot accuracy is weak** — exact match ~5-6%. Reframed with pivot window (±3 = 48.3%) and causal turn mass (31% on causal region). Do NOT make pivot a central claim.
4. **Phase 3 CF training didn't help** — 29 CF records too sparse. The tier weights in phase 2 already capture the signal. Frame as: CF validation provides dataset quality assurance, not direct training signal at this scale.
5. **GuardLens ≈ NoCF** — because phase 3 didn't improve, the ablation shows no CF benefit. Frame as: tier-weighted attribution loss in phase 2 is the primary mechanism.
6. **Test set is small** — 87 adversarial, 189 benign in test split. Use final_dataset.jsonl (549 malicious) for transfer eval, but primary paper tables use frozen test split.
7. **ShieldGemma transfer underpowered** — 90% refusal rate. Report honestly with n values. Random ≈ GuardLens on some subsets (15% ablation is destructive enough to flip regardless). Focus on GL vs SR gap, exclude random from transfer comparison.
8. **Synthetic data** — all training data is LLM-generated. Human annotation benchmark (100 records, in progress) partially addresses this. MHJ external eval addresses generalization.

### Likely reviewer questions:
- "Why not use LlamaGuard/GPT-4 as the evaluator?" — ShieldGemma was chosen for its open-weight availability and safety-specific design. LlamaGuard was considered but not staged on HPC.
- "Is the dataset too easy?" — Classification F1 near 1.0 suggests yes for classification. But attribution is the contribution, not classification.
- "Why DeBERTa-base not a larger model?" — 2.2M trainable params on frozen backbone is the point: lightweight attribution head, not another large safety model.

---

## 10. Decisions Already Made

1. **Three-family zero-circularity**: Qwen generates, Llama targets, Mistral validates. Non-negotiable design choice.
2. **Interactive generation over template-based**: v10 static templates had 3-10% jailbreak rate. v11 interactive achieves 45.6% (7B) and 27.0% (14B).
3. **Pre-split frozen test set**: train/dev/test are frozen. No re-splitting. MHJ is external-only.
4. **Surface risk is a construction oracle**: acknowledged and addressed through deconfounding, not hidden.
5. **best_attribution.pt for causal eval, best_detection.pt for classification**: separate checkpoints for separate tasks.
6. **Pivot reframed as auxiliary**: use window accuracy and causal turn mass, not exact pivot.
7. **Phase 3 reported honestly**: ran but didn't improve; tier weights in phase 2 are the primary mechanism.
8. **Random excluded from transfer comparison**: 15% ablation is too destructive — random also flips the external evaluator.
9. **ConversationDeBERTa needs FlatConversationCollator**: evaluate.py handles this with model_name check.

---

## 11. Open Questions Still Unresolved

1. **Human annotation benchmark**: instructions drafted, annotators assigned, results pending. Need 100 conversations (50 double-annotated for IAA/Cohen's κ). This will produce span-level P/R/F1 and validate synthetic labels.
2. **Span-level metrics**: post-hoc computation from existing predictions (causal span P/R/F1, incidental FPR, AUPRC). Not yet implemented.
3. **Tier ablation training**: construction-only vs +llm_confirmed vs +CF vs +incidental negatives. Would require 3-4 retraining runs (~12 GPU-hours). Not done due to Phase 3 showing no CF improvement.
4. **Token F1 inflation**: random achieves ~0.62 token F1 due to majority-class matching on null tokens. Needs positive-class-only F1 or AUPRC. Upstream fix in causal_eval.py needed.
5. **Attention baseline is proxy**: PyTorch TransformerEncoderLayer may not return actual attention weights. Falls back to representation-change proxy. Report as "attention/representation-change proxy."
6. **LlamaGuard transfer eval**: not attempted. Would require staging Llama-Guard-3-8B on HPC.

---

## 12. Important Terminology and Definitions

| Term | Definition |
|---|---|
| **Deviation Drop (DD@k%)** | Fractional change in model's adversarial probability when top-k% attributed tokens are masked |
| **Flip Rate (Flip@k%)** | Fraction of adversarial records where masking top-k% tokens flips classification to benign |
| **Necessity** | If masking attributed tokens changes the prediction, those tokens are necessary |
| **Sufficiency** | If keeping only attributed tokens preserves the prediction, they are sufficient |
| **Token F1** | Overlap between predicted attribution and ground-truth causal span labels |
| **Attribution Utility** | DD@k% − λ × FPR. Penalizes methods with high false positive rate |
| **Causal Turn Mass** | Fraction of total attribution mass falling on pivot/escalation/payload turns |
| **Pivot Window Accuracy** | Whether the highest-attributed turn is within ±W turns of the true pivot |
| **Surface Risk Score** | Deterministic keyword-matching score based on adversarial vocabulary |
| **FPAR** | False Positive Attribution Rate — attribution intensity on non-adversarial content |
| **Transfer Flip** | Whether masking attributed tokens in conversation X causes external model Y to flip its safety assessment |
| **Supervision Tier** | cf_strong > cf_weak > llm_confirmed > construction > benign_validated |
| **Transfer Tier** | transfer_success (both models jailbroken) > cross_only > target_only > benign |
| **Pivot Kind** | contextual_pivot (context-dependent) vs lexical_pivot (keyword-based) vs distributed |
| **Deconfounded Evaluation** | Test variants that remove construction artifacts (SR-neutralized, noise-equalized, SR-injected benign) |

---

## 13. File/Code Names and What Each File Does

### Project Structure (GuardLens-Transformer)
```
GuardLens-Transformer/
├── guardlens/
│   ├── __init__.py
│   ├── config.py                  # GuardLensConfig dataclass
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Classification eval entry, --output flag, FlatConversationCollator for DeBERTa
│   ├── eval_causal.py             # Causal attribution eval entry, --tier-eval, --transfer-eval
│   ├── data/
│   │   ├── dataset.py             # GuardLensDataset, GuardLensCollator, FlatConversationCollator
│   │   └── splits.py              # pair_aware_split()
│   ├── models/
│   │   ├── __init__.py            # MODEL_REGISTRY
│   │   ├── guardlens.py           # GuardLens model (attr_logits, attr_probs, pivot_logits)
│   │   ├── components.py          # TurnEncoder, CrossTurnFusion, AttributionHead
│   │   └── baselines.py           # ConversationDeBERTa, TurnLevelClassifier, NoFusion, NoCF
│   ├── training/
│   │   ├── trainer.py             # train_epoch(), evaluate(), 3-phase logic
│   │   ├── loss.py                # GuardLensLoss (cls + attr + cf + pivot)
│   │   └── schedule.py            # get_current_phase(), get_lambda_schedule()
│   └── evaluation/
│       ├── __init__.py
│       ├── causal_eval.py         # run_causal_evaluation(), attribution methods, surface_risk_attribution()
│       ├── eval_utils.py          # load_test_data(), partition_test_set_v11(), LaTeX output
│       ├── eval_implicit_explicit.py
│       ├── eval_attribution_precision.py
│       ├── eval_cross_dataset.py
│       ├── eval_cross_model_transfer.py  # --external-model, --threshold, --variants-cache
│       ├── eval_external.py
│       ├── eval_paraphrase.py
│       ├── eval_boundary_stress.py
│       ├── eval_surface_risk_fpr.py
│       ├── eval_attribution_utility.py   # Attribution Utility, causal turn mass, pivot window
│       └── eval_deconfounded.py          # SR-neutralized, noise-equalized, SR-injected, combined
├── eval_full.slurm               # Master eval job
├── eval_core.slurm               # Core causal + classification
├── eval_deconfound.slurm         # Utility + deconfounded
├── eval_transfer_stable.slurm    # Stabilized ShieldGemma
├── eval_boundary.slurm
├── eval_mhj.slurm
├── eval_phase3.slurm
├── train_all.slurm               # Full training (5 models)
├── train_phase3.slurm            # Resume phase 3
├── submit_eval_pipeline.sh       # Stage pipeline with dependencies
├── eval_stage{1-6}_*.slurm       # Legacy staged pipeline
└── mhj_loader.py                 # MHJ → v11 schema converter
```

### Data Generation Project (GuardLens-DataGen-V2, separate)
```
build_semantic_datasetv11.py     # Core pipeline: Turn, ConversationSample, generation
interactive_generator.py         # Interactive adversarial generation (Qwen vs Llama)
benign_generator.py              # Separate benign pool generation
causal_analysis.py               # 4-pass causal analysis
merge_validations.py             # Cross-model validation merge
postprocess_causal.py            # Tier relabeling, consistency
split_dataset.py                 # Train/dev/test splitting
inference_backend.py             # Pluggable backend (Ollama, vLLM, HF)
mhj_loader.py                   # MHJ loader
```

---

## 14. Best Next Steps

### Immediate (before paper writing):
1. **Complete human annotation** — 100 conversations, 2 annotators, span labels + IAA. Instructions already drafted.
2. **Run span-level metrics** — post-hoc from existing predictions. Causal span P/R/F1, incidental FPR.
3. **Fix token F1 to use positive-class-only** — currently inflated by majority null class.

### Paper writing order:
1. Results & Ablations section (all data available)
2. Methodology section (pipeline + model)
3. Dataset section (construction + statistics)
4. Introduction + Related Work
5. Abstract + Conclusion

### Stretch goals (if time permits):
- Tier ablation training (construction-only vs +tiers)
- LlamaGuard transfer eval
- Additional MHJ analysis

---

## 15. Writing/Paper Strategy

### Central claim:
> Surface-form keyword deletion is a strong but brittle shortcut. GuardLens provides validated, tier-aware attribution that separates causal adversarial evidence from benign surface risk in multi-turn settings.

### Contribution framing (NOT "we beat baselines"):
1. Interactive multi-turn adversarial generation with cross-model behavioral validation
2. Tiered causal supervision with conservative counterfactual repair
3. Hierarchical attribution model using token, turn, and supervision-tier signals
4. Analysis showing surface-risk baselines are strong but brittle (construction artifact + specificity failure)
5. Deconfounded evaluation methodology for attribution in safety-critical settings

### Key paper tables:
- Table 1: Causal Attribution Comparison (6 methods, DD/Flip/Nec/Suf/Token F1)
- Table 2: Classification Comparison (5 models)
- Table 3: Attribution Utility (DD vs FPR tradeoff)
- Table 4: Deconfounded Evaluation (SR-neutralized, noise-equalized, SR-injected)
- Table 5: External Generalization (MHJ results)
- Table 6: Per-supervision-tier attribution quality
- Table 7: Surface Risk FPR comparison across benign subsets
- Figure 1: Minimality sensitivity curve
- Figure 2: Pivot window accuracy vs exact

### Sections to handle carefully:
- **Surface risk outperformance**: do NOT hide it. Present it, explain the construction artifact, then show MHJ reversal and FPR comparison.
- **MHJ low recall**: lead with attribution transfer, acknowledge classification gap.
- **Pivot weakness**: reframe with window accuracy and causal turn mass.
- **Phase 3**: report honestly, frame as finding about data scale requirements.

---

## 16. Repeatedly Corrected/Clarified Items

1. **Always use SLURM jobs, never command-line GPU commands** — login node has no GPU access.
2. **Surface risk dictionaries must be aligned** — `eval_deconfounded.py` local function must match `causal_eval.py` `RISK_KEYWORDS`. This was the source of the "SR neutralization didn't collapse SR DD" bug.
3. **ConversationDeBERTa needs FlatConversationCollator** — crashes with [B,T,S] input from GuardLensCollator. Fixed in evaluate.py.
4. **evaluate.py must write clean JSON to file** — `--output` flag, stderr for logs. Previous version mixed transformer warnings into stdout JSON.
5. **Unique `--variants-cache` per subset** — without this, ShieldGemma transfer reuses wrong cached variants across subsets.
6. **`--threshold` must be saved as `args.threshold`** — was hardcoded as 0.5 in output JSON.
7. **Phase 3 DID run** (epochs 20-24) but didn't improve over phase 2. I incorrectly said it didn't run based on checkpoint epoch number.
8. **best_attribution.pt vs best_detection.pt** — use the right checkpoint for the right eval. Attribution eval uses best_attribution; classification uses best_detection.
9. **attr_probs not attribution** — model output key is `attr_probs` (or `attr_logits`), not `attribution`. This caused causal turn mass and pivot window to return 0 records.
10. **Turns are NOT in batch metadata** — use `record_by_id[conversation_id]` to look up original turns from records.
11. **SLURM paths**: `RESULTS_DIR=$HOME/work/results/guardlens_v11`, `TEST_DATA=$HOME/staging/dataset_gen_output/splits/test.jsonl`.
12. **combined_deconfound must mark high-SR turns BEFORE neutralization** — otherwise noise equalization has nothing to target (SR=0 after neutralization).
13. **Do not generate more data** — dataset is sufficient. Focus on evaluation quality and paper framing.
14. **Random baseline excluded from transfer comparison** — 15% ablation is destructive enough to flip the external evaluator regardless of which tokens are removed.
