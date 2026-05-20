# GuardLens Codebase Context Pack
**Last updated:** May 20, 2026

Two separate projects share code:
- **GuardLens-DataGen-V2** (`~/projects/GuardLens-DataGen-V2/`): Data generation pipeline
- **GuardLens-Transformer** (`~/projects/GuardLens-Transformer/`): Training + evaluation

---

## PROJECT 1: GuardLens-DataGen-V2 (Data Generation)

---

### `build_semantic_datasetv11.py`
- **Purpose:** Core pipeline module. Defines data structures (`Turn`, `ConversationSample`), generation paths, surface risk scoring, deduplication, JSONL I/O.
- **Key classes/functions:**
  - `Turn` — dataclass: turn_id, role, text, semantic_role, intent_score, surface_risk, is_trigger, trigger_kind, causal_type
  - `ConversationSample` — dataclass: full record schema including v11 fields (supervision_tier, pivot_kind, transfer_tier, etc.)
  - `surface_risk_score(text)` → float — keyword-based risk scoring
  - `sample_to_dict(sample)` / `_dict_to_sample(d)` — serialization
  - `assign_sample_tier(sample)` — assigns supervision tier and loss weight
  - `deduplicate_dataset(records, threshold)` — structural-path-aware dedup
  - `write_jsonl(records, path)` — output
- **Inputs:** Configuration parameters, attack strategy definitions
- **Outputs:** ConversationSample dicts as JSONL
- **Dependencies:** Standard library only (no torch)
- **Known issues:** Surface risk lexicon in this file is the "generation" dictionary. This is DIFFERENT from `causal_eval.py`'s `RISK_KEYWORDS`. This mismatch caused the deconfounding evaluation to not collapse SR scores initially.
- **Changes made:** v10→v11 rewrite with 11 issue fixes + 4 new features (implicit triggers, causal validation, CF analysis, real-world seeds)
- **Pending:** None
- **Pipeline connection:** Foundation module imported by all generation scripts

---

### `interactive_generator.py`
- **Purpose:** Interactive adversarial generation with feedback loop. Qwen-7B/14B (generator) crafts adaptive attacks against live Llama-8B (target).
- **Key classes/functions:**
  - `InteractiveAttackGenerator` — main class
  - `generate_interactive_dataset(n_pairs, ...)` — orchestrates generation
  - `ATTACK_STRATEGIES` — 10 strategies (perspective_shift, academic_framing, etc.)
  - `ADAPTATION_TACTICS` — 10 tactics for adapting based on target responses
- **Inputs:** Generator/target model URLs (vLLM), n_pairs, seed
- **Outputs:** Raw JSONL with `llama_validation` embedded (jailbreak_detected, pivot_turn_id, etc.)
- **Dependencies:** `build_semantic_datasetv11`, `inference_backend`, vLLM servers
- **Known issues:** `best_result` can be None — crash guard added. Sliding window on target chat (12 messages) to prevent context overflow.
- **Changes made:** Added max_unsafe_score tracking, use_as field, 4-GPU parallel generation support
- **Pending:** None
- **Pipeline connection:** Step 1 of data generation. Output feeds into validation (`launch_val.slurm`)

---

### `benign_generator.py`
- **Purpose:** Generates separate validated benign pool. 5 categories: clean_everyday (30%), research_technical (20%), topic_matched_safe (20%), hard_benign (20%), false_lead_benign (10%).
- **Key functions:**
  - `generate_benign_pool(n, ...)` — generates benign conversations
  - Category-specific generators for each benign type
- **Inputs:** Model backend, n records, category distribution
- **Outputs:** JSONL with length-matched 8-32 turn benign conversations
- **Dependencies:** `build_semantic_datasetv11`, `inference_backend`
- **Known issues:** None
- **Changes made:** Added dual-validation (Llama + Mistral) — 721/1000 accepted, 279 rejected → `benign_boundary.jsonl`
- **Pending:** None
- **Pipeline connection:** Produces clean_benign records for final dataset + boundary stress test data

---

### `causal_analysis.py`
- **Purpose:** 4-pass causal analysis pipeline.
- **Key functions:**
  - Pass 1: `annotate_spans()` — LLM span annotation on priority-ordered jailbreak records
  - Pass 2: `pivot_turn_counterfactual()` — whole-turn ablation using max_unsafe_score, Mistral baseline
  - Pass 3: `span_counterfactual()` — span-level neutralization + re-validation
  - Pass 4: `negative_control_validation()` — DECOY/BENIGN_CONTEXT spans should NOT be causal
- **Inputs:** Validated JSONL, inference backend
- **Outputs:** Annotated JSONL with causal_type per turn, span annotations with causal/incidental labels
- **Dependencies:** `build_semantic_datasetv11`, `inference_backend`
- **Known issues:** Message alternation fix in `replay_conversation()`. Stale checkpoint detection. Span annotator prompt refined ("Return 0-6 spans. Do not invent adversarial spans."). `causal_source_tier` stored per pivot-turn CF result.
- **Changes made:** All fixes listed above
- **Pending:** None
- **Pipeline connection:** Step 3. Produces supervision tiers (cf_strong, cf_weak, llm_confirmed, construction)

---

### `merge_validations.py`
- **Purpose:** Merges Llama generation-time validation with Mistral post-validation into unified records.
- **Key functions:**
  - `merge_dataset(interactive_records, qwen14b_records)` — assigns transfer_tier, success_targets, validation_status
- **Inputs:** Interactive generation output + Mistral validation output
- **Outputs:** Merged JSONL with transfer_tier (transfer_success | target_only | cross_only | no_jailbreak)
- **Dependencies:** `build_semantic_datasetv11`
- **Known issues:** None
- **Pending:** None
- **Pipeline connection:** Step 2. Combines cross-model validation before causal analysis

---

### `postprocess_causal.py`
- **Purpose:** Post-processing after causal analysis: relabel causal BENIGN_CONTEXT/DECOY → CONTEXT_BRIDGE, upgrade distributed-causal records, validate tier/weight consistency.
- **Key functions:** Three-step processing
- **Inputs:** Causal-annotated JSONL
- **Outputs:** Cleaned JSONL with consistent tiers
- **Changes made:** 32 spans relabeled, 88 records upgraded, 1,401 weights fixed
- **Pipeline connection:** Step 4. Produces final causal annotations before dataset assembly

---

### `split_dataset.py`
- **Purpose:** Train/dev/test splitting with pair-aware logic.
- **Key functions:**
  - `split_dataset(records, ...)` — stratified split preserving pair_id integrity
- **Inputs:** `final_dataset.jsonl`, optional `--mhj-input`
- **Outputs:** `splits/train.jsonl`, `splits/dev.jsonl`, `splits/test.jsonl`, `human_benchmark.jsonl`
- **Dependencies:** `build_semantic_datasetv11`
- **Known issues:** MHJ was not included in split (External test: 0). MHJ is now used as separate external eval.
- **Pipeline connection:** Step 6. Produces frozen splits for training

---

### `inference_backend.py`
- **Purpose:** Pluggable inference backend abstraction.
- **Key functions:**
  - `create_backend(backend_type, model, base_url)` — factory
  - Backends: `OllamaBackend`, `VLLMBackend`, `HuggingFaceBackend`
  - `backend.generate(prompt, system, temperature, max_tokens)` → str
- **Inputs:** Backend type, model name, URL
- **Outputs:** Generated text
- **Pipeline connection:** Used by all generation/validation scripts

---

### `mhj_loader.py`
- **Purpose:** Converts Multi-turn Human Jailbreak (MHJ) data to v11 schema.
- **Key functions:**
  - `load_mhj_jsonl(path)` — loads JSONL or flat prompt format
  - `convert_mhj_to_v11(record, infer_fields, inference_fn)` — schema conversion
  - `_infer_semantic_fields(turns, inference_fn)` — optional LLM-based field inference
- **Inputs:** MHJ JSONL/CSV
- **Outputs:** v11 JSONL with `is_external_test=True`, `training_eligible=False`, `source_dataset="mhj"`, `label=1`
- **Known issues:** All MHJ records are malicious-only. Classification F1/accuracy is misleading — report recall and mean P(adv).
- **Pipeline connection:** Produces `mhj_external_test.jsonl` for external evaluation only

---

### `augment_dataset.py`
- **Purpose:** Data augmentation (hard negatives, borderline cases, noise injection, false leads).
- **Known issues:** Eventually removed for interactive data — interactive generation produces naturalistic variety without augmentation.
- **Pipeline connection:** Legacy from v10, still available but not used for v11 interactive data

---

### `run_hpc.py`
- **Purpose:** HPC runner wiring inference backend into pipeline classes.
- **Pipeline connection:** Infrastructure for SLURM-based parallel generation

---

### Data Generation SLURM Scripts

| Script | Purpose | GPUs | Time |
|---|---|---|---|
| `launch_gen_interactive.slurm` | Interactive generation (Qwen-14B + Llama-8B via vLLM) | 2 | 18h |
| `launch_val.slurm` | Mistral-7B cross-validation | 1 | 4h |
| `launch_causal.slurm` | 4-pass causal analysis (4-GPU parallel) | 4 | 8h |
| `launch_benign.slurm` | Benign pool generation + dual validation | 1 | 4h |

---

## PROJECT 2: GuardLens-Transformer (Training + Evaluation)

---

### `guardlens/config.py`
- **Purpose:** `GuardLensConfig` dataclass with all hyperparameters.
- **Key fields:**
  - `backbone_name = "microsoft/deberta-v3-base"`
  - `max_turns = 32`, `max_tokens_per_turn = 192`, `max_total_tokens = 2048`
  - `batch_size = 4`, `gradient_accumulation = 4`
  - `learning_rate = 2e-4`, `weight_decay = 0.01`
  - `phase1_epochs = 5`, `phase2_epochs = 15`, `phase3_epochs = 5`
  - `use_pivot_head = True` (disabled for baselines)
  - `fusion_temperature` — temperature for attribution sigmoid
  - `causal_span_labels` includes `CONTEXT_BRIDGE`
- **Changes made:** max_turns=32 (was 16), CONTEXT_BRIDGE added, pos_weight auto-compute, CF oversampling config, pre-split paths
- **Pipeline connection:** Imported by all training and evaluation code

---

### `guardlens/data/dataset.py`
- **Purpose:** Dataset and collator classes.
- **Key classes:**
  - `GuardLensDataset` — converts records to tensors: input_ids [T, S], turn_mask [T], role_ids [T], token_labels [T, S], span_weights [T, S], pivot_labels [T+1], sample_weight, metadata dict
  - `GuardLensCollator` — batches into [B, T, S] tensors, pads turns and tokens
  - `FlatConversationCollator` — for ConversationDeBERTa: concatenates all turns into [B, L]
- **Key behaviors:**
  - `causal_type` is primary signal for token labels (not just span matching)
  - Incidental spans get explicit 0 labels (not -1)
  - Span tier weights: cf_strong=1.0, cf_weak=0.7, construction=0.4, etc.
  - `loss_weight` per sample from supervision tier
  - Pivot labels: turn index for malicious, `no_pivot_class` for benign, -1 for truncated turns
- **Known issues:** Metadata does NOT include turns — use `record_by_id[conversation_id]` to look up original turns
- **Changes made:** v11 rewrite with causal_type, incidental negatives, pivot labels, WeightedRandomSampler support
- **Pipeline connection:** Used by training and all evaluation scripts

---

### `guardlens/models/guardlens.py`
- **Purpose:** Main GuardLens model.
- **Key class:** `GuardLens(nn.Module)`
- **Architecture:**
  - `setup_backbone()` — loads and freezes DeBERTa
  - `encode_turns(input_ids, attention_mask)` → [B, T, S, D] — per-turn DeBERTa encoding
  - `CrossTurnFusion` — TransformerEncoder (2 layers, 4 heads)
  - `attr_head` — per-token attribution (linear → logit)
  - `cls_head` — classification (pooled → linear)
  - `pivot_head` — per-turn pivot logit + no-pivot embedding
- **forward() outputs:**
  ```python
  {
      "cls_logits": [B],          # classification logit
      "attr_logits": [B, T, S],   # raw attribution logits
      "attr_probs": [B, T, S],    # sigmoid(attr_logits / temperature)
      "pivot_logits": [B, T+1],   # pivot turn prediction
  }
  ```
  Note: `attr_probs` and `attr_logits` only present when `compute_attribution=True`
- **Known issues:** Key is `attr_probs` NOT `attribution` — this caused eval_attribution_utility.py to return 0 records until fixed
- **Changes made:** Pivot head shape fix (.squeeze removed), gate_weights in fusion
- **Pipeline connection:** Main model for training and evaluation

---

### `guardlens/models/baselines.py`
- **Purpose:** Baseline model variants.
- **Key classes:**
  - `ConversationDeBERTa` — flat concatenation, takes [B, L] input. **Must use FlatConversationCollator.**
  - `TurnLevelClassifier` — independent per-turn classification, no cross-turn reasoning
  - `GuardLensNoFusion` — GuardLens without CrossTurnFusion
  - `GuardLensNoCF` — GuardLens architecture, phase3_epochs=0
- **Known issues:** ConversationDeBERTa crashes with GuardLensCollator ([B,T,S] input). Fixed in evaluate.py with model_name check.
- **Pipeline connection:** Ablation baselines

---

### `guardlens/models/__init__.py`
- **Purpose:** Model registry.
- **Key dict:** `MODEL_REGISTRY = {"guardlens": GuardLens, "guardlens_no_fusion": ..., ...}`
- **Pipeline connection:** Used by all scripts to instantiate models from checkpoint `model_name`

---

### `guardlens/models/components.py`
- **Purpose:** Shared model components.
- **Key classes:** `TurnEncoder`, `CrossTurnFusion`, `AttributionHead`, `ClassificationHead`
- **No changes made in this conversation.**

---

### `guardlens/training/trainer.py`
- **Purpose:** Training loop and evaluation function.
- **Key functions:**
  - `train_epoch(model, loader, optimizer, scheduler, loss_fn, config, epoch, device)` → metrics dict
  - `evaluate(model, loader, loss_fn, config, device, threshold)` → results dict with f1, accuracy, attr_f1, pivot_accuracy, per-tier breakdowns
- **Key behaviors:**
  - 3-phase training: phase determined by `get_current_phase(epoch, config)`
  - OOM fallback (gradient accumulation on OOM)
  - Dev threshold tuning (scans 0.05-0.95)
  - Dual checkpoints: `best_detection.pt` (best F1) + `best_attribution.pt` (best composite=0.3*F1+0.7*attrF1)
  - Per-transfer-tier, per-benign-status, per-supervision-tier evaluation
  - Baselines skip Phase 3, disable attr/cf/pivot losses
- **Known issues:** `evaluate()` passes `turn_mask` and `role_ids` to all models — ConvDeBERTa accepts via **kwargs and ignores. This works but is implicit.
- **Changes made:** Pre-split loading, auto pos_weight, CF oversampling, truncated pivot uses -1 (ignore_index), pivot-kind loss filters, zero-weight spans excluded, CF loss in logged total
- **Pipeline connection:** Core training loop

---

### `guardlens/training/loss.py`
- **Purpose:** Multi-task loss function.
- **Key class:** `GuardLensLoss`
- **Loss components:**
  - Classification: BCE with pos_weight, sample-weighted (normalized by weight sum)
  - Attribution: BCE per-token, span-tier-weighted (zero-weight excluded)
  - Pivot: CrossEntropy over [T+1] classes
  - CF: contrastive loss between original and counterfactual representations (phase 3 only)
- **Changes made:** Per-tier logging, sample weight normalization, cf_loss included in total
- **Pipeline connection:** Used by trainer

---

### `guardlens/training/schedule.py`
- **Purpose:** Phase and lambda scheduling.
- **Key functions:**
  - `get_current_phase(epoch, config)` → 1/2/3
  - `get_lambda_schedule(epoch, config)` → (lambda_cls, lambda_attr, lambda_cf)
- **No changes made in this conversation.**

---

### `guardlens/data/splits.py`
- **Purpose:** `pair_aware_split(records, seed)` — stratified split preserving pair_id.
- **Used as fallback** when `--test-path` not provided. All eval scripts now prefer `--test-path`.

---

### `guardlens/evaluate.py`
- **Purpose:** Classification evaluation entry point.
- **Key function:** `main()` — loads checkpoint, evaluates, writes JSON
- **CLI args:** `--test-path`, `--data` (fallback), `--checkpoint`, `--output` (clean JSON file), `--batch-size`, `--device`
- **Key behaviors:**
  - Reads `threshold` from checkpoint
  - Uses `FlatConversationCollator` for `conversation_deberta`
  - Disables `use_pivot_head` for baselines
  - Prints to stderr, writes clean JSON to `--output`
- **Known issues fixed:**
  - Previously mixed transformer warnings into stdout JSON
  - Previously crashed on ConvDeBERTa due to wrong collator
  - Previously ignored dev-tuned threshold
- **Pipeline connection:** Called by eval_full.slurm for all 5 models

---

### `guardlens/eval_causal.py`
- **Purpose:** Causal attribution evaluation entry point.
- **CLI args:** `--test-path`, `--checkpoint`, `--output`, `--methods`, `--top-k`, `--tier-eval`, `--transfer-eval`
- **Key behaviors:**
  - Runs `run_causal_evaluation()` from `causal_eval.py`
  - Per-supervision-tier breakdown (cf_weak, construction, llm_confirmed)
  - Per-transfer-tier breakdown (transfer_success, target_only, cross_only, contextual_pivot, lexical_pivot)
  - Saves LaTeX table
- **Known issues fixed:** numpy import moved to top (was latent bug)
- **Pipeline connection:** Called by eval_full.slurm and eval_core.slurm

---

### `guardlens/evaluation/causal_eval.py`
- **Purpose:** Core causal evaluation framework. Implements all attribution methods and metrics.
- **Key functions:**
  - `run_causal_evaluation(model, loader, device, methods, top_k_fractions, tokenizer)` → dict per method
  - Attribution methods:
    - `guardlens_attribution()` — uses `outputs["attr_probs"]`
    - `surface_risk_attribution()` — uses `RISK_KEYWORDS` (token-level keyword matching). **This is the canonical surface risk function.**
    - `attention_attribution()` — hooks transformer attention (falls back to representation-change proxy)
    - `integrated_gradients_attribution()` — multi-step gradient integration
    - `grad_x_input_attribution()` — gradient × input embedding
    - `random_attribution()` — uniform random
  - `_get_prob()` — gets model probability with optional attribution_mask
  - Metrics: deviation_drop, flip_rate, necessity, sufficiency, token_f1, pivot_turn_accuracy
- **RISK_KEYWORDS set:** `{"bypass", "exploit", "ignore", "override", "evasion", "evade", "inject", "injection", "jailbreak", "unsafe", "unrestricted", "detectable", "convincing", "realistic", "blend", "plausible", "success", "effective", "credential", "phishing", "malware", "exfiltrate", "manipulate", "persuasive", "stealth", "payload", "disguise", "impersonate", "pretexting"}`
- **Known issues:**
  - Token F1 is inflated by majority null-class matching (random gets ~0.62). Needs positive-class-only F1 or AUPRC.
  - Attention may be representation-change proxy, not actual attention weights. Report as such.
  - Uses embedding masking (attribution_mask) for internal eval. External models need `eval_external.py` with token-removal instead.
- **Pipeline connection:** Core framework imported by all causal eval scripts

---

### `guardlens/evaluation/eval_utils.py`
- **Purpose:** Shared evaluation utilities.
- **Key functions:**
  - `load_test_data(test_path, data_path, seed)` → (records, test_idx) — pre-split or fallback
  - `add_test_path_args(parser)` — adds --test-path and --data args
  - `partition_test_set_v11(records)` → dict of subset_name → indices (contextual_pivot, lexical_pivot, distributed, hard_benign, false_lead, clean_benign, transfer_success, target_only, cross_only)
  - `partition_by_supervision_tier(records)` → dict of tier → indices
  - `print_subset_summary(subsets, records)` — prints n/mal/ben per subset
  - `results_to_latex_table(results, ...)` → LaTeX string
  - `comparison_to_latex(all_results, ...)` → LaTeX string
- **Pipeline connection:** Imported by all eval scripts for consistent data loading and subsetting

---

### `guardlens/evaluation/eval_implicit_explicit.py`
- **Purpose:** Implicit vs explicit trigger subset analysis. The key experiment for surface risk contrast.
- **Key function:** `main()` — runs causal eval on contextual_pivot, lexical_pivot, hard_benign, false_lead, clean_benign subsets
- **Uses:** `partition_test_set_v11()` with v11 fields (pivot_kind, benign_status)
- **Changes made:** Complete rewrite from v10 (implicit_trigger → pivot_kind, old families → v11 benign_status)
- **Pipeline connection:** Called by eval_full.slurm step 6

---

### `guardlens/evaluation/eval_attribution_precision.py`
- **Purpose:** Hard-negative attribution precision + minimality sensitivity curve.
- **Key outputs:**
  - FPAR (False Positive Attribution Rate) per subset
  - Minimality curve: flip rate at k=2%-30% for guardlens, surface_risk, random
  - Inflection point, slope, AUC
- **Changes made:** Updated family names for v11 (validated_benign_twin, hard_benign, false_lead_benign, research_technical), pre-split support
- **Pipeline connection:** Called by eval_stage6 SLURM

---

### `guardlens/evaluation/eval_cross_dataset.py`
- **Purpose:** Cross-dataset generalization. Extends single-turn seeds from AdvBench/HarmBench to 5-7 turn conversations and evaluates.
- **Key behavior:** Loads seeds, extends to multi-turn via LLM, mixes with test benign, evaluates GuardLens/NoCF/ConvDeBERTa
- **Changes made:** Benign pool loaded from test split directly (no re-splitting), `--benign-pool` accepts test.jsonl
- **Known issues:** Extension to 5-7 turns doesn't match training distribution (28 turns). Consider 15-20 turn extensions.
- **Pipeline connection:** Called by eval_stage3 SLURM

---

### `guardlens/evaluation/eval_cross_model_transfer.py`
- **Purpose:** Cross-model transfer evaluation using ShieldGemma-9B.
- **Key workflow:**
  - Phase 1: Load GuardLens, extract attribution, create ablated variants (guardlens_ablated, surface_risk_ablated, random_ablated)
  - Phase 2: Load ShieldGemma-9B (8-bit), score all variants
  - Compute transfer flip rate: does masking attributed tokens flip ShieldGemma's assessment?
- **CLI args:** `--external-model` (was --llamaguard-model), `--threshold` (was hardcoded 0.5), `--subset` (contextual_pivot, lexical_pivot, transfer_success), `--variants-cache` (unique per subset!)
- **Changes made:**
  - `--llamaguard-model` → `--external-model`
  - `args.llamaguard_model` → `args.external_model`
  - `--threshold` CLI arg added, `args.threshold` saved in output JSON
  - Subset choices updated to v11 fields
  - `has_implicit` uses `pivot_kind` not `implicit_trigger`
- **Critical requirement:** Each subset run MUST use unique `--variants-cache` path, otherwise subsets reuse wrong cached variants
- **Pipeline connection:** Called by eval_transfer_stable.slurm

---

### `guardlens/evaluation/eval_external.py`
- **Purpose:** Self-eval vs external-eval comparison. Tests whether GuardLens attribution transfers to other models' decision boundaries.
- **Changes made:** Pre-split support, v11 subset choices (contextual_pivot, lexical_pivot, transfer_success)
- **Pipeline connection:** Called by eval_stage2 SLURM

---

### `guardlens/evaluation/eval_paraphrase.py`
- **Purpose:** Paraphrase robustness. Measures attribution stability under paraphrase.
- **Key metrics:** Turn-level Spearman ρ, Token-level Spearman ρ, top-15% token stability, pivot turn stability
- **Changes made:** Fixed broken lexical_pivot filter (was double-filtering with implicit_trigger), `has_implicit` uses pivot_kind, pre-split support
- **Pipeline connection:** Called by eval_stage4 SLURM

---

### `guardlens/evaluation/eval_boundary_stress.py`
- **Purpose:** Classification on rejected boundary benign records (hardest negatives excluded from training).
- **Key outputs:** Accuracy, FPR, per-family breakdown, mean/p95/max P(adv)
- **NEW file** created in this conversation
- **Pipeline connection:** Called by eval_full.slurm step 4

---

### `guardlens/evaluation/eval_surface_risk_fpr.py`
- **Purpose:** Compute surface risk baseline FPR across benign subsets. No GPU needed.
- **Key function:** `compute_fpr_at_threshold(records, threshold)` — computes max SR across user turns, classifies, reports FPR
- **Outputs:** FPR at thresholds 0.3/0.4/0.5 for all_benign, false_lead, research_technical, hard_benign, boundary_rejected, and per-boundary-family subsets
- **NEW file** created in this conversation
- **Pipeline connection:** Called by eval_full.slurm step 1

---

### `guardlens/evaluation/eval_attribution_utility.py`
- **Purpose:** Three post-hoc metrics: Attribution Utility, Causal Turn Mass, Pivot Window Accuracy.
- **Key functions:**
  - `compute_utility_table(causal_results, gl_fpr, sr_fpr_thresholds, lambdas)` — Utility = DD@k - λ×FPR
  - `compute_causal_turn_mass(model, records, ...)` — fraction of attribution mass on causal turns (pivot/escalation/payload)
  - `compute_pivot_window_accuracy(model, records, ..., windows)` — pivot prediction within ±W turns
- **Key implementation details:**
  - Uses `outputs.get("attr_probs")` (not `"attribution"`)
  - Looks up turns via `record_by_id[conversation_id]` (not from metadata)
  - Skip counter warns if batches are dropped (missing attr output)
  - Pivot window: strict — skips record if pivot maps to non-user turn (no fuzzy approximation)
  - Causal mass uses single `is_causal_turn` check (no double-counting), clamped to max 1.0
  - Computes both `utility_all_benign` and `utility_boundary` variants
  - Headline utility shows only GuardLens vs Surface Risk (neural methods have FPR=0, not comparable)
- **NEW file** created in this conversation
- **Pipeline connection:** Called by eval_deconfound.slurm step 1

---

### `guardlens/evaluation/eval_deconfounded.py`
- **Purpose:** Deconfounded test variants to eliminate surface risk construction artifacts.
- **Four variants:**
  - **A: SR-neutralized** — replace RISK_KEYWORDS with semantically equivalent low-risk phrases
  - **B: Noise-equalized** — add typos to pristine adversarial turns (breaks grammar-label correlation)
  - **C: SR-injected benign** — inject adversarial vocabulary into clean benign records in safe contexts
  - **D: Combined** — marks high-SR turns BEFORE neutralization, applies noise to those turns after
- **Key implementation details:**
  - `surface_risk_score()` aligned with `causal_eval.py` `RISK_KEYWORDS` (not the generation pipeline dictionary)
  - SR-injected FPR computed from records directly (not batch metadata)
  - `neutralized_changed` slice: only records where phrases were actually replaced
  - Adversarial sanity check: counts how many neutralized records still detected as adversarial
  - Threshold loaded once from checkpoint, printed explicitly
- **SR_REPLACEMENTS dictionary:** covers all RISK_KEYWORDS plus generation pipeline keywords. Replacements avoid any word in RISK_KEYWORDS (e.g., "jailbreak" → "circumvent the rules" not "circumvent restrictions")
- **NEW file** created in this conversation
- **Pipeline connection:** Called by eval_deconfound.slurm step 2

---

### `guardlens/evaluation/__init__.py`
- **Purpose:** Exports from evaluation package.
- **Exports:** `run_causal_evaluation`, `print_comparison_table`, `ATTRIBUTION_METHODS`, `load_test_data`, `add_test_path_args`, `partition_test_set_v11`, `partition_by_supervision_tier`, `results_to_latex_table`, `comparison_to_latex`

---

### SLURM Scripts (Training)

#### `train_all.slurm`
- **Purpose:** Full training of all 5 model variants sequentially on 1 GPU.
- **Duration:** ~2.5 hours total
- **Config:** 25 epochs GuardLens, 20 epochs baselines, batch=4, A100 80GB
- **Output:** Checkpoints in `$HOME/work/results/guardlens_v11/checkpoints/{model_name}/`

#### `train_phase3.slurm`
- **Purpose:** Resume Phase 3 CF training from Phase 2 checkpoint.
- **Key config:** Forces `phase3_epochs=5`, `max_epochs=phase1+phase2+5`, LR=0.1× base, CF oversampling 3×/1.5×
- **Result:** Phase 3 ran but didn't improve (attrF1 0.890 vs phase 2 best 0.887). CF signal too sparse.

---

### SLURM Scripts (Evaluation)

#### `eval_full.slurm` (MASTER)
- **Purpose:** Complete evaluation from scratch. Uses correct checkpoints.
- **Runs in order:** SR FPR (no GPU) → Classification (5 models, best_detection.pt) → Core causal (best_attribution.pt, 6 methods + tier/transfer) → Boundary stress → MHJ → Implicit/explicit subset
- **Duration:** ~8 hours on 1 GPU
- **Outputs:** 13+ JSON files in `$HOME/work/results/guardlens_v11/results/`

#### `eval_deconfound.slurm`
- **Purpose:** Attribution utility metrics + deconfounded test variants.
- **Prerequisite:** eval_full.slurm must have completed (needs causal_eval_results.json, boundary_stress.json, surface_risk_fpr.json)
- **Duration:** ~5 hours
- **Uses:** `set -euo pipefail` with `|| true` on summary blocks

#### `eval_transfer_stable.slurm`
- **Purpose:** Stabilized ShieldGemma transfer with subset filtering + threshold sensitivity.
- **Uses:** `final_dataset.jsonl` (not test split) for maximum adversarial coverage
- **Runs:** A (transfer_success), B (lexical_pivot), C (contextual_pivot), D (threshold 0.3/0.4/0.5)
- **Each subset gets unique --variants-cache**

#### `eval_core.slurm`
- **Purpose:** Core causal + classification only (subset of eval_full)

#### `eval_boundary.slurm`
- **Purpose:** Standalone boundary stress test

#### `eval_mhj.slurm`
- **Purpose:** MHJ processing + classification + causal eval
- **Needs:** `MHJ_SOURCE` env var pointing to raw MHJ data

#### `eval_phase3.slurm`
- **Purpose:** Phase 2 vs Phase 3 comparison (classification + causal + boundary)

#### `eval_stage{1-6}_*.slurm` + `submit_eval_pipeline.sh`
- **Purpose:** Legacy staged pipeline with SLURM dependencies. Superseded by eval_full.slurm.
- **Status:** Still functional but eval_full.slurm is the preferred single-job approach.

---

## Key Cross-File Dependencies

```
causal_eval.py
  └── RISK_KEYWORDS ← must match eval_deconfounded.py SR_REPLACEMENTS
  └── surface_risk_attribution() ← the canonical SR baseline
  └── guardlens_attribution() ← reads outputs["attr_probs"]

eval_attribution_utility.py
  └── reads outputs["attr_probs"] or outputs["attr_logits"]
  └── looks up turns via record_by_id[conversation_id]
  └── reads causal_eval_results.json, boundary_stress.json, surface_risk_fpr.json

evaluate.py
  └── model_name == "conversation_deberta" → FlatConversationCollator
  └── ckpt.get("threshold") → classification threshold
  └── --output → clean JSON (no stdout pollution)

dataset.py
  └── GuardLensCollator → [B, T, S] tensors
  └── FlatConversationCollator → [B, L] tensors
  └── metadata does NOT include turns

guardlens.py
  └── forward() returns: cls_logits, attr_logits, attr_probs, pivot_logits
  └── attr_probs only when compute_attribution=True
```

---

## Environment Details

```
HPC: University cluster, 4× A100 80GB PCIe
SLURM account: V_cs_hat_capstone_mkhan74
Partition: defq
Conda env: $HOME/work/conda_envs/guardlens_train
Python: 3.11
PyTorch: 2.x with CUDA
Transformers: recent (DeBERTa-v2 support)
Models cached at: $HOME/work/hf_models/hub (HF_HUB_OFFLINE=1)

Key paths:
  Checkpoints: $HOME/work/results/guardlens_v11/checkpoints/
  Results:     $HOME/work/results/guardlens_v11/results/
  Test data:   $HOME/staging/dataset_gen_output/splits/test.jsonl
  Final data:  $HOME/staging/dataset_gen_output/final_dataset.jsonl
  Boundary:    $HOME/staging/dataset_gen_output/benign_boundary.jsonl
  MHJ:         $HOME/staging/dataset_gen_output/mhj_external_test.jsonl
```
