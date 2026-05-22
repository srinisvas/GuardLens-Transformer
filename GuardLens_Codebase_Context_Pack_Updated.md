# GuardLens Codebase Context Pack

Portable handoff focused on files, scripts, modules, and known code state for the GuardLens / NN Token Attribution project.

This file is intended to be pasted into a new ChatGPT or Claude chat so the next assistant can continue coding/reviewing without losing context.

---

## 1. High-level codebase map

The codebase supports the full GuardLens pipeline:

```text
interactive generation
  -> validation
  -> causal analysis / span annotation
  -> postprocessing
  -> benign pool merge
  -> train/dev/test split
  -> model training
  -> core causal evaluation
  -> boundary / surface-risk / external / deconfounded evaluation
  -> human benchmark
  -> paper tables
```

Main directories implied by code/logs:

```text
GuardLens-DataGen-V2/
GuardLens-Transformer/
guardlens/
guardlens/data/
guardlens/models/
guardlens/training/
guardlens/evaluation/
logs/
results/
staging/dataset_gen_output/
work/results/guardlens_v11/
```

Important cluster paths:

```text
$HOME/staging/dataset_gen_output/
$HOME/work/results/dataset_gen/
$HOME/work/results/guardlens_v11/
$HOME/work/hf_models/
$HOME/work/conda_envs/dataset_gen
$HOME/work/conda_envs/guardlens_train
```

Main conda envs:

```text
dataset_gen
guardlens_train
```

Main model cache:

```text
$HOME/work/hf_models
```

Offline mode is frequently used:

```bash
HF_HUB_OFFLINE=1
HF_HOME=$HOME/work/hf_models
TRANSFORMERS_CACHE=$HOME/work/hf_models/hub
```

---

## 2. Dataset generation and validation files

### `build_semantic_datasetv11.py`

**Purpose**  
Main interactive v11 dataset generator. Builds multi-turn adversarial and benign conversations using a generator model and a target model with feedback/adaptation.

**Important functions/classes/objects**
- Interactive generation loop
- Judge prompt and behavior compliance mapping
- Known introspection:
  - `_JUDGE_SYSTEM_PROMPT`: exists
  - `_BEHAVIOR_TO_COMPLIANCE`: exists
  - `_JUDGE_SYSTEM`: does not exist
- Likely includes `annotate_turn()` or related legacy annotation logic.

**Inputs**
- Generator server URL, usually Qwen:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `Qwen/Qwen2.5-14B-Instruct`
- Target server URL:
  - `meta-llama/Meta-Llama-3-8B-Instruct`
- Number of pairs
- Output JSONL path

**Outputs**
- Raw interactive JSONL, each pair usually produces:
  - malicious conversation
  - benign twin conversation
- Example outputs:
  - `semantic_multiturn_v11_interactive_raw.jsonl`
  - `semantic_multiturn_v11_interactive_14b_raw.jsonl`
  - shard files such as `semantic_multiturn_v11_interactive_shard0.jsonl`

**Dependencies**
- vLLM OpenAI-compatible servers
- Qwen generator
- Llama target
- JSONL writer
- Local judge logic

**Known bugs/issues**
- Legacy span annotation based on keyword/template matching does not work well for interactive natural phrasing.
- Interactive conversations may contain consecutive user turns when target response/adaptation failed, which caused replay problems later.
- Surface-risk dictionaries influenced generation, creating lexical leakage concerns.
- Some payload turns were forced to include high surface-risk phrases such as “improve success rate” or “less detectable”.

**Changes already made**
- Interactive feedback-loop generation became primary method.
- Template augmentation removed from interactive generation.
- Generation model made configurable.
- 7B generator accepted as viable after strong jailbreak rate with feedback loop.

**Pending changes**
- Do not use old keyword span annotation as primary source for interactive data.
- If further generation is done, make surface-risk phrase inclusion less deterministic.
- Consider grammar/noise equalization during generation only if new data is generated, but current recommendation is no more generic generation.

**Pipeline connection**
Generation source for final v11 adversarial and benign twin data.

---

### `launch_gen.slurm`

**Purpose**  
SLURM job for generation.

**Inputs**
- Generator model
- Target model
- Pair count
- GPU allocation
- Output directory

**Outputs**
- Raw interactive JSONL files
- vLLM logs

**Dependencies**
- vLLM
- A100 GPUs
- HF model cache
- `build_semantic_datasetv11.py`

**Known bugs/issues**
- Older versions included augmentation step, which is now deprecated.
- Earlier GPU allocation was inefficient with unused GPUs.
- Sequential generation was slow: 50 pairs took ~139-159 minutes depending on generator/model.

**Changes already made**
- Job made customizable for generator model.
- Parallel pipeline plan used for 7B generation.
- Augmentation removed for interactive generation.

**Pending changes**
- Ensure no augmentation step remains.
- Ensure output path explicitly names generator/model/version.

**Pipeline connection**
Launches interactive generation.

---

### `launch_gen (1).slurm`

**Purpose**  
Modified generation SLURM compared against `launch_gen.slurm`.

**Important note**
User dislikes confusion when filenames are similar. Be precise when comparing `(1)` versions.

**Known issues**
- Need to verify exact diff before making claims.
- Do not assume `(1)` file is final without checking content.

**Pipeline connection**
Generation job variant.

---

### `launch_val.slurm`

**Purpose**  
Validation SLURM for validating generated records against a validation model, especially Mistral or Qwen.

**Inputs**
- `INPUT_FILE`
- `OUTPUT_NAME`
- `VAL_MODEL`
- HF offline flag
- number of shards / GPUs

**Outputs**
- Validated JSONL:
  - e.g., `semantic_multiturn_v11_interactive_cf_validated.jsonl`
  - shard files such as `*_valshard0_validated.jsonl`

**Dependencies**
- vLLM
- validation model:
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `Qwen/Qwen2.5-7B-Instruct`
- validation script

**Known bugs/issues**
- Earlier validation/counterfactual replay broke on consecutive user turns.
- Must ensure shard merging preserves all records.

**Changes already made**
- Parallel validation enabled.
- Mistral validation completed for 7B and 14B generated data.

**Pending changes**
- Keep shard files until final verification.
- Clean up old shard files only after merge is validated.

**Pipeline connection**
Converts generated conversations into behaviorally validated records and transfer tiers.

---

### `merge_validations.py`

**Purpose**  
Merges validation outputs and computes transfer-tier statistics.

**Inputs**
- Interactive validated JSONL
- Output path

Example:

```bash
python3 merge_validations.py   --interactive $HOME/staging/dataset_gen_output/semantic_multiturn_v11_interactive_validated.jsonl   --output $HOME/staging/dataset_gen_output/merged_7b.jsonl
```

**Outputs**
- `merged_7b.jsonl`
- merged statistics including transfer tiers and jailbreak success.

**Important output stats from 7B merge**
- Total: 1100 records
- 550 malicious / 550 benign
- transfer_success: 227
- cross_only: 167
- target_only: 24
- no_jailbreak: 132
- benign: 550
- Llama target jailbreak: 251/550 = 45.6%
- Qwen transfer: 394/550 = 71.6%
- both: 227/550 = 41.3%
- any validator: 418/550 = 76.0%

**Dependencies**
- JSONL validation outputs
- fields such as `causal_validation`, `jailbreak_detected`, transfer metadata

**Known bugs/issues**
- If validators use different schemas, merge may misclassify.
- Must not treat cross_only as same quality as transfer_success for causal repair.

**Changes already made**
- Used to merge 7B validation results.
- Transfer-tier stats extracted.

**Pending changes**
- Ensure merged fields are standardized before split/training.

**Pipeline connection**
Creates merged generation/validation dataset for later causal analysis/postprocess.

---

## 3. Causal analysis and postprocessing files

### `launch_causal.slurm`

**Purpose**  
Runs 4-pass causal analysis in parallel over combined dataset.

**Inputs**
- `INPUT_FILE`, e.g. `combined_all.jsonl`
- `OUTPUT_NAME`, e.g. `causal_analyzed_all_final`
- Mistral validator server(s)
- shard count

**Outputs**
- `causal_analyzed_all_final.jsonl`
- shard files:
  - `causal_analyzed_all_final_shard0.jsonl`
  - etc.

**Dependencies**
- Mistral vLLM servers, one per GPU
- causal analysis script
- conversation replay
- LLM span annotation

**Known bugs/issues**
- Replay previously failed on malformed message sequences with consecutive user turns.
- Must normalize user/assistant alternation for Mistral chat template.
- Sparse CF output due to strict causal validation.

**Changes already made**
- Fixed replay issue.
- Full fresh 4-GPU run completed.

**Key output stats**
- Input: 1500 records
- Supervision tiers:
  - construction: 1204
  - llm_confirmed: 255
  - cf_weak: 30
  - cf_strong: 11
- Pivot-turn CF:
  - none: 487
  - distributed_or_unclear: 223
  - cf_turn_weak: 28
  - cf_turn_strong: 12
- Spans:
  - causal: 62
  - incidental: 146
  - unvalidated: 4333
- Training eligible: 1041/1500
- Avg loss weight: 0.583

**Pipeline connection**
Adds LLM span annotations, CF turn labels, span causal/incidental labels, and validation tiers.

---

### `postprocess_causal.py`

**Purpose**  
Postprocesses causal malicious data after 4-pass causal analysis.

**Inputs**
- `causal_analyzed_all_final.jsonl`

Example:

```bash
python3 postprocess_causal.py   --input $HOME/staging/dataset_gen_output/causal_analyzed_all_final.jsonl   --output $HOME/staging/dataset_gen_output/malicious_final.jsonl
```

**Outputs**
- `malicious_final.jsonl`

**Dependencies**
- causal analysis fields:
  - supervision tiers
  - causal spans
  - incidental spans
  - validation status
  - training eligibility
  - loss weights

**Known bugs/issues**
- Must ensure `BENIGN_CONTEXT` causal spans are relabeled or handled as `CONTEXT_BRIDGE`.
- Must preserve `causal_type`.
- Must not drop sparse CF labels.

**Changes already made**
- Used before final dataset merge.

**Pending changes**
- Verify final records preserve all fields needed by training/eval:
  - `loss_weight`
  - `supervision_tier`
  - `causal_type`
  - `pivot_kind`
  - `transfer_tier`

**Pipeline connection**
Prepares malicious examples for final dataset.

---

## 4. Benign pool and final split files

### `benign_pool_v11_llama_validated.jsonl`

**Purpose**  
Benign pool validated using Llama target model.

**Inputs**
- Generated benign records

**Outputs**
- Benign records with validation metadata.

**Known issues**
- A benign can be rejected by validator if it looks unsafe or prompts refusal.
- Rejected benign should not be trained as clean benign.

**Pipeline connection**
One half of dual-validator clean benign filtering.

---

### `benign_pool_v11_mistral_validated.jsonl`

**Purpose**  
Benign pool validated using Mistral.

**Pipeline connection**
Second half of dual-validator clean benign filtering.

---

### `benign_clean.jsonl`

**Purpose**  
Clean benign records accepted by both Llama and Mistral validators.

**Inputs**
- Llama benign validation
- Mistral benign validation

**Outputs**
- Records marked:
  - `benign_status = clean_benign`
  - `validation_status = validated`
  - `training_eligible = True`

**Pipeline connection**
Merged into final dataset with malicious records.

---

### `benign_boundary.jsonl`

**Purpose**  
Rejected/boundary benign records, not used for training.

**Inputs**
- Benign pool records flagged by either validator.

**Outputs**
- Records marked:
  - `benign_status = benign_boundary_rejected`
  - `validation_status = rejected`
  - `training_eligible = False`
  - `rejected_by = ["llama", "mistral"]` depending on validator

**Pipeline connection**
Used for boundary stress test and surface-risk FPR comparison.

---

### `final_dataset.jsonl`

**Purpose**  
Final merged dataset.

**Inputs**
- `malicious_final.jsonl`
- `benign_clean.jsonl`

**Output stats**
- Total: 1762 records

**Pipeline connection**
Input to final split.

---

### `split_dataset.py`

**Purpose**  
Creates train/dev/test and human benchmark splits.

**Inputs**
- final dataset
- optional `--mhj-input`
- output dir
- human benchmark counts

Example:

```bash
python3 split_dataset.py   --input $HOME/staging/dataset_gen_output/final_dataset.jsonl   --output-dir $HOME/staging/dataset_gen_output/splits/   --human-benchmark 100   --double-annotated 50
```

**Outputs**
- `splits/train.jsonl`
- `splits/dev.jsonl`
- `splits/test.jsonl`
- human benchmark files
- double annotation files

**Important final split stats**
- Loaded 1762 records
- Internal records: 1762
- External test: 0
- Pair groups: 1463

Train:
- 1220 records
- label 1: 383
- label 0: 837
- supervision tiers:
  - llm_confirmed: 237
  - construction: 452
  - cf_weak: 22
  - cf_strong: 7
  - benign_validated: 502

Dev:
- 266 records
- label 1: 79
- label 0: 187

Test:
- 276 records
- label 1: 87
- label 0: 189

Human benchmark:
- single annotated: 100
- double annotated: 50

**Known bugs/issues**
- MHJ was not included because `--mhj-input` was not passed.
- Decision made not to re-split with MHJ.
- MHJ should be external-only.

**Changes already made**
- Uses pair-aware and stratified grouping.
- Current frozen splits accepted.

**Pending changes**
- Do not re-split unless absolutely necessary.
- Keep human benchmark excluded from training/model selection.

**Pipeline connection**
Creates final training/eval splits.

---

## 5. MHJ files

### `mhj_loader.py`

**Purpose**  
Converts MHJ multi-turn data into GuardLens schema.

**Inputs**
- MHJ JSONL file

**Outputs**
- `mhj_external_test.jsonl`

Expected fields:
```json
{
  "source_dataset": "MHJ",
  "is_external_test": true,
  "training_eligible": false,
  "label": 1,
  "transfer_tier": "external_mhj",
  "supervision_tier": "external_eval",
  "pivot_kind": "unknown_or_mhj",
  "turns": [...]
}
```

**Dependencies**
- MHJ dataset file
- GuardLens schema conventions

**Known bugs/issues**
- Do not feed MHJ through single-turn extension pipeline because MHJ is already multi-turn.
- If MHJ is malicious-only, classification F1/accuracy is misleading. Report recall and attribution on detected subset.

**Changes already made**
- Decision made to keep MHJ separate external eval.

**Pending changes**
- Ensure MHJ labels and conversation turn roles are correct.
- Report MHJ as external stress test, not internal test.

**Pipeline connection**
External multi-turn generalization evaluation.

---

## 6. Training files

### `guardlens/train.py`

**Purpose**  
Training entry point, v11-compatible.

**Important args**
```bash
--train-path
--dev-path
--test-path
--data
--output
--model
--backbone
--batch-size
--lr
--epochs
--max-turns
--max-tokens
--seed
--device
--workers
--no-pivot-head
--no-oversample
--no-threshold-tune
```

**Supported models**
```python
["guardlens", "guardlens_no_fusion", "guardlens_no_cf", "turn_level", "conversation_deberta"]
```

**Inputs**
- Pre-split JSONL paths preferred
- Optional fallback single-file data path

**Outputs**
- Checkpoints per model:
  - `best_detection.pt`
  - `best_attribution.pt`
  - sometimes `best_phase2.pt`
  - test result JSON/logs

**Dependencies**
- `GuardLensConfig`
- `trainer.train`
- model registry

**Known bugs/issues**
- Must ensure model registry has exact keys:
  - `guardlens_no_fusion`
  - `guardlens_no_cf`
- Fallback re-split should not be used for final experiments.

**Changes already made**
- Supports pre-split files.
- Defaults:
  - batch size 4
  - max turns 32
  - max tokens 192
- Feature flags added.

**Pending changes**
- None critical.
- Ensure clean checkpoint naming for paper.

**Pipeline connection**
Launches model training.

---

### `guardlens/config.py`

**Purpose**  
Dataclass config for model/training/eval.

**Important fields**
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

use_gated_fusion = True
fusion_temperature = 1.0

max_turns = 32
max_tokens_per_turn = 192
max_total_tokens = 2048

learning_rate = 2e-4
weight_decay = 0.01
warmup_steps = 200
max_epochs = 25
batch_size = 4
gradient_accumulation = 4
max_grad_norm = 1.0

lambda_cls = 0.2
lambda_attr = 1.0
lambda_cf = 0.5
lambda_pivot = 0.3

phase1_epochs = 5
phase2_epochs = 15
phase3_epochs = 5
```

**Causal labels**
```python
causal_span_labels = (
    "MALICIOUS_TRIGGER", "PAYLOAD_SPAN", "CONTEXT_BRIDGE",
    "IMPLICIT_TRIGGER", "STRUCTURAL_TRIGGER",
)
```

**Tier weights**
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

**Inputs/outputs**
- Used by training and evaluation.
- Saved inside checkpoints.

**Known bugs/issues**
- `CONTEXT_BRIDGE` must be included.
- Dataset should primarily use `causal_type`, not just label name.

**Changes already made**
- Updated for v11.
- Increased max turns/tokens.
- Added pre-split path fields.
- Added oversampling and threshold tuning flags.

**Pipeline connection**
Central configuration for all train/eval code.

---

### `guardlens/data/dataset.py`

**Purpose**  
Dataset and collator definitions.

**Important classes**
- `GuardLensDataset`
- `GuardLensCollator`
- `FlatConversationCollator`

**Inputs**
- JSONL records
- `GuardLensConfig`
- tokenizer

**Outputs**
For GuardLens models:
- `input_ids`: likely `[B, T, S]`
- `attention_mask`
- `turn_mask`
- `role_ids`
- labels
- token labels
- span weights
- metadata

For flat DeBERTa:
- `input_ids`: `[B, L]`
- `attention_mask`: `[B, L]`
- labels
- metadata

**Dependencies**
- HuggingFace tokenizer
- config

**Known bugs/issues**
- ConversationDeBERTa crashes if given turn-shaped tensors. Must use `FlatConversationCollator`.
- Token labels must use `span["causal_type"] == "causal"` primarily.
- Incidental spans must be label 0, not ignored.
- Token F1 can be misleading if ignored/negative tokens are counted incorrectly.

**Changes already made**
- Flat collator used for ConversationDeBERTa evaluation.
- Max turns/tokens updated.

**Pending changes**
- Audit random token F1 metric, likely not dataset but eval metric.
- Ensure metadata includes fields needed for pivot/turn utility:
  - `turns`
  - `pivot_turn_id` or `pivot_turn`
  - `semantic_role`
  - `causal_type`

**Pipeline connection**
Feeds model training/evaluation.

---

### `guardlens/training/trainer.py`

**Purpose**  
Main training loop and model evaluation.

**Important functions**
- `train(config, data_path, output_dir, model_name)`
- `evaluate(model, loader, loss_fn, config, device, threshold=...)`

**Inputs**
- config
- train/dev/test data
- model name
- checkpoints

**Outputs**
- Checkpoints
- test evaluation
- threshold tuning
- logs

**Dependencies**
- model registry
- loss
- dataset/collator
- sklearn/metrics likely
- PyTorch

**Known bugs/issues**
- Must support flat baseline branch for ConversationDeBERTa.
- Earlier evaluation JSON was corrupted by stdout logs.
- `best_attribution.pt` can be from Phase 2 even if Phase 3 ran.

**Changes already made**
- Pre-split loading.
- Phase schedule:
  - P1 cls
  - P2 attr
  - P3 cf
- Baselines skip irrelevant CF phases.
- Weighted loss normalization fixed.
- Pivot labels ignore truncated pivots.

**Pending changes**
- Avoid overclaiming Phase 3 if selected checkpoint is Phase 2.
- Ensure test eval output schema stable for collation.

**Pipeline connection**
Runs all model training.

---

### `guardlens/training/loss.py`

**Purpose**  
Loss definitions.

**Important behavior**
- Classification BCE with class pos_weight.
- Attribution loss with span/tier weights.
- Pivot loss.
- CF consistency loss in Phase 3.

**Inputs**
- model outputs
- batch labels/weights
- config lambdas

**Outputs**
- scalar loss
- loss components

**Known bugs/issues fixed**
- Weighted classification normalization fixed.
- Span-weight clamp removed so zero-weight spans are ignored.
- Pivot-kind mask fixed for ignored pivots.
- No-pivot logit shape fixed.

**Pending changes**
- None urgent.

**Pipeline connection**
Optimizes model.

---

### `guardlens/models/*`

**Purpose**  
Model definitions and registry.

**Important models**
- `GuardLens`
- `GuardLensNoFusion`
- `GuardLensNoCF`
- `TurnLevelClassifier`
- `ConversationDeBERTa`

**Inputs**
- model config
- tokenized tensors

**Outputs**
- classification logits
- attribution outputs
- pivot logits if applicable

**Known bugs/issues**
- Attribution output key inconsistency caused utility script to initially read zero records.
- Utility script should handle keys:
  - `"attribution"`
  - `"attributions"`
  - `"token_attributions"`
  - `"attribution_scores"`
  - `"attr_logits"`

**Changes already made**
- Registry checked and used.
- NoFusion/NoCF ablations trained.

**Pending changes**
- Standardize attribution output key if possible.
- Avoid silent fallback to GuardLens if registry key missing.

**Pipeline connection**
Model implementations for training/eval.

---

### `train_all.slurm`

**Purpose**  
Full experiment training job.

**Inputs**
- train/dev/test JSONL
- model list
- hyperparameters
- output root

**Outputs**
- checkpoints:
  - `$HOME/work/results/guardlens_v11/checkpoints/guardlens/`
  - `guardlens_no_fusion/`
  - `guardlens_no_cf/`
  - `turn_level/`
  - `conversation_deberta/`

**Important successful log**
- `logs/train_all_24224.out`

**Key final table**
```text
guardlens                 F1 0.9943 Acc 0.9964 AttrF1 0.8778 PivAcc 0.7536 Thresh 0.56
guardlens_no_fusion       F1 1.0000 Acc 1.0000 AttrF1 0.8810 PivAcc 0.7536 Thresh 0.20
guardlens_no_cf           F1 0.9943 Acc 0.9964 AttrF1 0.8841 PivAcc 0.7645 Thresh 0.20
turn_level                F1 0.7582 Acc 0.8659 AttrF1 0.0000 PivAcc 0.0000 Thresh 0.55
conversation_deberta      F1 0.8324 Acc 0.8949 AttrF1 0.0000 PivAcc 0.0000 Thresh 0.59
```

**Known issues**
- Phase 3 ran but best attribution checkpoint selected from Phase 2.
- NoFusion/NoCF not worse than GuardLens.

**Pipeline connection**
Produces final model checkpoints.

---

## 7. Core evaluation files

### `guardlens/evaluate.py`

**Purpose**  
Classification evaluation entrypoint.

**Important args**
```bash
--test-path
--data
--checkpoint
--batch-size
--device
--workers
```

**Important behavior**
- Loads checkpoint.
- Gets config and threshold.
- Loads test data via `load_test_data`.
- Uses tokenizer.
- If model is `conversation_deberta`, uses `FlatConversationCollator`.
- Otherwise uses `GuardLensCollator`.
- Calls `trainer.evaluate`.

**Inputs**
- test split JSONL
- checkpoint

**Outputs**
- JSON metrics printed to stdout unless `--output` exists in latest version.
- Metrics:
  - f1
  - accuracy
  - precision
  - recall
  - threshold
  - pivot accuracy
  - family accuracy
  - transfer tier accuracy
  - benign accuracy
  - supervision tier accuracy

**Dependencies**
- `GuardLensConfig`
- `GuardLensDataset`
- `GuardLensCollator`
- `FlatConversationCollator`
- `MODEL_REGISTRY`
- `GuardLensLoss`
- `trainer.evaluate`
- `eval_utils.load_test_data`

**Known bugs/issues**
- Earlier mixed stdout logs with JSON, causing parse errors.
- Should add `--output` to write clean JSON.
- Must branch for flat DeBERTa.
- `trainer.evaluate` must support flat `[B,L]` batches.

**Changes already made**
- ConversationDeBERTa collator fix.
- DeBERTa results recovered.

**Pending changes**
- Ensure JSON output is clean in all SLURM jobs.
- Confirm `--output` argument exists in current code before using.

**Pipeline connection**
Produces classification baseline table.

---

### `guardlens/evaluation/eval_utils.py`

**Purpose**  
Shared evaluation helpers.

**Important functions**
- `load_test_data`
- `add_test_path_args`
- `results_to_latex_table`
- `comparison_to_latex`

**Inputs**
- test path or data path
- records
- metrics dicts

**Outputs**
- loaded records and indices
- LaTeX table strings

**Dependencies**
- JSONL
- split utils

**Known bugs/issues**
- Must not re-split when `--test-path` supplied.
- v11 subset names must be used.

**Changes already made**
- Added test-path support.
- Added LaTeX helpers.
- Added v11 subset partitioning.

**Pending changes**
- Ensure every eval script uses this consistently.

**Pipeline connection**
Shared utility for evaluation suite.

---

### `guardlens/evaluation/causal_eval.py`

**Purpose**  
Core causal attribution engine.

**Important functions**
- `run_causal_evaluation`
- methods for:
  - GuardLens attribution
  - attention
  - integrated gradients
  - grad×input
  - surface-risk
  - random
- deletion/masking evaluator

**Inputs**
- model
- DataLoader
- device
- methods
- top_k_fractions
- tokenizer

**Outputs**
Per method:
- deviation drops
- flip rates
- necessity
- sufficiency
- token F1
- pivot exact/within1
- trigger size

**Dependencies**
- model outputs
- tokenizer
- masking logic
- surface risk scoring

**Known bugs/issues**
- Random token F1 is suspiciously high due to metric definition; do not report it.
- Attention may fall back to representation-change proxy if true attention weights unavailable.
- Token F1 on transformed deconfounded variants invalid if span offsets not remapped.
- Internal self-eval is not the same as external evaluator token-removal.

**Changes already made**
- Added multiple methods.
- Surface-risk baseline included.
- Integrated gradients included.
- Used by deconfounded eval.

**Pending changes**
- Fix/replace token F1 with:
  - positive-class F1 over non-ignored annotated tokens
  - AUPRC
  - span-level F1
- Mark attention fallback in logs.

**Pipeline connection**
Central attribution evaluation.

---

### `guardlens/evaluation/eval_causal.py`

**Purpose**  
CLI wrapper for causal eval and stratified analyses.

**Inputs**
- test path
- checkpoint
- method list
- output path

**Outputs**
- `causal_eval_results.json`
- per-tier results
- per-transfer/pivot results
- LaTeX tables

**Added features**
- Per-supervision-tier eval:
  - cf_strong
  - cf_weak
  - llm_confirmed
  - construction
  - benign_validated
- Per-transfer-tier eval:
  - transfer_success
  - target_only
  - cross_only
- Pivot-kind eval:
  - contextual_pivot
  - lexical_pivot

**Dependencies**
- `causal_eval.py`
- `eval_utils.py`
- model checkpoint

**Known bugs/issues**
- Exact pivot metric low and over-harsh.
- Must report n per tier because CF tiers tiny.
- Surface-risk baseline dominates raw deletion but poor FPR.

**Changes already made**
- v11 split support.
- LaTeX output.
- Tier/pivot/transfer breakdowns.

**Pending changes**
- Ensure paper tables do not overinterpret tiny cf_strong/cf_weak.

**Pipeline connection**
Core attribution results.

---

## 8. Boundary, surface-risk, and utility evaluation files

### `eval_boundary_stress.py`

**Purpose**  
Evaluates model specificity on rejected/boundary benign records.

**Inputs**
- boundary benign file
- detection checkpoint

**Outputs**
- `boundary_stress.json`

**Key results**
- boundary records: 279
- GuardLens FPR: 0.0072
- false positives: 2/279
- accuracy: 0.9928

**Dependencies**
- best_detection checkpoint
- boundary benign JSONL

**Known bugs/issues**
- Must use best_detection checkpoint, not best_attribution checkpoint.
- Should report FPR, not just accuracy.

**Changes already made**
- Added as separate eval after missing earlier.
- Probability stats added/desired:
  - mean P(adv)
  - p95 P(adv)

**Pipeline connection**
Provides GuardLens FPR for utility metric.

---

### `eval_surface_risk_fpr.py`

**Purpose**  
Computes false positive rates of the hardcoded surface-risk baseline on benign/boundary subsets.

**Inputs**
- test split
- boundary benign file
- threshold(s)

**Outputs**
- `surface_risk_fpr.json`

**Key results**
Surface-risk FPR@0.5:
- all_benign: 0.143
- boundary_rejected: 0.373
- boundary_false_lead_benign: 0.576
- false_lead_benign: 0.667
- validated_benign_twin: 0.224

**Dependencies**
- surface_risk_score
- v11 family/benign_status fields

**Known bugs/issues**
- Must use v11 families, not old `hard_negative`, `borderline_benign`.
- Surface-risk threshold choice should be transparent.

**Changes already made**
- Added threshold-based FPR comparison.

**Pipeline connection**
Provides surface-risk FPR for utility and paper specificity table.

---

### `guardlens/evaluation/eval_attribution_utility.py`

**Purpose**  
Post-hoc metrics combining attribution quality with specificity.

Metrics:
1. Attribution Utility:
   - `Utility = DD@15 - λ * FPR`
2. Causal Turn Mass:
   - attribution mass on causal turn roles / total mass
3. Pivot Window Accuracy:
   - prediction within ±W user turns

**Important functions**
- `load_jsonl`
- `compute_attribution_utility`
- `compute_utility_table`
- `compute_causal_turn_mass`
- `compute_pivot_window_accuracy`

**Inputs**
```bash
--test-path
--causal-results
--boundary-results
--surface-fpr-results
--checkpoint
--output
--device
```

**Outputs**
- `attribution_utility.json`

**Dependencies**
- `causal_eval_results.json`
- `boundary_stress.json`
- `surface_risk_fpr.json`
- checkpoint
- dataset/collator/model registry

**Known bugs/issues fixed**
- Initially returned 0 records because it looked for wrong attribution output key.
- Fixed by handling correct attribution output key.
- Initially double-counted causal turn mass, fixed by single `is_causal_turn` check.
- Pivot user-turn vs raw-turn index mapping fixed.
- Boundary-specific utility added.

**Known caveats**
- FPR for neural attribution methods is set to 0 in all-benign utility table, which is not directly comparable. Main headline should compare only GuardLens vs surface-risk.
- Causal turn mass is moderate, not dominant.

**Latest results**
- GuardLens boundary FPR: 0.0072
- Surface-risk boundary FPR: 0.3728
- Headline utility:
  - GuardLens: DD 0.511, FPR 0.007, Utility 0.504
  - Surface-risk: DD 0.568, FPR 0.373, Utility 0.195
- Causal turn mass:
  - mean 0.310
  - median 0.307
- Pivot window:
  - exact 0.092
  - ±1 0.241
  - ±2 0.379
  - ±3 0.483
  - ±5 0.678

**Pending changes**
- Add multiplicative utility optional:
  - `DD@15 * (1 - FPR)`
- Add bootstrap CIs if time.

**Pipeline connection**
Main paper utility/specificity result.

---

### `guardlens/evaluation/eval_deconfounded.py`

**Purpose**  
Creates deconfounded test variants to test surface-risk artifacts.

**Variants**
A. SR-neutralized adversarial:
- Replace surface-risk phrases with semantically equivalent low-risk alternatives.

B. Noise-equalized adversarial:
- Add typos/noise to high-SR adversarial turns.

C. SR-injected benign:
- Inject high-surface-risk phrases into benign records in safe contexts.

D. Combined:
- SR-neutralization + noise equalization.

Also evaluates changed-only neutralized records.

**Important functions**
- `surface_risk_score`
- `add_typos`
- `neutralize_surface_risk`
- `equalize_noise`
- `inject_surface_risk`
- `combined_deconfound`
- `evaluate_variant`
- `print_variant_comparison`

**Inputs**
```bash
--test-path
--checkpoint
--output-dir
--methods guardlens surface_risk random
--top-k 0.05 0.10 0.15 0.20
--device
```

**Outputs**
- `deconfounded_results.json`
- `sr_neutralized.jsonl`
- printed comparison

**Dependencies**
- `GuardLensDataset`
- `GuardLensCollator`
- `run_causal_evaluation`
- model checkpoint
- surface risk phrase dictionary

**Known bugs/issues**
- Token F1 on transformed variants is invalid unless span offsets remapped. Use DD/flip/FPR only.
- Neutralization did not break surface-risk much, so do not claim exact lexical phrase leakage is the only reason surface-risk works.
- Replacement phrases may still preserve adversarial semantics and structural cues, which is actually useful.

**Changes already made**
- Changed-only neutralized slice added.
- Sanity check added: neutralized records still detected adversarial.
- SR-injected benign FPR computed directly from records, not batch metadata.
- Combined variant added.
- Threshold printed from checkpoint.

**Latest results**
- Original adversarial SR:
  - mean 0.750
  - max 1.000
  - >0.3: 80/87
- SR-neutralized:
  - 81/87 changed
  - mean SR 0.000
  - >0.3: 0/87
  - 87/87 still detected adversarial
- DD@15:
  - Original: GuardLens 0.473, Surface-risk 0.543
  - SR-neutralized: GuardLens 0.449, Surface-risk 0.534
  - Changed-only: GuardLens 0.415, Surface-risk 0.504
  - Noise-equalized: GuardLens 0.504, Surface-risk 0.502
  - Combined: GuardLens 0.484, Surface-risk 0.491
- SR-injected benign FPR:
  - GuardLens 0.005 = 1/189
  - Surface-risk 0.947 = 179/189

**Pending changes**
- Consider adding bootstrapped confidence intervals.
- Do not report token F1 for deconfounded variants.
- Use table in paper as artifact robustness/specificity.

**Pipeline connection**
Directly addresses surface-risk leakage/reviewer criticism.

---

### `eval_deconfound.slurm`

**Purpose**  
Runs attribution utility and deconfounded eval.

**Inputs**
```bash
RESULTS_DIR="$HOME/work/results/guardlens_v11"
CKPT="$RESULTS_DIR/checkpoints/guardlens/best_attribution.pt"
TEST_DATA="$HOME/staging/dataset_gen_output/splits/test.jsonl"
OUT_DIR="$RESULTS_DIR/results"
```

**Commands**
1. Runs `eval_attribution_utility.py`
2. Runs `eval_deconfounded.py`
3. Prints summary

**Dependencies**
Prerequisite JSONs:
- `$OUT_DIR/causal_eval_results.json`
- `$OUT_DIR/boundary_stress.json`
- `$OUT_DIR/surface_risk_fpr.json`

**Known bugs/issues**
- Uses `set -uo pipefail`; for final paper run, prefer `set -euo pipefail`.
- Current `cmd && echo SUCCESS || echo FAILED` lets job continue after failure. Good for exploration, bad for final run.

**Changes already made**
- Prerequisite checks added.
- Summary prints utility, causal turn mass, pivot window, deconfounded comparison, SR-injected benign FPR.

**Latest log**
- `eval_deconfound_24495.out`

**Pipeline connection**
Produces the newest/highest-value evaluation results.

---

## 9. Other evaluation files

### `eval_implicit_explicit.py`

**Purpose**  
Compares contextual vs lexical pivot cases.

**Inputs**
- test split
- checkpoint

**Outputs**
- contextual/lexical subset metrics
- LaTeX tables

**Dependencies**
- v11 pivot_kind fields:
  - `contextual_pivot`
  - `lexical_pivot`

**Known bugs/issues**
- Old names `implicit`/`explicit` were stale.
- Must not use old `implicit_trigger` field as primary criterion.

**Changes already made**
- Updated to v11 pivot names.
- LaTeX output added.

**Pipeline connection**
Supports multi-turn contextual claim.

---

### `eval_paraphrase.py`

**Purpose**  
Tests attribution stability under paraphrase.

**Inputs**
- test split
- checkpoint
- subset:
  - full
  - contextual_pivot
  - lexical_pivot

**Outputs**
- turn correlation
- token correlation
- top-k stability
- pivot stability

**Known bugs/issues**
- Old Stage 4 passed `--subset implicit` / `explicit`, which was invalid.
- Fixed to contextual/lexical.
- Lexical branch previously also filtered on implicit triggers incorrectly; should not.

**Results**
- Contextual:
  - turn ρ 0.983
  - token ρ 0.958
  - top-15 stability 0.692
  - pivot stability 0.500
- Lexical:
  - turn ρ 0.982
  - token ρ 0.957
  - top-15 stability 0.670
  - pivot stability 0.444

**Pipeline connection**
Semantic robustness section.

---

### `eval_attribution_precision.py`

**Purpose**  
Attribution precision and minimality curves, including hard benign precision.

**Inputs**
- test split
- checkpoint

**Outputs**
- top-k precision
- minimality curve

**Known bugs/issues**
- Old code used old families:
  - hard_negative
  - borderline_benign
  - false_positive_trap
- Must use v11:
  - clean_everyday
  - research_technical
  - topic_matched_safe
  - hard_benign
  - false_lead_benign
  - interactive_benign_twin
  - benign_status

**Changes already made**
- Updated subset logic to v11.

**Pending changes**
- Ensure span-level metrics are actually run and stored.
- Random token F1 issue should not affect reported precision.

**Pipeline connection**
Supports attribution label quality.

---

### `eval_external.py`

**Purpose**  
External evaluator/token removal test.

**Inputs**
- test split
- model checkpoint
- external model/evaluator
- subset

**Outputs**
- external DD/flip metrics

**Known bugs/issues**
- Parser choices earlier stale:
  - `implicit`, `explicit`, `clean_holdout`
- Should use:
  - `contextual_pivot`
  - `lexical_pivot`
  - `transfer_success`
- Must distinguish internal self-eval from external evaluator.

**Pipeline connection**
Secondary external validation.

---

### `eval_cross_dataset.py`

**Purpose**  
Cross-dataset generalization using AdvBench/HarmBench/JailbreakBench seeds extended into multi-turn conversations.

**Inputs**
- single-turn adversarial seed datasets
- benign pool/test split
- checkpoint

**Outputs**
- cross-dataset detection results

**Results**
- GuardLens F1 0.607, precision 0.992, recall 0.437
- NoCF F1 0.556
- ConversationDeBERTa F1 0.082

**Known bugs/issues**
- Do not use this for MHJ because MHJ is already multi-turn.
- Benign sampling should come from clean benign pool or test split label 0.

**Pipeline connection**
External-ish generalization to AdvBench/HarmBench.

---

### `eval_cross_model_transfer.py`

**Purpose**  
Tests whether token deletion guided by GuardLens transfers to external safety evaluator such as ShieldGemma or LlamaGuard.

**Inputs**
- data/final dataset or test split
- checkpoint
- external model
- subset
- threshold
- variants cache

**Outputs**
- ShieldGemma/LlamaGuard transfer flip rates

**Known bugs/issues fixed**
- Parser once used `--external-model` but code referenced `args.llamaguard_model`.
- Stage 6 used `--llamaguard-model` while parser expected `--external-model`.
- Variants cache was reused across subsets, invalidating lexical/contextual runs.
- Threshold saved as hardcoded 0.5; should save `args.threshold`.

**Current results**
ShieldGemma:
- transfer_success: GuardLens 0.909 vs surface-risk 0.636, usable n 11
- lexical: GuardLens 0.789 vs surface-risk 0.737, usable n 38
- contextual: GuardLens 0.789 vs surface-risk 0.737, usable n 19

**Known caveat**
- ShieldGemma denominator low.
- Random sometimes ties.
- Use as secondary/appendix.

**Pending changes**
- If using LlamaGuard, implement separate backend; do not reuse ShieldGemma yes/no logits.
- Keep cached variants but evaluator-specific scoring.

**Pipeline connection**
Secondary external evaluator transfer.

---

### `eval_mhj.slurm`

**Purpose**  
Runs MHJ external evaluation.

**Inputs**
- `mhj_external_test.jsonl`
- best_detection checkpoint
- best_attribution checkpoint

**Outputs**
- MHJ classification results
- MHJ causal attribution results

**Known bugs/issues**
- If MHJ malicious-only, F1/accuracy misleading.
- Report recall and detected-case attribution.

**Current results**
- MHJ detection F1 0.0614
- recall 0.0317
- detected-case GuardLens DD@15 0.4949 vs surface-risk 0.2038

**Pipeline connection**
External multi-turn stress test.

---

### `eval_full.slurm`

**Purpose**  
Master/core evaluation script.

**Inputs**
- test split
- boundary file
- checkpoints

**Uses**
- `best_attribution.pt` for causal attribution
- `best_detection.pt` for classification
- boundary stress
- surface-risk FPR
- collation

**Known bugs/issues**
- Should use clean JSON output from `evaluate.py`.
- Must fail if main checkpoint missing.
- Walltime must cover IG if included.

**Changes already made**
- Fixed to use best_attribution for causal and best_detection for classification.
- Boundary added separately earlier because missed.

**Pipeline connection**
Core paper results.

---

## 10. Human annotation files

### `guardlens_human_annotation_guide_updated.md`

**Purpose**  
Human annotation guide for blind audit.

**Annotation task**
Annotators fill:
- `classification`: `ADVERSARIAL`, `BENIGN`, `UNCERTAIN`
- `pivot_turn`: 0-indexed user turn, `DISTRIBUTED`, or blank
- `causal_spans`: concise user spans
- `confidence`: `HIGH`, `MEDIUM`, `LOW`
- `notes`

**Inputs**
- Spreadsheet of 100 conversations

**Outputs**
- Completed spreadsheet columns D-H
- double annotation agreement

**Important definitions**
- Sensitive words alone do not make a conversation adversarial.
- Only annotate user turns.
- Use `DISTRIBUTED` when no single pivot captures causality.
- Prefer concise spans.

**Agreement metrics**
- classification Cohen’s κ, target ≥ 0.70
- pivot exact, within-one, distributed agreement
- span token/character overlap F1

**Known bugs/issues**
- Must ensure human benchmark excluded from training/model selection.
- Need clear spreadsheet columns.

**Pipeline connection**
Mitigates synthetic-label reviewer risk.

---

## 11. Paper/writing context files already created

### `GuardLens_Project_Context_Pack.md`

**Purpose**  
Earlier portable project context pack.

**Status**
Superseded by updated project context pack after deconfounded results.

---

### `GuardLens_Codebase_Context_Pack.md`

**Purpose**  
Earlier codebase handoff.

**Status**
Superseded by this updated codebase context pack.

---

### `GuardLens_Paper_Writing_Context_Pack.md`

**Purpose**  
Paper writing / venue / framing handoff.

**Status**
Still useful, but should be updated with latest deconfounded results:
- utility metric
- SR-injected benign FPR
- pivot-window results

---

### `GuardLens_Project_Context_Pack_Updated.md`

**Purpose**  
Latest project context pack after deconfounded results.

**Status**
Use as high-level project handoff.

---

## 12. Logs and result files

### `logs/train_all_24224.out`

**Purpose**
Full training log for main suite.

**Key content**
- Shows Phase 3 did run.
- Shows best attribution from Phase 2.
- Shows final model summary table.
- Shows ConversationDeBERTa success.

**Known interpretation**
- Phase 3 not absent; it ran but did not improve selected checkpoint.

---

### `logs/eval_deconfound_24495.out`

**Purpose**
Latest successful deconfounded and utility result log.

**Key content**
- Utility:
  - GuardLens 0.504
  - Surface-risk boundary utility 0.195
- Causal turn mass:
  - mean 0.310
- Pivot window:
  - ±5 = 0.678
- SR-injected benign FPR:
  - GuardLens 0.005
  - Surface-risk 0.947
- Deconfounded DD table.

**Pipeline connection**
Use for paper main claims.

---

### `causal_eval_results.json`

**Purpose**
Core causal attribution results.

**Required by**
- `eval_attribution_utility.py`
- summary tables

---

### `boundary_stress.json`

**Purpose**
Boundary benign FPR for GuardLens.

**Required by**
- utility script

---

### `surface_risk_fpr.json`

**Purpose**
Surface-risk FPR across benign/boundary subsets.

**Required by**
- utility script

---

### `attribution_utility.json`

**Purpose**
Output of `eval_attribution_utility.py`.

**Contains**
- utility_all_benign
- utility_boundary
- causal_turn_mass
- pivot_window

---

### `deconfounded_results.json`

**Purpose**
Output of `eval_deconfounded.py`.

**Contains**
- original
- sr_neutralized
- sr_neutralized_changed
- noise_equalized
- combined
- sr_injected_benign

---

## 13. Known cross-file bugs / gotchas

1. **Attribution output key mismatch**
   - Utility script initially expected `"attribution"`.
   - Model may output different key.
   - Add helper to search possible keys.

2. **ConversationDeBERTa collator**
   - Must use `FlatConversationCollator`.
   - Otherwise DeBERTa gets `[B,T,S]` and crashes.

3. **JSON corruption**
   - If evaluation prints logs and JSON to stdout, redirected JSON is invalid.
   - Use `--output` or separate log/stderr.

4. **Surface-risk cache**
   - Cross-model transfer variants cache must be unique per subset.

5. **Threshold hardcoding**
   - Cross-model transfer must save actual `args.threshold`.

6. **Pivot turn indexing**
   - Human/dataset pivot is 0-indexed user turn.
   - Raw turns include assistant.
   - Utility script must map user-turn index vs raw index.

7. **Causal turn mass double-counting**
   - Fixed by single `is_causal_turn` check.

8. **Random token F1**
   - Inflated and unreliable.
   - Do not report.

9. **Deconfounded span offsets**
   - After text replacement/noise, original span offsets may be wrong.
   - Do not report token F1 on transformed variants.

10. **MHJ should not be re-split into train/test**
   - Keep external-only.

11. **Old v10 references**
   - Many old docs/scripts referenced `semantic_multiturn_v10_augmented.jsonl`.
   - Must update to v11 split paths.

12. **Old family names**
   - Replace old:
     - `hard_negative`
     - `borderline_benign`
     - `false_positive_trap`
   - With v11:
     - `hard_benign`
     - `false_lead_benign`
     - `research_technical`
     - `topic_matched_safe`
     - `interactive_benign_twin`

---

## 14. Pending code changes

Highest priority:

1. Clean `evaluate.py` JSON output with `--output`.
2. Fix/replace token F1 metric, especially random.
3. Add span-level metrics:
   - span precision
   - span recall
   - span F1
   - incidental FPR
   - token AUPRC over non-ignored tokens
4. Add bootstrap confidence intervals for:
   - utility
   - DD@15
   - boundary FPR
   - SR-injected benign FPR
5. Ensure human benchmark CSV creation and adjudication scripts work.
6. Add paper table collation script that pulls:
   - classification
   - core causal
   - utility
   - deconfounded
   - boundary
   - human audit
7. Ensure all SLURM scripts fail fast for final runs.

Nice to have:

8. LlamaGuard external transfer backend separate from ShieldGemma.
9. Multiplicative utility option:
   - `DD@15 * (1 - FPR)`
10. Error taxonomy extraction.

---

## 15. How files connect in the final pipeline

```text
build_semantic_datasetv11.py
  -> semantic_multiturn_v11_interactive_raw.jsonl
  -> launch_val.slurm
  -> validated JSONLs
  -> merge_validations.py
  -> combined_all.jsonl
  -> launch_causal.slurm
  -> causal_analyzed_all_final.jsonl
  -> postprocess_causal.py
  -> malicious_final.jsonl

benign_pool_v11_llama_validated.jsonl
benign_pool_v11_mistral_validated.jsonl
  -> benign_clean.jsonl
  -> benign_boundary.jsonl

malicious_final.jsonl + benign_clean.jsonl
  -> final_dataset.jsonl
  -> split_dataset.py
  -> splits/train.jsonl
  -> splits/dev.jsonl
  -> splits/test.jsonl

splits/*
  -> train.py / train_all.slurm
  -> checkpoints/*

checkpoints + splits/test.jsonl + benign_boundary.jsonl
  -> eval_full.slurm
  -> causal_eval_results.json
  -> boundary_stress.json
  -> surface_risk_fpr.json
  -> classification JSONs

causal_eval_results.json + boundary_stress.json + surface_risk_fpr.json
  -> eval_attribution_utility.py
  -> attribution_utility.json

splits/test.jsonl + best_attribution.pt
  -> eval_deconfounded.py
  -> deconfounded_results.json

MHJ raw
  -> mhj_loader.py
  -> mhj_external_test.jsonl
  -> eval_mhj.slurm

human benchmark split
  -> annotation spreadsheet
  -> human audit metrics
```

---

## 16. Final note for next assistant

The codebase is now evaluation-heavy. Do not recommend more generic data generation. The highest-value work is:

1. human audit benchmark,
2. clean paper tables,
3. fixing metric reporting bugs,
4. emphasizing attribution utility and deconfounded specificity,
5. careful EMNLP framing.

Main empirical story is not that GuardLens dominates every baseline. It is:

> Surface-risk is a strong lexical deletion heuristic but brittle and non-specific. GuardLens is a learned, tier-supervised multi-turn attribution model with much better causal-specificity tradeoff, strong boundary robustness, and strong label alignment.
