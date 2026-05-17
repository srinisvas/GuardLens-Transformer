"""Dataset and collation for GuardLens — v11 dataset compatible."""

from typing import Dict, List

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from guardlens.config import GuardLensConfig


# =========================================================
# Pivot kind mapping
# =========================================================

PIVOT_KIND_MAP = {
    "lexical_pivot": 0,
    "contextual_pivot": 1,
    "distributed": 2,
    "misleading_decoy": 3,
    "none": 4,
}


class GuardLensDataset(Dataset):
    """
    Loads JSONL records and prepares them for the hierarchical model.

    v11 changes:
      - Uses causal_type field (not just label name) for attribution
      - Incidental spans are explicit negatives (0), not ignored (-1)
      - Stores span_tier_weight per token for weighted attribution loss
      - Stores sample loss_weight for weighted classification loss
      - Stores pivot_turn_id and pivot_kind for pivot head
      - Benign records: only annotated DECOY/incidental spans get labels,
        unannotated tokens stay -1 (not forced to 0)
    """

    def __init__(self, records: List[Dict], config: GuardLensConfig):
        self.records = records
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        turns = record["turns"][:self.config.max_turns]

        turn_texts = []
        turn_roles = []
        char_labels_per_turn = []
        char_tier_weights_per_turn = []

        is_malicious = record.get("label", 0) == 1

        for turn in turns:
            text = turn["text"][:500]
            role = 0 if turn["role"] == "user" else 1
            turn_texts.append(text)
            turn_roles.append(role)

            # Initialize: -1 = unannotated (ignored in loss)
            char_labels = [-1] * len(text)
            char_weights = [0.0] * len(text)

            for span in turn.get("span_annotations", []):
                cs = span.get("char_start", 0)
                ce = span.get("char_end", 0)
                causal_type = span.get("causal_type", "unvalidated")
                label_name = span.get("label", "")
                span_tier = span.get("supervision_tier", "construction")

                # Get tier weight for this span
                tier_weight = self.config.span_tier_weights.get(span_tier, 0.40)

                # Determine token label from causal_type (primary) and label name (fallback)
                if causal_type == "causal":
                    token_label = 1
                elif causal_type == "incidental":
                    token_label = 0  # Explicit negative — not -1
                elif label_name in self.config.causal_span_labels:
                    token_label = 1
                elif label_name in self.config.incidental_span_labels:
                    token_label = 0
                else:
                    continue  # Skip unclassifiable spans

                for i in range(cs, min(ce, len(text))):
                    # For conflicts: causal (1) takes priority over incidental (0)
                    if token_label == 1 or char_labels[i] == -1:
                        char_labels[i] = token_label
                        char_weights[i] = tier_weight

            # For benign records without annotations:
            # Don't force all tokens to negative. Only annotated
            # DECOY/incidental spans get explicit 0.
            # Unannotated benign tokens stay -1 (ignored).

            char_labels_per_turn.append(char_labels)
            char_tier_weights_per_turn.append(char_weights)

        # Pivot info
        pivot_turn_id = record.get("pivot_turn_id")
        pivot_kind = record.get("pivot_kind", "none")
        pivot_kind_id = PIVOT_KIND_MAP.get(pivot_kind, 4)

        return {
            "turn_texts": turn_texts,
            "turn_roles": turn_roles,
            "label": record["label"],
            "loss_weight": record.get("loss_weight", 0.5),
            "char_labels": char_labels_per_turn,
            "char_tier_weights": char_tier_weights_per_turn,
            "conversation_id": record.get("conversation_id", ""),
            "difficulty": record.get("difficulty", "medium"),
            "family": record.get("family", "unknown"),
            "pivot_turn_id": pivot_turn_id,
            "pivot_kind": pivot_kind,
            "pivot_kind_id": pivot_kind_id,
            "supervision_tier": record.get("supervision_tier", "construction"),
            "transfer_tier": record.get("transfer_tier", "unknown"),
            "benign_status": record.get("benign_status", "none"),
        }


class GuardLensCollator:
    """
    Tokenizes turns, aligns character-level labels to token-level,
    pads everything into tensors.

    v11 changes:
      - Produces span_weights: [B, T, S] per-token tier weights
      - Produces sample_weights: [B] per-sample loss weights
      - Produces pivot_labels: [B] pivot turn index (0-based, T for no-pivot)
      - Produces pivot_kind_labels: [B] pivot kind class
    """

    def __init__(self, tokenizer, config: GuardLensConfig):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_turns = min(
            max(len(item["turn_texts"]) for item in batch),
            self.config.max_turns,
        )

        all_input_ids = []
        all_attention_masks = []
        all_turn_masks = []
        all_role_ids = []
        all_token_labels = []
        all_span_weights = []
        all_labels = []
        all_sample_weights = []
        all_pivot_labels = []
        all_pivot_kind_labels = []
        metadata = []

        for item in batch:
            turn_input_ids = []
            turn_attention_masks = []
            turn_mask = []
            turn_role_ids = []
            turn_token_labels = []
            turn_span_weights = []

            for t_idx in range(max_turns):
                if t_idx < len(item["turn_texts"]):
                    text = item["turn_texts"][t_idx]
                    role = item["turn_roles"][t_idx]
                    char_labels = item["char_labels"][t_idx]
                    char_weights = item["char_tier_weights"][t_idx]

                    enc = self.tokenizer(
                        text,
                        max_length=self.config.max_tokens_per_turn,
                        padding="max_length",
                        truncation=True,
                        return_offsets_mapping=True,
                        return_tensors="pt",
                    )

                    input_ids = enc["input_ids"].squeeze(0)
                    attn_mask = enc["attention_mask"].squeeze(0)
                    offsets = enc["offset_mapping"].squeeze(0)

                    tok_labels = torch.full(
                        (self.config.max_tokens_per_turn,), -1, dtype=torch.long,
                    )
                    tok_weights = torch.zeros(
                        self.config.max_tokens_per_turn, dtype=torch.float,
                    )

                    for tok_idx, (start, end) in enumerate(offsets):
                        if end <= start or attn_mask[tok_idx] == 0:
                            tok_labels[tok_idx] = -1
                            continue

                        span_labels = char_labels[start:end]
                        span_wts = char_weights[start:end]

                        if not span_labels:
                            tok_labels[tok_idx] = -1
                            continue

                        # Token label: max of character labels
                        # If any char is 1 (causal), token is 1
                        # If any char is 0 (incidental), token is 0
                        # If all chars are -1, token is -1
                        max_label = max(span_labels)
                        if max_label >= 0:
                            tok_labels[tok_idx] = max_label
                            # Weight: max of character weights
                            tok_weights[tok_idx] = max(span_wts) if span_wts else 0.40
                        else:
                            tok_labels[tok_idx] = -1

                    turn_input_ids.append(input_ids)
                    turn_attention_masks.append(attn_mask)
                    turn_mask.append(1)
                    turn_role_ids.append(role)
                    turn_token_labels.append(tok_labels)
                    turn_span_weights.append(tok_weights)
                else:
                    s = self.config.max_tokens_per_turn
                    turn_input_ids.append(torch.zeros(s, dtype=torch.long))
                    turn_attention_masks.append(torch.zeros(s, dtype=torch.long))
                    turn_mask.append(0)
                    turn_role_ids.append(0)
                    turn_token_labels.append(torch.full((s,), -1, dtype=torch.long))
                    turn_span_weights.append(torch.zeros(s, dtype=torch.float))

            all_input_ids.append(torch.stack(turn_input_ids))
            all_attention_masks.append(torch.stack(turn_attention_masks))
            all_turn_masks.append(torch.tensor(turn_mask, dtype=torch.long))
            all_role_ids.append(torch.tensor(turn_role_ids, dtype=torch.long))
            all_token_labels.append(torch.stack(turn_token_labels))
            all_span_weights.append(torch.stack(turn_span_weights))
            all_labels.append(item["label"])
            all_sample_weights.append(item["loss_weight"])

            # Pivot label: index into turns (max_turns = no pivot)
            pivot_id = item["pivot_turn_id"]
            if pivot_id is None:
                pivot_idx = max_turns  # True no-pivot class
            elif pivot_id < max_turns:
                pivot_idx = pivot_id   # Pivot within retained turns
            else:
                pivot_idx = -1         # Truncated pivot — ignore in loss

            all_pivot_labels.append(pivot_idx)
            all_pivot_kind_labels.append(item["pivot_kind_id"])

            metadata.append({
                "conversation_id": item["conversation_id"],
                "difficulty": item["difficulty"],
                "family": item["family"],
                "pivot_turn_id": item["pivot_turn_id"],
                "pivot_kind": item.get("pivot_kind", "none"),
                "supervision_tier": item["supervision_tier"],
                "transfer_tier": item["transfer_tier"],
                "benign_status": item["benign_status"],
            })

        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_masks),
            "turn_mask": torch.stack(all_turn_masks),
            "role_ids": torch.stack(all_role_ids),
            "token_labels": torch.stack(all_token_labels),
            "span_weights": torch.stack(all_span_weights),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "sample_weights": torch.tensor(all_sample_weights, dtype=torch.float),
            "pivot_labels": torch.tensor(all_pivot_labels, dtype=torch.long),
            "pivot_kind_labels": torch.tensor(all_pivot_kind_labels, dtype=torch.long),
            "metadata": metadata,
        }


class FlatConversationCollator:
    """
    Collator for ConversationDeBERTa baseline.
    v11: increased max_total_tokens to 2048.
    """

    def __init__(self, tokenizer, config: GuardLensConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.sep = tokenizer.sep_token or "[SEP]"

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_attention_masks = []
        all_token_labels = []
        all_labels = []
        all_sample_weights = []
        metadata = []

        for item in batch:
            full_text = f" {self.sep} ".join(item["turn_texts"])

            enc = self.tokenizer(
                full_text,
                max_length=self.config.max_total_tokens,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attn_mask = enc["attention_mask"].squeeze(0)
            offsets = enc["offset_mapping"].squeeze(0)

            char_labels = []
            for cl in item["char_labels"]:
                char_labels.extend(cl)
                char_labels.extend([-1] * len(f" {self.sep} "))
            char_labels = char_labels[:len(full_text)]

            tok_labels = torch.full(
                (self.config.max_total_tokens,), -1, dtype=torch.long,
            )
            for tok_idx, (start, end) in enumerate(offsets):
                if end <= start or attn_mask[tok_idx] == 0:
                    continue
                if start < len(char_labels):
                    span = char_labels[start:min(end, len(char_labels))]
                    valid = [s for s in span if s >= 0]
                    tok_labels[tok_idx] = max(valid) if valid else -1
                else:
                    tok_labels[tok_idx] = -1

            all_input_ids.append(input_ids)
            all_attention_masks.append(attn_mask)
            all_token_labels.append(tok_labels)
            all_labels.append(item["label"])
            all_sample_weights.append(item["loss_weight"])
            metadata.append({
                "conversation_id": item["conversation_id"],
                "difficulty": item["difficulty"],
                "family": item["family"],
                "pivot_turn_id": item["pivot_turn_id"],
                "supervision_tier": item.get("supervision_tier", "unknown"),
                "transfer_tier": item.get("transfer_tier", "unknown"),
                "benign_status": item.get("benign_status", "none"),
            })

        B = len(all_input_ids)
        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_masks),
            "turn_mask": torch.ones(B, 1, dtype=torch.long),
            "role_ids": torch.zeros(B, 1, dtype=torch.long),
            "token_labels": torch.stack(all_token_labels),
            "span_weights": torch.ones(B, self.config.max_total_tokens, dtype=torch.float) * 0.4,
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "sample_weights": torch.tensor(all_sample_weights, dtype=torch.float),
            "pivot_labels": torch.full((B,), 0, dtype=torch.long),
            "pivot_kind_labels": torch.full((B,), 4, dtype=torch.long),  # "none"
            "metadata": metadata,
        }


# =========================================================
# Sampler: oversample CF-labeled records
# =========================================================

def build_weighted_sampler(records: List[Dict], config: GuardLensConfig) -> WeightedRandomSampler:
    """
    Build a sampler that oversamples cf_strong/cf_weak records
    so they appear more frequently during attribution training.
    """
    weights = []
    for r in records:
        tier = r.get("supervision_tier", "construction")
        if tier in ("cf_strong", "cf_weak"):
            weights.append(float(config.cf_oversample_factor))
        elif tier == "llm_confirmed":
            weights.append(1.5)
        else:
            weights.append(1.0)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(records),
        replacement=True,
    )