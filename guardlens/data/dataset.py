"""Dataset and collation for GuardLens — v11 dataset compatible.

NAACL repair addition: ``pivot_supervision_ignore`` distinguishes an unknown
malicious pivot from a true no-pivot benign example. This prevents absence of
counterfactual evidence from being trained as evidence for the no-pivot class.
"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from guardlens.config import GuardLensConfig

PIVOT_KIND_MAP = {
    "lexical_pivot": 0,
    "contextual_pivot": 1,
    "distributed": 2,
    "misleading_decoy": 3,
    "none": 4,
}


class GuardLensDataset(Dataset):
    def __init__(self, records: List[Dict], config: GuardLensConfig):
        self.records = records
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        turns = record["turns"][:self.config.max_turns]
        turn_texts, turn_roles = [], []
        char_labels_per_turn, char_tier_weights_per_turn = [], []

        for turn in turns:
            text = turn["text"][:500]
            role = 0 if turn["role"] == "user" else 1
            turn_texts.append(text)
            turn_roles.append(role)
            char_labels = [-1] * len(text)
            char_weights = [0.0] * len(text)

            for span in turn.get("span_annotations", []):
                cs = span.get("char_start", 0)
                ce = span.get("char_end", 0)
                causal_type = span.get("causal_type", "unvalidated")
                label_name = span.get("label", "")
                span_tier = span.get("supervision_tier", "construction")
                tier_weight = self.config.span_tier_weights.get(span_tier, 0.40)

                if causal_type == "causal":
                    token_label = 1
                elif causal_type == "incidental":
                    token_label = 0
                elif label_name in self.config.causal_span_labels:
                    token_label = 1
                elif label_name in self.config.incidental_span_labels:
                    token_label = 0
                else:
                    continue

                if not isinstance(cs, int) or not isinstance(ce, int):
                    continue
                if cs < 0 or ce <= cs:
                    continue
                for i in range(cs, min(ce, len(text))):
                    if token_label == 1 or char_labels[i] == -1:
                        char_labels[i] = token_label
                        char_weights[i] = tier_weight

            char_labels_per_turn.append(char_labels)
            char_tier_weights_per_turn.append(char_weights)

        pivot_turn_id = record.get("pivot_turn_id")
        pivot_kind = record.get("pivot_kind", "none")
        pivot_kind_id = PIVOT_KIND_MAP.get(pivot_kind, 4)
        pivot_supervision_ignore = bool(record.get("pivot_supervision_ignore", False))

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
            "pivot_supervision_ignore": pivot_supervision_ignore,
            "supervision_tier": record.get("supervision_tier", "construction"),
            "transfer_tier": record.get("transfer_tier", "unknown"),
            "benign_status": record.get("benign_status", "none"),
        }


class GuardLensCollator:
    def __init__(self, tokenizer, config: GuardLensConfig):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_turns = min(max(len(item["turn_texts"]) for item in batch), self.config.max_turns)
        all_input_ids, all_attention_masks = [], []
        all_turn_masks, all_role_ids = [], []
        all_token_labels, all_span_weights = [], []
        all_labels, all_sample_weights = [], []
        all_pivot_labels, all_pivot_kind_labels = [], []
        metadata = []

        for item in batch:
            turn_input_ids, turn_attention_masks = [], []
            turn_mask, turn_role_ids = [], []
            turn_token_labels, turn_span_weights = [], []

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
                    tok_labels = torch.full((self.config.max_tokens_per_turn,), -1, dtype=torch.long)
                    tok_weights = torch.zeros(self.config.max_tokens_per_turn, dtype=torch.float)

                    for tok_idx, (start, end) in enumerate(offsets):
                        start_i, end_i = int(start), int(end)
                        if end_i <= start_i or attn_mask[tok_idx] == 0:
                            continue
                        span_labels = char_labels[start_i:end_i]
                        span_wts = char_weights[start_i:end_i]
                        if not span_labels:
                            continue
                        max_label = max(span_labels)
                        if max_label >= 0:
                            tok_labels[tok_idx] = max_label
                            tok_weights[tok_idx] = max(span_wts) if span_wts else 0.40

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

            pivot_id = item["pivot_turn_id"]
            if item.get("pivot_supervision_ignore", False):
                pivot_idx = -1
            elif pivot_id is None:
                pivot_idx = max_turns
            elif isinstance(pivot_id, int) and 0 <= pivot_id < max_turns:
                pivot_idx = pivot_id
            else:
                pivot_idx = -1

            all_pivot_labels.append(pivot_idx)
            all_pivot_kind_labels.append(item["pivot_kind_id"])
            metadata.append({
                "conversation_id": item["conversation_id"],
                "difficulty": item["difficulty"],
                "family": item["family"],
                "pivot_turn_id": item["pivot_turn_id"],
                "pivot_kind": item.get("pivot_kind", "none"),
                "pivot_supervision_ignore": item.get("pivot_supervision_ignore", False),
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
    def __init__(self, tokenizer, config: GuardLensConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.sep = tokenizer.sep_token or "[SEP]"

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        all_input_ids, all_attention_masks = [], []
        all_token_labels, all_labels, all_sample_weights = [], [], []
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
            sep_pad = [-1] * len(f" {self.sep} ")
            for cl in item["char_labels"]:
                char_labels.extend(cl)
                char_labels.extend(sep_pad)
            char_labels = char_labels[:len(full_text)]
            tok_labels = torch.full((self.config.max_total_tokens,), -1, dtype=torch.long)
            for tok_idx, (start, end) in enumerate(offsets):
                start_i, end_i = int(start), int(end)
                if end_i <= start_i or attn_mask[tok_idx] == 0:
                    continue
                if start_i < len(char_labels):
                    span = char_labels[start_i:min(end_i, len(char_labels))]
                    valid = [s for s in span if s >= 0]
                    tok_labels[tok_idx] = max(valid) if valid else -1

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
            "pivot_kind_labels": torch.full((B,), 4, dtype=torch.long),
            "metadata": metadata,
        }


def build_weighted_sampler(records: List[Dict], config: GuardLensConfig) -> WeightedRandomSampler:
    weights = []
    for r in records:
        tier = r.get("supervision_tier", "construction")
        if tier in ("cf_strong", "cf_weak"):
            weights.append(float(config.cf_oversample_factor))
        elif tier == "llm_confirmed":
            weights.append(1.5)
        else:
            weights.append(1.0)
    return WeightedRandomSampler(weights=weights, num_samples=len(records), replacement=True)
