"""Dataset and collation for GuardLens."""

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from guardlens.config import GuardLensConfig


class GuardLensDataset(Dataset):
    """
    Loads JSONL records and prepares them for the hierarchical model.

    Each sample produces:
      - turn_texts: List[str] of turn texts
      - turn_roles: List[int] (0=user, 1=assistant)
      - label: int (0=benign, 1=adversarial)
      - char_labels: per-turn character-level binary labels
      - metadata: dict for evaluation breakdowns
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
        token_labels_per_turn = []

        for turn in turns:
            text = turn["text"][:500]
            role = 0 if turn["role"] == "user" else 1
            turn_texts.append(text)
            turn_roles.append(role)

            # Build per-character label array
            char_labels = [0] * len(text)
            for span in turn.get("span_annotations", []):
                label_name = span.get("label", "")
                cs = span.get("char_start", 0)
                ce = span.get("char_end", 0)

                if label_name in self.config.causal_span_labels:
                    for i in range(cs, min(ce, len(text))):
                        char_labels[i] = 1

            token_labels_per_turn.append(char_labels)

        return {
            "turn_texts": turn_texts,
            "turn_roles": turn_roles,
            "label": record["label"],
            "char_labels": token_labels_per_turn,
            "conversation_id": record.get("conversation_id", ""),
            "difficulty": record.get("difficulty", "medium"),
            "family": record.get("family", "unknown"),
            "pivot_turn_id": record.get("pivot_turn_id"),
        }


class GuardLensCollator:
    """
    Tokenizes turns, aligns character-level labels to token-level,
    and pads everything into tensors.

    Output shapes:
      input_ids:     [B, T, S]
      attention_mask: [B, T, S]
      turn_mask:     [B, T]
      role_ids:      [B, T]
      token_labels:  [B, T, S]  (1=causal, 0=not, -1=ignore)
      labels:        [B]
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
        all_labels = []
        metadata = []

        for item in batch:
            turn_input_ids = []
            turn_attention_masks = []
            turn_mask = []
            turn_role_ids = []
            turn_token_labels = []

            for t_idx in range(max_turns):
                if t_idx < len(item["turn_texts"]):
                    text = item["turn_texts"][t_idx]
                    role = item["turn_roles"][t_idx]
                    char_labels = item["char_labels"][t_idx]

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

                    # Align char labels to token labels
                    tok_labels = torch.full(
                        (self.config.max_tokens_per_turn,), -1,
                        dtype=torch.long,
                    )
                    for tok_idx, (start, end) in enumerate(offsets):
                        if end <= start:
                            # Special token, padding, or degenerate offset
                            tok_labels[tok_idx] = -1
                            continue
                        if attn_mask[tok_idx] == 0:
                            tok_labels[tok_idx] = -1
                            continue
                        span_labels = char_labels[start:end]
                        tok_labels[tok_idx] = max(span_labels) if span_labels else 0

                    turn_input_ids.append(input_ids)
                    turn_attention_masks.append(attn_mask)
                    turn_mask.append(1)
                    turn_role_ids.append(role)
                    turn_token_labels.append(tok_labels)
                else:
                    # Padding turn
                    s = self.config.max_tokens_per_turn
                    turn_input_ids.append(torch.zeros(s, dtype=torch.long))
                    turn_attention_masks.append(torch.zeros(s, dtype=torch.long))
                    turn_mask.append(0)
                    turn_role_ids.append(0)
                    turn_token_labels.append(
                        torch.full((s,), -1, dtype=torch.long)
                    )

            all_input_ids.append(torch.stack(turn_input_ids))
            all_attention_masks.append(torch.stack(turn_attention_masks))
            all_turn_masks.append(torch.tensor(turn_mask, dtype=torch.long))
            all_role_ids.append(torch.tensor(turn_role_ids, dtype=torch.long))
            all_token_labels.append(torch.stack(turn_token_labels))
            all_labels.append(item["label"])

            metadata.append({
                "conversation_id": item["conversation_id"],
                "difficulty": item["difficulty"],
                "family": item["family"],
                "pivot_turn_id": item["pivot_turn_id"],
            })

        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_masks),
            "turn_mask": torch.stack(all_turn_masks),
            "role_ids": torch.stack(all_role_ids),
            "token_labels": torch.stack(all_token_labels),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "metadata": metadata,
        }


class FlatConversationCollator:
    """
    Collator for ConversationDeBERTa baseline.

    Concatenates all turns into a single sequence with [SEP] between
    them, truncates to max_total_tokens. No turn structure preserved.

    Output shapes:
      input_ids:     [B, L]
      attention_mask: [B, L]
      labels:        [B]
      token_labels:  [B, L]  (for compatibility, mostly -1)
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
        metadata = []

        for item in batch:
            # Concatenate turns with [SEP]
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

            # Build char labels for the concatenated text
            char_labels = []
            for cl in item["char_labels"]:
                char_labels.extend(cl)
                char_labels.extend([0] * (len(f" {self.sep} ")))
            char_labels = char_labels[:len(full_text)]

            # Align to tokens
            tok_labels = torch.full(
                (self.config.max_total_tokens,), -1, dtype=torch.long,
            )
            for tok_idx, (start, end) in enumerate(offsets):
                if end <= start or attn_mask[tok_idx] == 0:
                    continue
                if start < len(char_labels):
                    span = char_labels[start:min(end, len(char_labels))]
                    tok_labels[tok_idx] = max(span) if span else 0
                else:
                    tok_labels[tok_idx] = 0

            all_input_ids.append(input_ids)
            all_attention_masks.append(attn_mask)
            all_token_labels.append(tok_labels)
            all_labels.append(item["label"])
            metadata.append({
                "conversation_id": item["conversation_id"],
                "difficulty": item["difficulty"],
                "family": item["family"],
                "pivot_turn_id": item["pivot_turn_id"],
            })

        # Dummy turn_mask and role_ids for interface compatibility
        B = len(all_input_ids)
        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_masks),
            "turn_mask": torch.ones(B, 1, dtype=torch.long),
            "role_ids": torch.zeros(B, 1, dtype=torch.long),
            "token_labels": torch.stack(all_token_labels),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "metadata": metadata,
        }
