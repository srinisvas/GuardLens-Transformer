"""Fail-closed dataset and dynamic collator for causal localization."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from guardlens.config import GuardLensConfig
from guardlens.data.causal_targets import (
    build_evidence_turn_targets,
    span_supervision_target,
)
from guardlens.data.training_contract import (
    classification_loss_weight,
    is_auxiliary_detection_record,
    localization_supervision_ignored,
    training_label,
)


class GuardLensDataset(Dataset):
    """Convert frozen records to model-visible text and supervision targets."""

    def __init__(self, records: List[Dict], config: GuardLensConfig):
        self.records = records
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        cid = str(record.get("conversation_id", "")) or "<missing>"
        turns = list(record.get("turns", []))
        if not turns:
            raise RuntimeError(f"{cid}: empty realized trajectory")
        if len(turns) > self.config.max_turns:
            raise RuntimeError(
                f"{cid}: {len(turns)} turns exceed max_turns={self.config.max_turns}; "
                "silent turn truncation is forbidden"
            )

        expected_ids = list(range(len(turns)))
        realized_ids = [turn.get("turn_id") for turn in turns]
        if realized_ids != expected_ids:
            raise RuntimeError(
                f"{cid}: realized turn_id values must equal indices 0..{len(turns)-1}"
            )

        localization_ignore = localization_supervision_ignored(record)
        turn_texts: List[str] = []
        turn_roles: List[int] = []
        char_labels_per_turn: List[List[int]] = []
        char_weights_per_turn: List[List[float]] = []

        for turn in turns:
            text = str(turn.get("text", ""))
            role_name = str(turn.get("role", "")).lower()
            if role_name not in {"user", "assistant"}:
                raise RuntimeError(f"{cid}: unsupported turn role {role_name!r}")
            turn_texts.append(text)
            turn_roles.append(0 if role_name == "user" else 1)

            char_labels = [-1] * len(text)
            char_weights = [0.0] * len(text)
            if not localization_ignore:
                for span in turn.get("span_annotations", []) or []:
                    target = span_supervision_target(span)
                    if target is None:
                        continue
                    token_label, tier_weight = target
                    cs = span.get("char_start")
                    ce = span.get("char_end")
                    if (
                        isinstance(cs, bool) or isinstance(ce, bool)
                        or not isinstance(cs, int) or not isinstance(ce, int)
                        or cs < 0 or ce <= cs or ce > len(text)
                    ):
                        raise RuntimeError(
                            f"{cid}: invalid supervised span offsets {cs!r}:{ce!r} "
                            f"for text length {len(text)}"
                        )
                    for char_idx in range(cs, ce):
                        # Positive evidence wins only if overlapping annotations disagree.
                        if token_label == 1 or char_labels[char_idx] == -1:
                            char_labels[char_idx] = token_label
                            char_weights[char_idx] = float(tier_weight)

            char_labels_per_turn.append(char_labels)
            char_weights_per_turn.append(char_weights)

        evidence_turn_labels, evidence_turn_weights = build_evidence_turn_targets(
            record, turns
        )

        return {
            "turn_texts": turn_texts,
            "turn_roles": turn_roles,
            "label": training_label(record),
            "detection_weight": classification_loss_weight(record),
            "auxiliary_detection_only": is_auxiliary_detection_record(record),
            "localization_supervision_ignore": localization_ignore,
            "char_labels": char_labels_per_turn,
            "char_tier_weights": char_weights_per_turn,
            "evidence_turn_labels": evidence_turn_labels,
            "evidence_turn_weights": evidence_turn_weights,
            "conversation_id": record.get("conversation_id", ""),
            "difficulty": record.get("difficulty", "medium"),
            "family": record.get("family", "unknown"),
            "evidence_turn_ids": list(record.get("evidence_turn_ids") or []),
            "pivot_turn_id": record.get("pivot_turn_id"),
            "pivot_kind": record.get("pivot_kind", "none"),
            "supervision_tier": record.get("supervision_tier", "unknown"),
            "transfer_tier": record.get("transfer_tier", "unknown"),
            "benign_status": record.get("benign_status", "none"),
        }


class GuardLensCollator:
    """Tokenize without truncation and dynamically pad each batch.

    max_tokens_per_turn is enforced as a hard ceiling. A record that exceeds it
    raises instead of silently dropping text or localization targets.
    """

    def __init__(self, tokenizer, config: GuardLensConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

    def _encode_turn(self, item: Dict, t_idx: int) -> Dict:
        text = item["turn_texts"][t_idx]
        enc = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])
        if len(ids) > self.config.max_tokens_per_turn:
            cid = item.get("conversation_id", "<missing>")
            raise RuntimeError(
                f"{cid}: turn {t_idx} tokenizes to {len(ids)} tokens, exceeding "
                f"max_tokens_per_turn={self.config.max_tokens_per_turn}; "
                "silent token truncation is forbidden"
            )

        char_labels = item["char_labels"][t_idx]
        char_weights = item["char_tier_weights"][t_idx]
        token_labels = [-1] * len(ids)
        token_weights = [0.0] * len(ids)

        for tok_idx, pair in enumerate(offsets):
            start_i, end_i = int(pair[0]), int(pair[1])
            if end_i <= start_i:
                continue
            span_labels = char_labels[start_i:end_i]
            span_weights = char_weights[start_i:end_i]
            if not span_labels:
                continue

            # Positive evidence wins an overlap, but its confidence weight must
            # come from positive characters only. Otherwise an overlapping
            # incidental span could silently upgrade a weak positive to 1.0.
            if 1 in span_labels:
                token_labels[tok_idx] = 1
                token_weights[tok_idx] = max(
                    weight
                    for label, weight in zip(span_labels, span_weights)
                    if label == 1
                )
            elif 0 in span_labels:
                token_labels[tok_idx] = 0
                token_weights[tok_idx] = max(
                    weight
                    for label, weight in zip(span_labels, span_weights)
                    if label == 0
                )

        return {
            "ids": ids,
            "token_labels": token_labels,
            "token_weights": token_weights,
        }

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise RuntimeError("empty batch")

        max_turns = max(len(item["turn_texts"]) for item in batch)
        if max_turns > self.config.max_turns:
            raise RuntimeError("batch exceeds configured max_turns")

        encoded_batch: List[List[Dict]] = []
        max_seq_len = 1
        for item in batch:
            encoded_turns = []
            for t_idx in range(len(item["turn_texts"])):
                encoded = self._encode_turn(item, t_idx)
                encoded_turns.append(encoded)
                max_seq_len = max(max_seq_len, len(encoded["ids"]))
            encoded_batch.append(encoded_turns)

        all_input_ids = []
        all_attention_masks = []
        all_turn_masks = []
        all_role_ids = []
        all_token_labels = []
        all_span_weights = []
        all_labels = []
        all_detection_weights = []
        all_turn_labels = []
        all_turn_weights = []
        metadata = []

        for item, encoded_turns in zip(batch, encoded_batch):
            turn_input_ids = []
            turn_attention_masks = []
            turn_mask = []
            turn_role_ids = []
            turn_token_labels = []
            turn_span_weights = []
            turn_labels = []
            turn_weights = []

            for t_idx in range(max_turns):
                if t_idx < len(encoded_turns):
                    encoded = encoded_turns[t_idx]
                    n = len(encoded["ids"])
                    pad = max_seq_len - n
                    turn_input_ids.append(torch.tensor(
                        encoded["ids"] + [self.pad_token_id] * pad,
                        dtype=torch.long,
                    ))
                    turn_attention_masks.append(torch.tensor(
                        [1] * n + [0] * pad, dtype=torch.long
                    ))
                    turn_token_labels.append(torch.tensor(
                        encoded["token_labels"] + [-1] * pad, dtype=torch.long
                    ))
                    turn_span_weights.append(torch.tensor(
                        encoded["token_weights"] + [0.0] * pad, dtype=torch.float
                    ))
                    turn_mask.append(1)
                    turn_role_ids.append(item["turn_roles"][t_idx])
                    turn_labels.append(item["evidence_turn_labels"][t_idx])
                    turn_weights.append(item["evidence_turn_weights"][t_idx])
                else:
                    turn_input_ids.append(torch.full(
                        (max_seq_len,), self.pad_token_id, dtype=torch.long
                    ))
                    turn_attention_masks.append(torch.zeros(
                        max_seq_len, dtype=torch.long
                    ))
                    turn_token_labels.append(torch.full(
                        (max_seq_len,), -1, dtype=torch.long
                    ))
                    turn_span_weights.append(torch.zeros(
                        max_seq_len, dtype=torch.float
                    ))
                    turn_mask.append(0)
                    turn_role_ids.append(0)
                    turn_labels.append(-1)
                    turn_weights.append(0.0)

            all_input_ids.append(torch.stack(turn_input_ids))
            all_attention_masks.append(torch.stack(turn_attention_masks))
            all_turn_masks.append(torch.tensor(turn_mask, dtype=torch.long))
            all_role_ids.append(torch.tensor(turn_role_ids, dtype=torch.long))
            all_token_labels.append(torch.stack(turn_token_labels))
            all_span_weights.append(torch.stack(turn_span_weights))
            all_labels.append(item["label"])
            all_detection_weights.append(item["detection_weight"])
            all_turn_labels.append(torch.tensor(turn_labels, dtype=torch.long))
            all_turn_weights.append(torch.tensor(turn_weights, dtype=torch.float))
            metadata.append({
                "conversation_id": item["conversation_id"],
                "difficulty": item["difficulty"],
                "family": item["family"],
                "evidence_turn_ids": item["evidence_turn_ids"],
                "pivot_turn_id": item["pivot_turn_id"],
                "pivot_kind": item["pivot_kind"],
                "auxiliary_detection_only": item["auxiliary_detection_only"],
                "localization_supervision_ignore": item[
                    "localization_supervision_ignore"
                ],
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
            "detection_weights": torch.tensor(
                all_detection_weights, dtype=torch.float
            ),
            "turn_labels": torch.stack(all_turn_labels),
            "turn_weights": torch.stack(all_turn_weights),
            "metadata": metadata,
        }


class FlatConversationCollator:
    """Legacy flat baseline collator retained until baseline migration."""

    def __init__(self, tokenizer, config: GuardLensConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.sep = tokenizer.sep_token or "[SEP]"

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        ids_list = []
        masks = []
        labels = []
        detection_weights = []
        metadata = []

        for item in batch:
            full_text = f" {self.sep} ".join(item["turn_texts"])
            enc = self.tokenizer(
                full_text,
                max_length=self.config.max_total_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids_list.append(enc["input_ids"].squeeze(0))
            masks.append(enc["attention_mask"].squeeze(0))
            labels.append(item["label"])
            detection_weights.append(item["detection_weight"])
            metadata.append({
                "conversation_id": item["conversation_id"],
                "difficulty": item["difficulty"],
                "family": item["family"],
                "evidence_turn_ids": item["evidence_turn_ids"],
                "supervision_tier": item["supervision_tier"],
            })

        batch_size = len(batch)
        return {
            "input_ids": torch.stack(ids_list),
            "attention_mask": torch.stack(masks),
            "turn_mask": torch.ones(batch_size, 1, dtype=torch.long),
            "role_ids": torch.zeros(batch_size, 1, dtype=torch.long),
            "token_labels": torch.full(
                (batch_size, self.config.max_total_tokens), -1, dtype=torch.long
            ),
            "span_weights": torch.zeros(
                batch_size, self.config.max_total_tokens, dtype=torch.float
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "detection_weights": torch.tensor(detection_weights, dtype=torch.float),
            "turn_labels": torch.full((batch_size, 1), -1, dtype=torch.long),
            "turn_weights": torch.zeros(batch_size, 1, dtype=torch.float),
            "metadata": metadata,
        }
