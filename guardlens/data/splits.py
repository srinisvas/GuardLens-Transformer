"""Pair-aware train/val/test splitting."""

import random
from typing import Dict, List, Tuple


def pair_aware_split(
    records: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split by pair_id so both twins go into the same split.
    Prevents leakage where the model sees one twin's topic
    during training and gets a free hint on the other during eval.
    """
    rng = random.Random(seed)

    pair_map = {}
    for idx, r in enumerate(records):
        pid = r.get("pair_id", str(idx))
        pair_map.setdefault(pid, []).append(idx)

    pair_ids = list(pair_map.keys())
    rng.shuffle(pair_ids)

    n_train = int(len(pair_ids) * train_ratio)
    n_val = int(len(pair_ids) * val_ratio)

    train_pairs = pair_ids[:n_train]
    val_pairs = pair_ids[n_train:n_train + n_val]
    test_pairs = pair_ids[n_train + n_val:]

    train_idx = [i for pid in train_pairs for i in pair_map[pid]]
    val_idx = [i for pid in val_pairs for i in pair_map[pid]]
    test_idx = [i for pid in test_pairs for i in pair_map[pid]]

    return train_idx, val_idx, test_idx