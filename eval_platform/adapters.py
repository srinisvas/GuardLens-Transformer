"""Source-specific conversion. Never manufacture evidence labels or benign controls."""
from __future__ import annotations

from collections import Counter

from .contract import digest, validate_record

ROLE = {"user": "user", "human": "user", "assistant": "assistant", "model": "assistant", "gpt": "assistant", "ai": "assistant", "system": "system"}


def turns_from(raw):
    if not isinstance(raw, list):
        raise ValueError("expected a list of realized messages, not a flattened prompt")
    result = []
    for i, t in enumerate(raw):
        role = ROLE.get(str(t.get("role", "")).lower())
        if role is None:
            raise ValueError(f"unsupported source role: {t.get('role')!r}")
        text = t.get("text", t.get("content"))
        result.append({"turn_id": i, "role": role, "text": text})
    return result


def adapt(raw, source, revision, split, subset=None):
    if not revision or revision in {"main", "latest"}:
        raise ValueError("provide an immutable source revision or export SHA256")
    if source == "canonical":
        if raw["source_revision"] != revision or raw["split"] != split:
            raise ValueError("canonical provenance differs from declared import")
        return validate_record(raw)
    if source == "mtid":
        if subset not in {"harmful", "benign"}:
            raise ValueError("MTID requires its original harmful/benign subset")
        turns = turns_from(raw["conversation"])
        label = int(subset == "harmful")  # NOT is_jailbreak_success (true also on benign).
        cluster = f"mtid:{subset}:{split}:{raw['sample_index']}"
        rid = f"{cluster}:{raw['rollout_id']}"
        objective = raw.get("meta_intent")
        semantics = "attack_intent"
    elif source == "mhj":
        turns = turns_from(raw.get("turns", raw.get("messages", raw.get("conversation"))))
        label, semantics = 1, "attack_intent"
        rid = "mhj:" + str(raw.get("conversation_id", raw.get("id", digest(turns))))
        # Behavior/task ID clusters multiple submissions against the same goal.
        cluster = "mhj:" + str(raw.get("behavior_id", raw.get("task_id", rid)))
        objective = raw.get("objective", raw.get("behavior"))
    elif source == "internal":
        # Uses the exact authoritative training target builder, never pivot fallback.
        from guardlens.data.causal_targets import build_evidence_turn_targets, span_supervision_target
        from guardlens.data.training_contract import training_label, localization_supervision_ignored, is_auxiliary_detection_record
        turns = turns_from(raw["turns"])
        label, semantics = training_label(raw), "unsafe_trajectory"
        rid = str(raw["conversation_id"])
        cluster = str(raw.get("split_group_id") or raw.get("pair_id") or rid)
        objective = raw.get("objective")
        labels, _ = build_evidence_turn_targets(raw, raw["turns"])
        gold = {"turns": {str(i): y for i, y in enumerate(labels) if y >= 0}, "spans": [], "provenance": "restored_a_targets_v2"}
        ignored = localization_supervision_ignored(raw) or is_auxiliary_detection_record(raw)
        if not ignored:
            for i, turn in enumerate(raw["turns"]):
                for span in turn.get("span_annotations", []) or []:
                    target = span_supervision_target(span)
                    if target is not None and turn["role"] == "user":
                        gold["spans"].append({"turn_id": i, "start": span["char_start"], "end": span["char_end"], "label": target[0]})
    else:
        raise ValueError("unsupported source. Use documented canonical schema for other public datasets")
    r = {"id": rid, "cluster_id": cluster, "dataset": source, "split": split,
         "source_revision": revision, "label": label, "label_semantics": semantics,
         "turns": turns, "gold": gold if source == "internal" else {},
         "strata": {k: str(raw.get(k, "unknown")) for k in ("family", "supervision_tier", "corpus_source", "difficulty", "pivot_kind", "transfer_tier")},
         "objective": objective, "raw_sha256": digest(raw)}
    return validate_record(r)


def check_overlap(records, reference_records):
    """Exact transcript, stable ID, and known group overlap are hard errors."""
    keys = lambda r: {("id", r["id"]), ("cluster", r["cluster_id"]), ("text", digest(r["turns"]))}
    forbidden = set().union(*(keys(r) for r in reference_records)) if reference_records else set()
    leaked = [r["id"] for r in records if keys(r) & forbidden]
    if leaked:
        raise ValueError(f"evaluation overlaps training/dev exclusions: {leaked[:10]}")


def require_mhj_cohort(records):
    """An MHJ-specific launcher must not silently evaluate internal data."""
    if not records or any(r.get("dataset") != "mhj" or r.get("label_semantics") != "attack_intent" for r in records):
        raise ValueError("eval_mhj requires prepared MHJ attack_intent records, not an internal or mixed cohort")
    return len(records)


def require_internal_dev_strata(records):
    """Verify named benign utility cohorts on the real prepared dev freeze."""
    if not records or any(
        r.get("dataset") != "internal"
        or r.get("split") not in {"dev", "valid", "validation"}
        or r.get("label_semantics") != "unsafe_trajectory"
        for r in records
    ):
        raise ValueError("expected one prepared internal unsafe-trajectory dev cohort")
    families = Counter(
        r.get("strata", {}).get("family")
        for r in records if r.get("label") == 0
    )
    missing = sorted(
        name for name in ("frontier_authored_benign", "interactive_benign_twin")
        if families[name] == 0
    )
    if missing:
        raise ValueError(
            "prepared internal dev is missing named benign utility strata: "
            + ", ".join(missing)
        )
    return dict(sorted(families.items()))


def validate_collection(records):
    seen = set()
    for r in records:
        validate_record(r)
        if r["id"] in seen:
            raise ValueError(f"duplicate ID {r['id']}")
        seen.add(r["id"])
    if not records:
        raise ValueError("empty evaluation collection")
    if len({r["label_semantics"] for r in records}) != 1:
        raise ValueError("do not pool attack-intent and realized-outcome tasks")
    return records
