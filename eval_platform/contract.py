"""Immutable experiment identity, strict JSON I/O and model-visible views."""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path

from . import VERSION


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line, text in enumerate(f, 1):
            if not text.strip():
                raise ValueError(f"{path}:{line}: blank record")
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line}: expected an object")
            yield value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temp.write_text(canonical(value) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid probability {value!r}")
    return float(value)


def validate_record(r):
    for key in ("id", "cluster_id", "dataset", "split", "source_revision", "label_semantics"):
        if not isinstance(r.get(key), str) or not r[key]:
            raise ValueError(f"missing {key}")
    if type(r.get("label")) is not int or r["label"] not in (0, 1):
        raise ValueError("label must be a source-supported binary integer")
    if r["label_semantics"] not in {"attack_intent", "unsafe_trajectory"}:
        raise ValueError("unsupported label semantics")
    turns = r.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("empty conversation")
    for i, t in enumerate(turns):
        if set(t) != {"role", "text", "turn_id"} or t["turn_id"] != i:
            raise ValueError("turns must contain only role, text, sequential turn_id")
        if t["role"] not in {"user", "assistant", "system"} or not isinstance(t["text"], str) or not t["text"].strip():
            raise ValueError("invalid role or empty text")
        if t["role"] == "system" and i != 0:
            raise ValueError("only an initial system turn is supported")
    if not any(t["role"] == "user" for t in turns):
        raise ValueError("no user turn")
    gold = r.get("gold", {})
    if set(gold) - {"turns", "spans", "provenance"}:
        raise ValueError("unknown gold fields")
    for tid, target in gold.get("turns", {}).items():
        idx = int(tid)
        if str(idx) != str(tid) or not 0 <= idx < len(turns) or turns[idx]["role"] != "user" or type(target) is not int or target not in (0, 1):
            raise ValueError("invalid assessed user-turn label")
    for s in gold.get("spans", []):
        tid, a, b, y = (s[k] for k in ("turn_id", "start", "end", "label"))
        if any(type(x) is not int for x in (tid, a, b, y)) or not 0 <= tid < len(turns) or turns[tid]["role"] != "user" or not 0 <= a < b <= len(turns[tid]["text"]) or y not in (0, 1):
            raise ValueError("invalid assessed character span")
    if gold and not gold.get("provenance"):
        raise ValueError("gold requires annotation provenance")
    return r


def visible(r, view):
    """No labels, rationales, objectives, annotations or source metadata enter a model."""
    if view not in {"pre_response", "retrospective"}:
        raise ValueError("view must be pre_response or retrospective")
    turns = r["turns"]
    if view == "pre_response":
        end = max(i for i, t in enumerate(turns) if t["role"] == "user") + 1
        turns = turns[:end]
    return [{"role": t["role"], "text": t["text"], "turn_id": t["turn_id"]} for t in turns]


def validate_protocol(p):
    allowed_fields = {
        "version", "view", "threshold_source", "turn_threshold",
        "span_threshold", "guard_threshold", "budgets", "methods", "scope",
        "operation", "context_diagnostics", "random_repeats", "seed",
        "bootstrap_repeats", "utility_lambdas",
    }
    unknown_fields = set(p) - allowed_fields
    if unknown_fields:
        raise ValueError(f"unknown protocol fields: {sorted(unknown_fields)}")
    if p.get("version") != VERSION:
        raise ValueError(f"protocol version must be {VERSION}")
    if p.get("view") not in {"pre_response", "retrospective"}:
        raise ValueError("freeze an explicit view")
    if p.get("threshold_source") != "checkpoint_internal_dev":
        raise ValueError("external/test threshold fitting is forbidden")
    for k in ("turn_threshold", "span_threshold", "guard_threshold"):
        probability(p[k])
    if not p.get("budgets") or any(isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 < b <= 1 for b in p["budgets"]):
        raise ValueError("budgets must be fractions in (0,1]")
    if type(p.get("random_repeats")) is not int or p["random_repeats"] < 2:
        raise ValueError("at least two random repeats required")
    if p.get("scope") not in {"all_user", "context_only", "non_surface"}:
        raise ValueError("invalid intervention scope")
    if p.get("operation") not in {"delete", "blank"}:
        raise ValueError("invalid intervention operation")
    if type(p.get("context_diagnostics")) is not bool:
        raise ValueError("context_diagnostics must be boolean")
    allowed = {"guardlens", "random", "span_random", "surface", "last_user", "loto", "llm"}
    if not p.get("methods") or set(p["methods"]) - allowed:
        raise ValueError("unknown attribution method")
    if len(set(p["methods"])) != len(p["methods"]):
        raise ValueError("duplicate methods")
    if p.get("bootstrap_repeats", 0) < 100:
        raise ValueError("use at least 100 cluster bootstrap replicates")
    if type(p.get("seed")) is not int or not p.get("utility_lambdas") or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in p["utility_lambdas"]):
        raise ValueError("freeze a seed and nonnegative finite utility penalties")
    if len(set(p["budgets"])) != len(p["budgets"]):
        raise ValueError("duplicate budgets")
    return p


class RunStore:
    """A cache hit requires the complete manifest and request payload to match."""
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = {"platform_version": VERSION, **manifest}
        self.manifest_id = digest(self.manifest)
        path = self.root / "manifest.json"
        envelope = {"manifest_id": self.manifest_id, **self.manifest}
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != envelope:
            raise ValueError("run directory belongs to a different manifest")
        write_json(path, envelope)

    def cached(self, namespace, payload, compute):
        key = digest({"manifest": self.manifest_id, "namespace": namespace, "payload": payload})
        path = self.root / "cache" / namespace / (key + ".json")
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("key") != key or row.get("payload") != payload or digest(row["result"]) != row.get("result_sha256"):
                raise ValueError("cache corruption")
            return row["result"]
        result = compute()
        write_json(path, {"key": key, "payload": payload, "result": result, "result_sha256": digest(result)})
        return result
