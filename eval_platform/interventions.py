"""Exact character-offset textual interventions, before re-tokenization."""
import math
import random
import re
from collections import Counter


def words(turns):
    """Budget unit: Unicode non-whitespace runs, including attached punctuation."""
    return [{"turn_id": t["turn_id"], "start": m.start(), "end": m.end(), "text": m.group()}
            for t in turns if t["role"] == "user" for m in re.finditer(r"\S+", t["text"])]


def risk_scores(units, lexicon):
    lexicon = {x.casefold() for x in lexicon}
    return [float(bool(set(re.findall(r"\w+", w["text"].casefold())) & lexicon)) for w in units]


def eligible_indices(units, turns, scope, surface):
    last = max(t["turn_id"] for t in turns if t["role"] == "user")
    return [i for i, w in enumerate(units)
            if (scope != "context_only" or w["turn_id"] != last)
            and (scope != "non_surface" or surface[i] == 0)]


def select(units, scores, fraction, eligible, method="rank", seed=0, matched=None):
    if len(scores) != len(units) or any(not math.isfinite(s) for s in scores):
        raise ValueError("one finite score required per word")
    if not 0 < fraction <= 1 or len(set(eligible)) != len(eligible):
        raise ValueError("invalid intervention budget")
    # All methods use the same eligible population and exact count, including ties.
    count = min(len(eligible), math.ceil(fraction * len(eligible)))
    rng = random.Random(seed)
    if method == "random":
        return sorted(rng.sample(eligible, count))
    if method == "span_random":
        if matched is None or len(matched) != count:
            raise ValueError("span-random needs the reference selection")
        # Match each selected contiguous run length and its user turn. Place
        # nonoverlapping runs by randomized backtracking. No count fallback.
        runs = []
        for i in sorted(matched):
            if runs and i == runs[-1][-1] + 1 and units[i]["turn_id"] == units[runs[-1][-1]]["turn_id"]:
                runs[-1].append(i)
            else:
                runs.append([i])
        available = set(eligible)
        tasks = sorted([(units[r[0]]["turn_id"], len(r)) for r in runs], key=lambda x: -x[1])
        def place(pos, occupied):
            if pos == len(tasks):
                return occupied
            tid, size = tasks[pos]
            starts = [i for i in eligible if all(j in available and j not in occupied and units[j]["turn_id"] == tid for j in range(i, i + size))]
            rng.shuffle(starts)
            for start in starts:
                result = place(pos + 1, occupied | set(range(start, start + size)))
                if result is not None:
                    return result
            return None
        result = place(0, set())
        if result is None:
            raise ValueError("cannot place matched random spans")
        return sorted(result)
    if method != "rank":
        raise ValueError("unknown selector")
    return sorted(sorted(eligible, key=lambda i: (-scores[i], units[i]["turn_id"], units[i]["start"]))[:count])


def edit(turns, units, selected, operation="delete", keep=False):
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(units) for i in selected):
        raise ValueError("invalid selected word indices")
    if operation not in {"delete", "blank"}:
        raise ValueError("unknown operation")
    chosen = set(selected)
    ranges = {}
    for i, w in enumerate(units):
        if (i in chosen) != keep:
            ranges.setdefault(w["turn_id"], []).append(w)
    result = []
    for t in turns:
        text = t["text"]
        for w in sorted(ranges.get(t["turn_id"], []), key=lambda x: -x["start"]):
            if t["role"] != "user" or text[w["start"]:w["end"]] != w["text"]:
                raise ValueError("stale offsets or attempted non-user edit")
            replacement = "" if operation == "delete" else " " * (w["end"] - w["start"])
            text = text[:w["start"]] + replacement + text[w["end"]:]
        result.append({**t, "text": text})
    return result


def project_tokens(units, token_spans):
    """Max over overlapping model subwords, never decode/reconstruct the text."""
    return [max((s["score"] for s in token_spans if s["turn_id"] == w["turn_id"] and s["start"] < w["end"] and s["end"] > w["start"]), default=0.0) for w in units]


def loto_scores(turns, score):
    """Actual leave-one-user-turn-out detector re-inference, signed probability drop."""
    original = score(turns)
    return {t["turn_id"]: original - score([u for u in turns if u["turn_id"] != t["turn_id"]])
            for t in turns if t["role"] == "user"}


def audit_edit(turns, units, selected, changed):
    return {"selected_words": [units[i] for i in selected], "selected_count": len(selected),
            "total_user_words": len(units), "per_turn_count": dict(Counter(units[i]["turn_id"] for i in selected)),
            "changed_turn_ids": [a["turn_id"] for a, b in zip(turns, changed) if a["text"] != b["text"]]}
