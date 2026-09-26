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
        eligible_set = set(eligible)
        if (len(set(matched)) != len(matched) or
                any(type(i) is not int or i not in eligible_set for i in matched)):
            raise ValueError("span-random reference must be unique and eligible")

        # Preserve the reference run-length multiset within each contiguous
        # eligible block and turn. Randomize run order and surrounding gaps.
        # This construction is linear and always succeeds because the original
        # matched selection proves that every block has sufficient capacity.
        def contiguous_runs(indices):
            runs = []
            for i in sorted(indices):
                if (runs and i == runs[-1][-1] + 1 and
                        units[i]["turn_id"] == units[runs[-1][-1]]["turn_id"]):
                    runs[-1].append(i)
                else:
                    runs.append([i])
            return runs

        blocks = contiguous_runs(eligible)
        block_by_index = {
            index: block_id
            for block_id, block in enumerate(blocks)
            for index in block
        }
        lengths_by_block = [[] for _ in blocks]
        for run in contiguous_runs(matched):
            block_ids = {block_by_index.get(index) for index in run}
            if len(block_ids) != 1 or None in block_ids:
                raise ValueError("matched span crosses an ineligible boundary")
            lengths_by_block[block_ids.pop()].append(len(run))

        result = []
        for block, lengths in zip(blocks, lengths_by_block, strict=True):
            if not lengths:
                continue
            rng.shuffle(lengths)
            # Distinct reference runs are separated by at least one eligible,
            # unselected word. Reserve those internal separators before
            # randomizing the remaining slack so adjacent sampled runs cannot
            # merge into a different run-length multiset.
            internal_separators = len(lengths) - 1
            free_slack = len(block) - sum(lengths) - internal_separators
            if free_slack < 0:
                raise ValueError("matched spans exceed eligible block capacity")
            # Uniform stars-and-bars composition of the remaining slack into
            # leading, internal, and trailing gaps. Each internal gap then gets
            # its mandatory separator.
            bars = sorted(rng.sample(
                range(free_slack + len(lengths)), len(lengths)
            ))
            points = [-1, *bars, free_slack + len(lengths)]
            gaps = [points[i + 1] - points[i] - 1 for i in range(len(points) - 1)]
            for gap_index in range(1, len(gaps) - 1):
                gaps[gap_index] += 1
            cursor = gaps[0]
            for position, length in enumerate(lengths):
                result.extend(block[cursor:cursor + length])
                cursor += length + gaps[position + 1]
        if len(result) != count or len(set(result)) != count:
            raise RuntimeError("span-random construction violated the exact budget")
        reference_shape = sorted(
            (block_by_index[run[0]], len(run))
            for run in contiguous_runs(matched)
        )
        result_shape = sorted(
            (block_by_index[run[0]], len(run))
            for run in contiguous_runs(result)
        )
        if result_shape != reference_shape:
            raise RuntimeError("span-random construction changed the matched runs")
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
            "total_user_words": len(units), "per_turn_count": {str(k): v for k, v in Counter(units[i]["turn_id"] for i in selected).items()},
            "changed_turn_ids": [a["turn_id"] for a, b in zip(turns, changed, strict=True) if a["text"] != b["text"]]}
