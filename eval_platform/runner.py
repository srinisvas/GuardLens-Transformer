"""One attribution plan shared by self-faithfulness, independent guards and replay."""
import json
from collections import defaultdict

from .contract import digest, visible, write_json
from .interventions import words, risk_scores, eligible_indices, select, edit, project_tokens, loto_scores, audit_edit
from .metrics import binary, average, localization, span_agreement, effects, cluster_interval, utility_grid
from .runtime import CoverageError


def infer(store, name, backend, turns):
    def compute():
        try:
            return backend.predict(turns)
        except CoverageError as exc:
            return {"error": "context_coverage", "detail": str(exc)}
        # Runtime/model errors fail the job. They must not become safe predictions.
    return store.cached("inference", {"backend": name, "turns": turns}, compute)


def intervention_plan(turns, prediction, protocol, lexicon, detector_score=None, llm_prediction=None):
    units = words(turns)
    surface = risk_scores(units, lexicon)
    eligible = eligible_indices(units, turns, protocol["scope"], surface)
    gl_scores = project_tokens(units, prediction["token_scores"])
    last = max(t["turn_id"] for t in turns if t["role"] == "user")
    method_scores = {"guardlens": gl_scores, "surface": surface,
                     "last_user": [float(w["turn_id"] == last) for w in units]}
    method_errors = {}
    if "loto" in protocol["methods"]:
        try:
            scores = loto_scores(turns, detector_score)
            method_scores["loto"] = [scores[w["turn_id"]] for w in units]
        except CoverageError as exc:
            method_errors["loto"] = str(exc)
    if llm_prediction and "error" not in llm_prediction:
        method_scores["llm"] = project_tokens(units, llm_prediction["token_scores"])
    for fraction in protocol["budgets"]:
        matched = select(units, gl_scores, fraction, eligible)
        for method in protocol["methods"]:
            repeats = protocol["random_repeats"] if method in {"random", "span_random"} else 1
            for repeat in range(repeats):
                key = {"method": method, "fraction": fraction, "repeat": repeat}
                if method in method_errors:
                    yield {**key, "error": "method_context_coverage", "detail": method_errors[method]}
                    continue
                if not eligible or (method == "llm" and "llm" not in method_scores):
                    yield {**key, "error": "no_eligible_words" if not eligible else "invalid_llm_attribution"}
                    continue
                # Stable per-conversation random draw, independent of file ordering.
                seed = int(digest({"seed": protocol["seed"], "turns": turns, **key})[:16], 16)
                selected = select(units, method_scores.get(method, gl_scores), fraction, eligible,
                    method=method if method in {"random", "span_random"} else "rank", seed=seed, matched=matched)
                changed = edit(turns, units, selected, protocol["operation"])
                complement = [i for i in eligible if i not in selected]
                kept = edit(turns, units, complement, protocol["operation"])
                # LOTO has only turn resolution. Its word projection is constant
                # within a turn and deterministic ties must not imply token skill.
                turn_ranking = {}
                if method == "loto":
                    turn_ranking = {str(k): v for k, v in scores.items()}
                elif method == "last_user":
                    turn_ranking = {str(t["turn_id"]): float(t["turn_id"] == last) for t in turns if t["role"] == "user"}
                elif method == "llm":
                    turn_ranking = llm_prediction["turn_scores"]
                elif method == "surface":
                    turn_ranking = {str(t["turn_id"]): max((surface[i] for i, w in enumerate(units) if w["turn_id"] == t["turn_id"]), default=0.) for t in turns if t["role"] == "user"}
                yield {**key, "eligible_n": len(eligible), "seed": seed, "edited": changed, "kept": kept, "turn_ranking": turn_ranking,
                       "audit": audit_edit(turns, units, selected, changed)}


def run(records, detector, guards, protocol, lexicon, store, llm=None, shard_index=None, shard_count=None):
    sharded = shard_index is not None
    if sharded and (type(shard_index) is not int or type(shard_count) is not int or
                    shard_count < 1 or not 0 <= shard_index < shard_count):
        raise ValueError("invalid shard index/count")
    predictions, interventions = [], []
    for index, r in enumerate(records):
        if sharded and index % shard_count != shard_index:
            continue
        artifact = store.root / "shards" / f"record-{index:06d}.json" if sharded else None
        if artifact is not None and artifact.exists():
            completed = load_shard_record(artifact, store.manifest_id, index, r["id"])
            predictions.append(completed["prediction"])
            interventions.extend(completed["interventions"])
            print(f"[{index + 1}/{len(records)}] {r['id']} cached record", flush=True)
            continue
        start = len(interventions)
        turns = visible(r, protocol["view"])
        prediction = infer(store, "detector", detector, turns)
        row = {"id": r["id"], "cluster_id": r["cluster_id"], "label": r["label"], "dataset": r["dataset"],
               "strata": r.get("strata", {}), "prediction": prediction, "visible_sha256": digest(turns)}
        llm_pred = infer(store, "llm", llm, turns) if llm else None
        if llm_pred:
            row["llm_prediction"] = llm_pred
        if "error" not in prediction:
            if protocol.get("context_diagnostics", False):
                users = [t for t in turns if t["role"] == "user"]
                row["context_diagnostics"] = {"last_user_only": infer(store, "detector", detector, users[-1:]),
                                               "without_assistant_history": infer(store, "detector", detector, users)}
            row["localization"] = localization(r, prediction, protocol["turn_threshold"])
            row["span_agreement"] = span_agreement(r, prediction, protocol["span_threshold"])
            def detector_score(ts):
                value = infer(store, "detector", detector, ts)
                if "error" in value:
                    raise CoverageError(value["detail"])
                return value["probability"]
            before = {name: infer(store, name, backend, turns) for name, backend in guards.items()}
            row["guard_originals"] = before
            try:
                plans = list(intervention_plan(turns, prediction, protocol, lexicon, detector_score, llm_pred)) if guards else []
            except CoverageError as e:
                # A LOTO request can exceed coverage only for malformed/empty context.
                # Fail instead of silently dropping all other methods for this record.
                raise CoverageError(f"{r['id']}: LOTO cannot cover this context: {e}") from e
            for plan in plans:
                if plan.get("turn_ranking") and plan["fraction"] == protocol["budgets"][0]:
                    row.setdefault("baseline_localization", {})[plan["method"]] = localization(r, {"turn_scores": plan["turn_ranking"]}, threshold=None)
                common = {"id": r["id"], "cluster_id": r["cluster_id"], "label": r["label"],
                    "detector_probability": prediction["probability"], **plan}
                if "error" in plan:
                    for name in guards:
                        interventions.append({**common, "guard": name, "before": None, "after": None})
                    continue
                for name, backend in guards.items():
                    changed = infer(store, name, backend, plan["edited"])
                    kept = infer(store, name, backend, plan["kept"])
                    interventions.append({**common, "guard": name, "before": before[name].get("probability"),
                        "after": changed.get("probability"), "kept": kept.get("probability"),
                        "original_result": before[name], "edited_result": changed, "kept_result": kept})
        predictions.append(row)
        if sharded:
            result = {"prediction": row, "interventions": interventions[start:]}
            write_json(artifact, {"manifest_id": store.manifest_id, "index": index, "id": r["id"],
                                  "result": result, "result_sha256": digest(result)})
        else:
            write_json(store.root / "progress.json", {"completed": index + 1, "total": len(records), "last_id": r["id"]})
        print(f"[{index + 1}/{len(records)}] {r['id']}", flush=True)
    if sharded:
        print(f"shard {shard_index}/{shard_count} complete: {len(predictions)} records", flush=True)
        return None
    write_json(store.root / "predictions.json", predictions)
    write_json(store.root / "interventions.json", interventions)
    report = summarize(records, predictions, interventions, protocol, detector.threshold, list(guards))
    report["manifest_id"] = store.manifest_id
    write_json(store.root / "report.json", report)
    return report


def load_shard_record(path, manifest_id, index, record_id):
    row = json.loads(path.read_text(encoding="utf-8"))
    result = row.get("result", {})
    if (row.get("manifest_id") != manifest_id or row.get("index") != index or
            row.get("id") != record_id or digest(result) != row.get("result_sha256") or
            result.get("prediction", {}).get("id") != record_id or
            not isinstance(result.get("interventions"), list) or
            any(x.get("id") != record_id for x in result["interventions"])):
        raise ValueError(f"invalid shard record: {path}")
    return result


def finalize_shards(records, store, shard_count):
    """Assemble the full cohort in source order and compute statistics only once."""
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("invalid shard count")
    directory = store.root / "shards"
    expected = {f"record-{i:06d}.json" for i in range(len(records))}
    actual = {p.name for p in directory.glob("record-*.json")}
    if actual != expected:
        raise ValueError(f"shards incomplete: {len(expected - actual)} missing, {len(actual - expected)} unexpected")
    predictions, interventions = [], []
    for index, record in enumerate(records):
        row = load_shard_record(directory / f"record-{index:06d}.json", store.manifest_id, index, record["id"])
        predictions.append(row["prediction"])
        interventions.extend(row["interventions"])
    protocol = store.manifest["protocol"]
    write_json(store.root / "predictions.json", predictions)
    write_json(store.root / "interventions.json", interventions)
    report = summarize(records, predictions, interventions, protocol,
                       store.manifest["detector"]["threshold"], list(store.manifest["guards"]))
    report["manifest_id"] = store.manifest_id
    write_json(store.root / "report.json", report)
    return report


def summarize(records, predictions, interventions, protocol, detection_threshold, guard_names=None):
    valid = [r for r in predictions if "error" not in r["prediction"]]
    valid_by_id = {r["id"]: r for r in valid}
    ci = lambda rows, stat: cluster_interval(rows, stat, protocol["bootstrap_repeats"], protocol["seed"])
    def detect(rows):
        return binary([r["label"] for r in rows], [r["prediction"]["probability"] for r in rows], detection_threshold)
    detection = detect(valid)
    positives = sum(r["label"] for r in records)
    negatives = len(records) - positives
    missing_pos = positives - detection["positive_n"]
    missing_neg = negatives - detection["negative_n"]
    detection["recall_full_cohort_bounds"] = [(detection["tp"] + m) / positives for m in (0, missing_pos)] if positives else None
    detection["fpr_full_cohort_bounds"] = [(detection["fp"] + m) / negatives for m in (0, missing_neg)] if negatives else None
    detection["uncertainty"] = {k: ci(valid, lambda rows, k=k: detect(rows)[k]) for k in ("recall", "fpr", "f1", "ap", "auroc")}
    strata = {}
    for key in ("dataset", "family", "supervision_tier", "corpus_source", "difficulty", "pivot_kind", "transfer_tier"):
        attempted_groups = defaultdict(list)
        for r in records:
            attempted_groups[r["dataset"] if key == "dataset" else r.get("strata", {}).get(key, "unknown")].append(r)
        strata[key] = {}
        for name, attempted in attempted_groups.items():
            scored = [valid_by_id[r["id"]] for r in attempted if r["id"] in valid_by_id]
            metrics = detect(scored)
            attempted_pos = sum(r["label"] for r in attempted)
            attempted_neg = len(attempted) - attempted_pos
            missing_pos = attempted_pos - metrics["positive_n"]
            missing_neg = attempted_neg - metrics["negative_n"]
            metrics["recall_full_cohort_bounds"] = [(metrics["tp"] + m) / attempted_pos for m in (0, missing_pos)] if attempted_pos else None
            metrics["fpr_full_cohort_bounds"] = [(metrics["fp"] + m) / attempted_neg for m in (0, missing_neg)] if attempted_neg else None
            strata[key][name] = {"coverage": {"attempted_n": len(attempted), "scored_n": len(scored), "missing_n": len(attempted) - len(scored)},
                                 "detection": metrics, "localization": summarize_localization(scored)}
    output = {"task": records[0]["label_semantics"], "view": protocol["view"],
              "coverage": {"total": len(records), "scored": len(valid), "failures": [{"id": r["id"], **r["prediction"]} for r in predictions if "error" in r["prediction"]]},
              "detection": detection, "localization": summarize_localization(valid), "strata": strata,
              "transfer_claim": "independent guard judgment sensitivity, not target attack-success reduction",
              "effects": {}, "paired_differences": {}, "utility": {}}
    output["baseline_turn_agreement"] = {}
    for name in sorted({name for r in valid for name in r.get("baseline_localization", {})}):
        metrics = [r["baseline_localization"][name] for r in valid if name in r.get("baseline_localization", {})]
        output["baseline_turn_agreement"][name] = {"scored_n": len(metrics), "topk": {k: {metric: average([x["known_positive_hits"][k][metric] for x in metrics]) for metric in ("hit", "coverage", "chance_hit", "chance_adjusted_hit")} for k in ("1", "2", "3", "5")}}
    output["guard_detection"] = {}
    output["context_diagnostics"] = {}
    for name in ("last_user_only", "without_assistant_history"):
        rows = [r for r in valid if "probability" in r.get("context_diagnostics", {}).get(name, {})]
        if rows:
            output["context_diagnostics"][name] = {"scored_n": len(rows),
                **binary([r["label"] for r in rows], [r["context_diagnostics"][name]["probability"] for r in rows], detection_threshold),
                "interpretation": "inference input intervention, not a retrained architecture ablation"}
    for name in sorted({name for r in valid for name in r.get("guard_originals", {})}):
        guard_rows = [r for r in valid if "probability" in r.get("guard_originals", {}).get(name, {})]
        threshold = detection_threshold if name == "self" else protocol["guard_threshold"]
        output["guard_detection"][name] = {"attempted_n": len(records), "scored_n": len(guard_rows),
            **binary([r["label"] for r in guard_rows], [r["guard_originals"][name]["probability"] for r in guard_rows], threshold)}
    if any("llm_prediction" in r for r in predictions):
        llm_rows = [r for r in predictions if "probability" in r.get("llm_prediction", {})]
        output["llm_detection"] = {"attempted_n": len(records), "parsed_n": len(llm_rows),
            **binary([r["label"] for r in llm_rows], [r["llm_prediction"]["probability"] for r in llm_rows], .5)}
    # Average repeated interventions inside the conversation before bootstrap.
    grouped = defaultdict(list)
    for r in interventions:
        grouped[(r["guard"], r["method"], r["fraction"], r["id"])].append(r)
    collapsed = defaultdict(list)
    for name in guard_names or []:
        for method in protocol["methods"]:
            for fraction in protocol["budgets"]:
                collapsed[(name, method, fraction)] = []
    for (guard, method, fraction, _rid), rs in grouped.items():
        threshold = detection_threshold if guard == "self" else protocol["guard_threshold"]
        complete = all(r.get("before") is not None and r.get("after") is not None for r in rs)
        row = {k: rs[0][k] for k in ("id", "cluster_id", "label", "detector_probability")}
        row.update({k: average([r.get(k) for r in rs]) if complete else None for k in ("before", "after", "kept")})
        if complete:
            row["flip"] = average([float(r["before"] >= threshold and r["after"] < threshold) for r in rs])
            row["gated_flip"] = row["flip"] * (row["detector_probability"] >= detection_threshold)
        collapsed[(guard, method, fraction)].append(row)
    for (guard, method, fraction), rs in collapsed.items():
        # Predictions that failed were still part of the requested population.
        seen = {r["id"] for r in rs}
        rs.extend({"id": r["id"], "cluster_id": r["cluster_id"], "label": r["label"], "before": None, "after": None}
                  for r in records if r["id"] not in seen)
        key = f"{guard}/{method}/{fraction:g}"
        threshold = detection_threshold if guard == "self" else protocol["guard_threshold"]
        output["effects"][key] = effects(rs, threshold, detection_threshold, protocol["bootstrap_repeats"], protocol["seed"])
        for comparator in ("random", "span_random", "surface", "last_user", "loto", "llm"):
            if method != "guardlens" or (guard, comparator, fraction) not in collapsed:
                continue
            other = {r["id"]: r for r in collapsed[(guard, comparator, fraction)]}
            paired = [{"cluster_id": r["cluster_id"], "delta": (r["before"] - r["after"]) - (other[r["id"]]["before"] - other[r["id"]]["after"])}
                      for r in rs if r["label"] == 1 and r.get("after") is not None and r["id"] in other and other[r["id"]].get("after") is not None]
            output["paired_differences"][f"{key}-minus-{comparator}"] = ci(paired, lambda rows: average([r["delta"] for r in rows]))
    # Utility must name the benign population. Shared detector FPR is the same
    # for every attribution method, so this cannot establish attribution specificity.
    benign_groups = {
        "all_benign": [r for r in valid if r["label"] == 0],
        "hard_benign": [r for r in valid if r["label"] == 0 and r["strata"].get("difficulty") == "hard"],
        "frontier_authored_benign": [r for r in valid if r["label"] == 0 and r["strata"].get("family") == "frontier_authored_benign"],
        "interactive_benign": [r for r in valid if r["label"] == 0 and r["strata"].get("family") == "interactive_benign"],
    }
    for population, benign in benign_groups.items():
        fpr = detect(benign)["fpr"]
        population_entry = {"benign_ids": [r["id"] for r in benign], "n": len(benign), "fpr": fpr,
            "mode": "shared_detector", "attribution_specificity_claim": False, "grid": {}, "break_even_vs_surface": {}}
        for key, effect in output["effects"].items():
            drop = effect["cohorts"]["all_positive_complete"]["drop"]["estimate"]
            population_entry["grid"][key] = utility_grid(drop, fpr, protocol["utility_lambdas"])
        output["utility"][population] = population_entry
    # Minimal among the predeclared budgets only, never a claim of true minimality.
    minimum = {}
    for (guard, method, fraction), rs in collapsed.items():
        for r in rs:
            if r["label"] == 1 and r.get("flip") == 1:
                key = f"{guard}/{method}/{r['id']}"
                minimum[key] = min(minimum.get(key, fraction), fraction)
    output["smallest_tested_budget_with_all_repeats_flipped"] = minimum
    return output


def summarize_localization(rows):
    locs = [r["localization"] for r in rows if "localization" in r]
    result = {"records_with_assessed_turns": sum(x["assessed_n"] > 0 for x in locs),
              "records_with_positive_evidence": sum(x["positive_n"] > 0 for x in locs),
              "known_positive_hits": {}}
    result["turn_macro_ap_assessed_mixed_class"] = average([x["assessed_binary"]["ap"] for x in locs])
    for k in ("1", "2", "3", "5"):
        result["known_positive_hits"][k] = {name: average([x["known_positive_hits"][k][name] for x in locs])
            for name in ("hit", "coverage", "chance_hit", "chance_adjusted_hit", "observed_precision_lower_bound")}
    for title, metric in (("turn", "assessed_binary"), ("span_character", None)):
        items = [x[metric] for x in locs] if metric else [r["span_agreement"] for r in rows if "span_agreement" in r]
        counts = {k: sum(x[k] for x in items) for k in ("tp", "fp", "tn", "fn", "n")}
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        counts.update({"precision": tp / (tp + fp) if tp + fp else None,
                       "recall": tp / (tp + fn) if tp + fn else None,
                       "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None})
        result[title] = counts
    return result
