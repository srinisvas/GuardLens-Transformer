"""Scientific-contract regressions and end-to-end CPU platform tests."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eval_platform.adapters import adapt, check_overlap, validate_collection
from eval_platform.contract import RunStore, canonical, visible, validate_protocol
from eval_platform.interventions import words, select, edit, risk_scores, project_tokens, eligible_indices, loto_scores
from eval_platform.metrics import binary, cluster_interval, effects, localization, utility_grid, break_even
from eval_platform.replay import replay_suffix, judge_behavior, run_replay
from eval_platform.runner import run, intervention_plan
from eval_platform.runtime import CoverageError
from eval_platform.studies import human_report, robustness_report, method_utility, spearman


def record(cid="a", label=1):
    return {"id": cid, "cluster_id": cid, "dataset": "fixture", "split": "test", "source_revision": "fixed",
        "label": label, "label_semantics": "attack_intent", "strata": {"family": "hard_benign", "supervision_tier": "cf_strong"},
        "turns": [{"turn_id": 0, "role": "user", "text": "alpha café beta"},
                  {"turn_id": 1, "role": "assistant", "text": "OLD response"},
                  {"turn_id": 2, "role": "user", "text": "gamma delta"},
                  {"turn_id": 3, "role": "assistant", "text": "OLD final"}],
        "gold": {}, "objective": "Complete the example task."}


def protocol():
    value = json.loads(Path("configs/eval_protocol.json").read_text(encoding="utf-8"))
    value.update(budgets=[.2, .4], random_repeats=3, bootstrap_repeats=100)
    return value


class FakeDetector:
    threshold = .5
    identity = {"fixture": True}

    def __init__(self):
        self.calls = []

    def predict(self, turns):
        self.calls.append(copy.deepcopy(turns))
        if any("OVERFLOW" in t["text"] for t in turns):
            raise CoverageError("too long")
        p = .8 if any("alpha" in t["text"] for t in turns if t["role"] == "user") else .2
        return {"probability": p, "turn_scores": {str(t["turn_id"]): p for t in turns if t["role"] == "user"},
            "token_scores": [{**{k: w[k] for k in ("turn_id", "start", "end")}, "score": .9 if w["text"] == "alpha" else .1} for w in words(turns)]}

    def score(self, turns):
        return self.predict(turns)["probability"]


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, messages, seed):
        self.calls.append((copy.deepcopy(messages), seed))
        return {"text": f"NEW-{seed}", "hit_output_limit": False, "generated_tokens": 2}


class FakeJudge:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def generate(self, messages, seed):
        self.calls.append((messages, seed))
        return {"text": "not JSON" if self.invalid else canonical({"unsafe": False, "refused": False, "helpfulness": .9, "confidence": .9, "rationale": "fixture"}), "hit_output_limit": False}


class ContractTests(unittest.TestCase):
    def test_model_view_never_contains_metadata_or_final_response(self):
        r = record()
        r["hidden_rationale"] = "secret"
        r["turns"][0]["hidden_rationale"] = "secret"
        result = visible(r, "pre_response")
        self.assertEqual(len(result), 3)
        self.assertNotIn("secret", canonical(result))
        self.assertEqual(len(visible(r, "retrospective")), 4)

    def test_mtid_benign_success_is_not_an_attack(self):
        raw = {"conversation": [{"role": "user", "content": "hello", "hidden_rationale": "secret"}],
               "is_jailbreak_success": True, "sample_index": 8, "rollout_id": 1, "meta_intent": "goal"}
        a = adapt(raw, "mtid", "frozen", "test", "benign")
        b = adapt({**raw, "rollout_id": 2}, "mtid", "frozen", "test", "benign")
        self.assertEqual(a["label"], 0)
        self.assertEqual(a["cluster_id"], b["cluster_id"])
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(a["gold"], {})
        self.assertNotIn("secret", canonical(visible(a, "pre_response")))

    def test_mhj_stable_id_and_no_fake_evidence(self):
        raw = {"turns": [{"role": "human", "content": "hello"}], "pivot_turn_id": 0}
        a = adapt(raw, "mhj", "frozen", "test")
        self.assertEqual(a, adapt(raw, "mhj", "frozen", "test"))
        self.assertEqual(a["gold"], {})
        self.assertEqual(a["label"], 1)
        with self.assertRaises(ValueError):
            adapt({"prompt": "flat"}, "mhj", "frozen", "test")
        with self.assertRaises(ValueError):
            adapt({"turns": [{"role": "unknown", "content": "x"}]}, "mhj", "frozen", "test")

    def test_overlap_and_duplicate_detection(self):
        a, b = record(), record("b")
        with self.assertRaises(ValueError):
            check_overlap([a], [b])
        with self.assertRaises(ValueError):
            validate_collection([a, a])
        with self.assertRaises(ValueError):
            validate_collection([a, {**b, "label_semantics": "unsafe_trajectory"}])

    def test_manifest_and_cache_refuse_drift(self):
        with tempfile.TemporaryDirectory() as d:
            s = RunStore(d, {"checkpoint": "one"})
            self.assertEqual(s.cached("x", {"input": 1}, lambda: {"p": .5}), {"p": .5})
            self.assertEqual(s.cached("x", {"input": 1}, lambda: self.fail()), {"p": .5})
            with self.assertRaises(ValueError):
                RunStore(d, {"checkpoint": "two"})
            path = next((Path(d) / "cache" / "x").glob("*.json"))
            row = json.loads(path.read_text(encoding="utf-8"))
            row["result"]["p"] = .8
            path.write_text(canonical(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                s.cached("x", {"input": 1}, lambda: None)

    def test_protocol_rejects_external_tuning(self):
        with self.assertRaises(ValueError):
            validate_protocol({**protocol(), "threshold_source": "external_dev"})

    def test_protocol_rejects_typos_and_boolean_numeric_values(self):
        with self.assertRaisesRegex(ValueError, "unknown protocol fields"):
            validate_protocol({**protocol(), "context_diagnostic": True})
        with self.assertRaisesRegex(ValueError, "budgets"):
            validate_protocol({**protocol(), "budgets": [True]})
        with self.assertRaisesRegex(ValueError, "utility penalties"):
            validate_protocol({**protocol(), "utility_lambdas": [False]})


class InterventionTests(unittest.TestCase):
    def test_exact_ties_unicode_and_assistant_preservation(self):
        ts = record()["turns"]
        units = words(ts)
        chosen = select(units, [1.] * len(units), .4, list(range(len(units))))
        self.assertEqual(chosen, [0, 1])
        changed = edit(ts, units, chosen)
        self.assertEqual(changed[0]["text"], "  beta")
        self.assertEqual(changed[1], ts[1])
        self.assertEqual(changed[3], ts[3])
        self.assertEqual(units[1]["text"], "café")
        self.assertEqual(project_tokens(units, [{"turn_id": 0, "start": 6, "end": 8, "score": .8}])[1], .8)

    def test_context_scope_keeps_final_request_in_both_interventions(self):
        ts = visible(record(), "pre_response")
        p = protocol()
        p.update(scope="context_only", methods=["guardlens"])
        plans = list(intervention_plan(ts, FakeDetector().predict(ts), p, []))
        for plan in plans:
            self.assertEqual(plan["edited"][-1], ts[-1])
            self.assertEqual(plan["kept"][-1], ts[-1])

    def test_matched_random_preserves_turn_counts_and_budget(self):
        units = words(record()["turns"])
        for seed in range(15):
            chosen = select(units, [0.] * 5, .6, list(range(5)), "span_random", seed, matched=[0, 1, 4])
            self.assertEqual(len(chosen), 3)
            self.assertEqual(sum(units[i]["turn_id"] == 0 for i in chosen), 2)
            first = [i for i in chosen if units[i]["turn_id"] == 0]
            self.assertEqual(first[1] - first[0], 1)

    def test_non_surface_masks_all_methods_and_empty_scope_is_explicit(self):
        ts = record()["turns"]
        units = words(ts)
        scores = risk_scores(units, ["alpha"])
        self.assertNotIn(0, eligible_indices(units, ts, "non_surface", scores))
        p = protocol()
        p.update(scope="non_surface", methods=["guardlens"])
        plans = list(intervention_plan(ts, FakeDetector().predict(ts), p, [w["text"] for w in units]))
        self.assertTrue(all(x["error"] == "no_eligible_words" for x in plans))

    def test_loto_reinfers_text_and_retains_negative_drops(self):
        ts = record()["turns"]
        calls = []
        def score(turns):
            calls.append(turns)
            return .9 if not any(t["turn_id"] == 0 for t in turns) else .6
        value = loto_scores(ts, score)
        self.assertAlmostEqual(value[0], -.3)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(t["turn_id"] != 0 for t in calls[1]))


class MetricTests(unittest.TestCase):
    def test_single_class_mhj_does_not_report_f1_or_ap(self):
        r = binary([1, 1, 1], [.9, .2, .8], .5)
        self.assertAlmostEqual(r["recall"], 2 / 3)
        for k in ("f1", "ap", "auroc", "fpr", "precision"):
            self.assertIsNone(r[k])

    def test_tie_aware_metrics_match_sklearn(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        labels, ps = [1, 0, 1, 0, 1], [.4, .4, .9, .1, .4]
        result = binary(labels, ps, .5)
        self.assertAlmostEqual(result["ap"], average_precision_score(labels, ps))
        self.assertAlmostEqual(result["auroc"], roc_auc_score(labels, ps))

    def test_cluster_bootstrap_not_rollout_bootstrap(self):
        rows = [{"cluster_id": "one", "v": 1}] * 20
        result = cluster_interval(rows, lambda rs: sum(r["v"] for r in rs) / len(rs))
        self.assertEqual(result["clusters"], 1)
        self.assertIsNone(result["ci95"])

    def test_missing_effects_have_bounds_not_safe_defaults(self):
        rows = [{"cluster_id": "a", "label": 1, "detector_probability": .2, "before": .8, "after": .1},
                {"cluster_id": "b", "label": 1, "before": None, "after": None}]
        report = effects(rows, .5, .5, 100)
        self.assertEqual(report["end_to_end_flip"]["missing_bounds"], [0., .5])
        self.assertIsNone(report["end_to_end_flip"]["estimate"])
        self.assertIsNone(report["cohorts"]["detector_detected_complete"]["drop"]["estimate"])

    def test_negative_effect_is_not_clipped(self):
        r = effects([{"cluster_id": "a", "label": 1, "before": .2, "after": .8, "detector_probability": .9}], .5, .5, 100)
        self.assertAlmostEqual(r["cohorts"]["all_positive_complete"]["drop"]["estimate"], -.6)

    def test_chance_adjustment_and_unknown_turns(self):
        r = record()
        r["gold"] = {"turns": {"0": 1}, "provenance": "human"}
        p = FakeDetector().predict(r["turns"])
        m = localization(r, p)
        self.assertEqual(m["unknown_n"], 1)
        self.assertEqual(m["known_positive_hits"]["1"]["chance_hit"], .5)
        self.assertIsNone(m["known_positive_hits"]["5"]["chance_adjusted_hit"])

    def test_utility_requires_same_named_cohort(self):
        self.assertEqual(utility_grid(.4, .1, [2])[0]["utility"], .2)
        self.assertAlmostEqual(break_even(.6, .2, .5, .1), 1.)
        rows = [{"id": "a", "cluster_id": "a", "method": "one", "label": 1, "drop": .2, "population_id": "p", "detector_policy_id": "dev"},
                {"id": "b", "cluster_id": "b", "method": "two", "label": 1, "drop": .2, "population_id": "p", "detector_policy_id": "dev"}]
        with self.assertRaises(ValueError):
            method_utility(rows)


class ReplayTests(unittest.TestCase):
    def test_missing_attribution_remains_in_replay_denominator(self):
        p = protocol()
        p.update(methods=["guardlens"], budgets=[.2])
        with tempfile.TemporaryDirectory() as d:
            report = run_replay([record()], [], FakeGenerator(), FakeJudge(), [42, 43, 44], RunStore(d, {"test": True}), .5, 100, p)
            row = report["methods"]["guardlens/0.2"]
            self.assertEqual(row["positive_attempted_n"], 1)
            self.assertEqual(row["positive_complete_n"], 0)
            self.assertEqual(row["end_to_end_rescue_full_cohort"]["missing_bounds"], [0., 1.])
            self.assertIsNone(row["end_to_end_rescue_full_cohort"]["estimate"])

    def test_replay_rng_is_paired_within_record_and_isolated_across_records(self):
        a, b = record("a"), record("b")
        plans = []
        for r in (a, b):
            units = words(r["turns"])
            from eval_platform.interventions import audit_edit
            changed = edit(r["turns"], units, [0])
            plans.append({"id": r["id"], "cluster_id": r["cluster_id"], "label": 1,
                "method": "guardlens", "fraction": .2, "repeat": 0,
                "detector_probability": .8, "edited": changed,
                "audit": audit_edit(r["turns"], units, [0], changed)})
        with tempfile.TemporaryDirectory() as d:
            run_replay([a, b], plans, FakeGenerator(), FakeJudge(), [42, 43, 44],
                       RunStore(d, {"test": True}), .5, 100)
            rows = json.loads((Path(d) / "replay_rows.json").read_text(encoding="utf-8"))
        streams = {(r["id"], r["seed"]): r["conversation_seed"] for r in rows}
        self.assertNotEqual(streams[("a", 42)], streams[("b", 42)])
        self.assertEqual(len(streams), len(rows))

    def test_suffix_has_no_stale_assistant_and_seeds_are_paired(self):
        ts = record()["turns"]
        generator = FakeGenerator()
        a = replay_suffix(ts, 0, generator, 42)
        b = replay_suffix(edit(ts, words(ts), [0]), 0, generator, 42)
        self.assertNotIn("OLD", canonical(a))
        self.assertNotIn("OLD", canonical(b))
        self.assertEqual([s for _, s in generator.calls[:2]], [s for _, s in generator.calls[2:]])
        self.assertEqual(a["history"][-2]["content"], "gamma delta")

    def test_invalid_judgment_is_missing_not_safe(self):
        result = judge_behavior("original goal", {"generated": []}, FakeJudge(True), 42)
        self.assertEqual(result["error"], "invalid_behavior_judgment")
        self.assertNotIn("unsafe", result)

    def test_replay_preserves_objective_and_common_fork(self):
        r = record()
        plans = []
        units = words(r["turns"])
        for method, index in (("guardlens", 0), ("random", 4)):
            from eval_platform.interventions import audit_edit
            changed = edit(r["turns"], units, [index])
            plans.append({"id": r["id"], "cluster_id": r["cluster_id"], "label": 1, "method": method, "fraction": .2,
                "repeat": 0, "detector_probability": .8, "edited": changed, "audit": audit_edit(r["turns"], units, [index], changed)})
        judge = FakeJudge()
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(d, {"test": True})
            report = run_replay([r], plans, FakeGenerator(), judge, [42, 43, 44], store, .5, 100)
            rows = json.loads((Path(d) / "replay_rows.json").read_text(encoding="utf-8"))
            self.assertEqual({row["fork"] for row in rows}, {0})
            self.assertTrue(all(json.loads(messages[1]["content"])["original_objective"] == r["objective"] for messages, _ in judge.calls))
            self.assertIsNone(report["methods"]["guardlens/0.2"]["rescue_given_fresh_unsafe"]["estimate"])


class IntegrationTests(unittest.TestCase):
    def test_all_overflow_still_reports_each_effect_denominator(self):
        r = record()
        r["turns"][0]["text"] = "OVERFLOW"
        with tempfile.TemporaryDirectory() as d:
            report = run([r], FakeDetector(), {"self": FakeDetector()}, protocol(), ["alpha"], RunStore(d, {"test": True}))
            self.assertEqual(report["effects"]["self/guardlens/0.2"]["positive_attempted_n"], 1)
            self.assertEqual(report["effects"]["self/guardlens/0.2"]["missing_n"], 1)
            self.assertEqual(report["detection"]["recall_full_cohort_bounds"], [0., 1.])
            group = report["strata"]["dataset"]["fixture"]
            self.assertEqual(group["coverage"], {"attempted_n": 1, "scored_n": 0, "missing_n": 1})
            self.assertEqual(group["detection"]["recall_full_cohort_bounds"], [0., 1.])

    def test_full_run_preserves_misses_failures_and_resume(self):
        a, b, c = record(), record("b", 0), record("c")
        b["turns"][0]["text"] = "safe café beta"
        c["turns"][0]["text"] = "OVERFLOW"
        detector = FakeDetector()
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(d, {"test": True})
            report = run([a, b, c], detector, {"self": detector}, protocol(), ["alpha"], store)
            self.assertEqual(report["coverage"]["scored"], 2)
            self.assertEqual(report["detection"]["recall_full_cohort_bounds"], [.5, 1.])
            effect = report["effects"]["self/guardlens/0.2"]
            self.assertEqual(effect["positive_attempted_n"], 2)
            self.assertEqual(effect["missing_n"], 1)
            calls = len(detector.calls)
            self.assertEqual(report, run([a, b, c], detector, {"self": detector}, protocol(), ["alpha"], store))
            self.assertEqual(len(detector.calls), calls)
            self.assertTrue((Path(d) / "report.json").exists())

    def test_prepare_cli_is_strict_and_does_not_invent_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / "source.jsonl", Path(d) / "out.jsonl"
            source.write_text(canonical({"conversation_id": "a", "turns": [{"role": "human", "content": "hello"}]}) + "\n", encoding="utf-8")
            command = [sys.executable, "-m", "eval_platform", "prepare", "--source", "mhj", "--source-revision", "fixed", "--split", "test", "--input", str(source), "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["gold"], {})
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_human_agreement_and_robustness_require_real_alignment(self):
        r = record()
        pred = {"id": r["id"], "cluster_id": r["cluster_id"], "prediction": FakeDetector().predict(r["turns"])}
        ann = [{"id": r["id"], "annotator_id": who, "label": 1, "evidence_turn_ids": [0], "spans": []} for who in ("A", "B")]
        result = human_report([r], ann, [pred], .5)
        self.assertEqual(result["pairwise"]["A/B"]["turn_jaccard_nonempty_union"], 1.)
        pair = {"original_id": r["id"], "variant_id": r["id"], "turn_mapping": {"0": 0, "2": 2}, "kind": "paraphrase"}
        with self.assertRaises(ValueError):
            robustness_report([pred], [pred], [pair])
        self.assertIsNone(spearman([1, 1], [2, 2]))


if __name__ == "__main__":
    unittest.main()
