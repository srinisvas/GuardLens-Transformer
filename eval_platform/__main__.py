"""Run python -m eval_platform --help. Preparation/reporting never load a model."""
import argparse
import csv
import json
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path

from .adapters import adapt, check_overlap, validate_collection
from .contract import RunStore, canonical, digest, file_hash, read_jsonl, validate_protocol, write_json
from .runtime import GuardLensBackend, HFChat, ShieldGemmaBackend, LLMAttributor


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.glob("eval_platform/*.py"), *root.glob("guardlens/**/*.py")])
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unavailable"
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"git_sha": sha, "python_source_sha256": digest({str(p.relative_to(root)): file_hash(p) for p in files}),
            "python": platform.python_version(), "packages": versions}


def prepare(args):
    path = Path(args.input)
    if path.suffix == ".csv":
        if not args.messages_column:
            raise ValueError("CSV requires --messages-column containing a JSON message array")
        with path.open(newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
        for r in raw:
            r["turns"] = json.loads(r[args.messages_column])
    else:
        raw = list(read_jsonl(path))
    rows = validate_collection([adapt(r, args.source, args.source_revision, args.split, args.subset) for r in raw])
    output = Path(args.output)
    if output.exists():
        raise ValueError("prepared output already exists, choose a fresh path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")
    write_json(str(output) + ".manifest.json", {"input_sha256": file_hash(path), "output_sha256": file_hash(output),
        "source": args.source, "revision": args.source_revision, "split": args.split, "subset": args.subset,
        "raw_n": len(raw), "converted_n": len(rows), "rejected_n": 0, "code": source_identity()})
    print(canonical({"records": len(rows), "sha256": file_hash(output), "output": str(output)}))


def load_dataset(path, expected):
    if file_hash(path) != expected:
        raise ValueError("dataset bytes differ from the frozen SHA256")
    return validate_collection(list(read_jsonl(path)))


def evaluate(args):
    from .runner import run
    protocol = validate_protocol(load(args.protocol))
    records = load_dataset(args.data, args.data_sha256)
    if any(r["split"] in {"train", "dev", "valid", "validation"} for r in records):
        raise ValueError("evaluation input must be an explicitly held-out cohort")
    exclusions, exclusion_hashes, raw_hashes = [], {}, set()
    for path in args.exclude:
        exclusions.extend(list(read_jsonl(path)))
        exclusion_hashes[str(Path(path).resolve())] = file_hash(path)
        sidecar = load(path + ".manifest.json")
        if sidecar["output_sha256"] != file_hash(path):
            raise ValueError("exclusion data changed after preparation")
        raw_hashes.add(sidecar["input_sha256"])
    check_overlap(records, exclusions)
    lexicon = load(args.lexicon)
    if not isinstance(lexicon, list) or not lexicon or not all(isinstance(x, str) and x for x in lexicon):
        raise ValueError("lexicon must be a frozen nonempty string list")
    detector = GuardLensBackend(args.checkpoint, args.device)
    if not set(detector.identity["training_data_sha256"].values()) <= raw_hashes:
        raise ValueError("--exclude must include prepared copies of this checkpoint's exact train AND dev files")
    guards = {"self": detector} if args.guard in {"self", "both"} else {}
    if args.guard in {"shield", "both"}:
        if not args.shield_config:
            raise ValueError("--shield-config required")
        spec = load(args.shield_config)
        guards["shield"] = ShieldGemmaBackend(HFChat(**spec["model"], device=args.device), spec["policies"], protocol["guard_threshold"])
    llm = None
    if "llm" in protocol["methods"]:
        if not args.llm_config:
            raise ValueError("LLM method requires --llm-config")
        llm = LLMAttributor(HFChat(**load(args.llm_config), device=args.device))
    manifest = {"dataset_sha256": args.data_sha256, "protocol": protocol, "lexicon_sha256": file_hash(args.lexicon),
        "detector": detector.identity, "guards": {k: v.identity for k, v in guards.items()},
        "llm": llm.identity if llm else None, "exclusion_hashes": exclusion_hashes, "code": source_identity()}
    store = RunStore(args.output, manifest)
    report = run(records, detector, guards, protocol, lexicon, store, llm)
    print(canonical({"report": str(store.root / "report.json"), "coverage": report["coverage"], "detection": report["detection"]}))


def replay(args):
    from .replay import run_replay, BEHAVIOR_RUBRIC
    source = Path(args.run)
    manifest = load(source / "manifest.json")
    records = load_dataset(args.data, manifest["dataset_sha256"])
    target_spec, judge_spec = load(args.target_config), load(args.judge_config)
    if (target_spec["model"], target_spec["revision"]) == (judge_spec["model"], judge_spec["revision"]):
        raise ValueError("independent behavior judge must differ from the target")
    target, judge = HFChat(**target_spec, device=args.device), HFChat(**judge_spec, device=args.device)
    store = RunStore(args.output, {"stage": "paired_scripted_replay", "parent_manifest_id": manifest["manifest_id"],
        "dataset_sha256": manifest["dataset_sha256"], "interventions_sha256": file_hash(source / "interventions.json"),
        "target": target.identity, "judge": judge.identity, "judge_rubric": BEHAVIOR_RUBRIC,
        "seeds": args.seeds, "code": source_identity()})
    run_replay(records, load(source / "interventions.json"), target, judge, args.seeds, store,
               manifest["detector"]["threshold"], manifest["protocol"]["bootstrap_repeats"], manifest["protocol"])
    print(store.root / "replay_report.json")


def shield_config(args):
    from huggingface_hub import HfApi, hf_hub_download
    from .runtime import pinned
    revision = args.revision or HfApi().model_info(args.model).sha
    pinned(revision)
    card_path = hf_hub_download(args.model, "README.md", revision=revision)
    card = Path(card_path).read_text()
    table = card.split("Use Case 1: Prompt-only Content Classification", 1)[1].split("Use Case 2:", 1)[0]
    labels = {"Dangerous Content": "dangerous", "Harassment": "harassment", "Hate Speech": "hate", "Sexually Explicit Information": "sexual"}
    policies = {}
    for title, key in labels.items():
        match = re.search(r"\|\s*" + re.escape(title) + r"\s*\|\s*`([^`]+)`", table, re.S)
        if match is None:
            raise ValueError(f"pinned model card policy format changed: {title}")
        policies[key] = match.group(1).strip()
    write_json(args.output, {"model": {"model": args.model, "revision": revision, "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": 1, "temperature": 0}, "policies": policies, "model_card_sha256": file_hash(card_path)})
    print(args.output)


def task_export(a):
    from .studies import human_tasks
    write_json(a.output, human_tasks(validate_collection(list(read_jsonl(a.data)))))


def human(a):
    from .studies import human_report
    manifest = load(Path(a.run) / "manifest.json")
    records = load_dataset(a.data, manifest["dataset_sha256"])
    write_json(a.output, {"parent_manifest_id": manifest["manifest_id"], "annotations_sha256": file_hash(a.annotations),
        **human_report(records, list(read_jsonl(a.annotations)), load(Path(a.run) / "predictions.json"), manifest["detector"]["threshold"])})


def robustness(a):
    from .studies import robustness_report
    left, right = Path(a.original_run), Path(a.variant_run)
    ma, mb = load(left / "manifest.json"), load(right / "manifest.json")
    if ma["detector"] != mb["detector"] or ma["protocol"] != mb["protocol"]:
        raise ValueError("robustness requires identical checkpoint and protocol")
    write_json(a.output, {"parent_manifests": [ma["manifest_id"], mb["manifest_id"]], "pairs_sha256": file_hash(a.pairs),
        **robustness_report(load(left / "predictions.json"), load(right / "predictions.json"), list(read_jsonl(a.pairs)))})


def compare(a):
    from .studies import compare_runs
    runs = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        if name in runs:
            raise ValueError("duplicate run name")
        runs[name] = {"manifest": load(Path(path) / "manifest.json"), "report": load(Path(path) / "report.json")}
    write_json(a.output, compare_runs(runs))


def utility(a):
    from .studies import method_utility
    write_json(a.output, {"rows_sha256": file_hash(a.rows), **method_utility(list(read_jsonl(a.rows)))})


def parser():
    p = argparse.ArgumentParser(description="GuardLens reproducible evaluation platform")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare", help="strict source conversion")
    for name in ("input", "output", "source-revision", "split"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--source", choices=["internal", "mhj", "mtid", "canonical"], required=True)
    q.add_argument("--subset", choices=["harmful", "benign"])
    q.add_argument("--messages-column")
    q.set_defaults(func=prepare)
    q = sub.add_parser("run", help="detection, localization, and text interventions")
    for name in ("data", "data-sha256", "checkpoint", "output"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--protocol", default="configs/eval_protocol.json")
    q.add_argument("--lexicon", default="configs/eval_surface_lexicon.json")
    q.add_argument("--exclude", nargs="+", required=True, help="canonical train + dev with preparation sidecars")
    q.add_argument("--guard", choices=["none", "self", "shield", "both"], default="both")
    q.add_argument("--shield-config")
    q.add_argument("--llm-config")
    q.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    q.set_defaults(func=evaluate)
    q = sub.add_parser("replay", help="paired fresh generations and independent behavior judging")
    for name in ("run", "data", "target-config", "judge-config", "output"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    q.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    q.set_defaults(func=replay)
    q = sub.add_parser("shield-config", help="freeze official model revision and prompt policies")
    q.add_argument("--model", default="google/shieldgemma-9b")
    q.add_argument("--revision")
    q.add_argument("--max-input-tokens", type=int, default=8191)
    q.add_argument("--output", required=True)
    q.set_defaults(func=shield_config)
    for name, func, arguments in [("human-tasks", task_export, ["data", "output"]),
            ("human-report", human, ["data", "annotations", "run", "output"]),
            ("robustness-report", robustness, ["original-run", "variant-run", "pairs", "output"]),
            ("utility-report", utility, ["rows", "output"])]:
        q = sub.add_parser(name)
        for arg in arguments:
            q.add_argument("--" + arg, required=True)
        q.set_defaults(func=func)
    q = sub.add_parser("compare-runs", help="compare ablations/sensitivity/seeds under one protocol")
    q.add_argument("--runs", nargs="+", required=True, help="name=directory")
    q.add_argument("--output", required=True)
    q.set_defaults(func=compare)
    return p


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
