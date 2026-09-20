"""Recompute raw summaries from saved predictions before creating a portable ZIP."""
import csv
import json
from pathlib import Path
import zipfile

from .contracts import canonical, digest
from .data import case_context, load_job
from .metrics import summarize
from .storage import read_json, write_json


def collect_summaries(root):
    root = Path(root).resolve()
    run = read_json(root / "run.json")
    summaries = []
    for path in sorted(root.rglob("summary.json")):
        saved = read_json(path)
        frame, metadata, split = load_job(root, saved["dataset"], saved["seed"])
        records = [json.loads(line) for line in path.with_name("predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        job = read_json(path.with_name("job.json"))
        if job["run_id"] != run["run_id"] or job["snapshot_sha256"] != metadata["sha256"]:
            raise ValueError("Job belongs to another study or snapshot")
        selected = split["partitions"][saved["partition"]]
        expected = selected["case_ids"]
        if "provider" in job:
            if any(job[key] != saved[key] for key in ("provider", "dataset", "seed", "partition")):
                raise ValueError("Provider summary does not match its job")
            limit = job["limit"]
            if saved["partition"] != "train" and limit is not None:
                raise ValueError("Policy/test jobs must include the entire partition")
            expected = expected[:limit] if limit is not None else expected
            if job["case_ids"] != expected:
                raise ValueError("Job case identities do not match the frozen partition")
        elif job["baseline"] != saved["baseline"] or job["split"] != digest(split) or saved["partition"] != "test":
            raise ValueError("Baseline summary does not match its job")
        if [r["case_id"] for r in records] != expected:
            raise ValueError("Exported prediction IDs/order do not match the job")
        indices = selected["indices"][:len(expected)]
        if [r["label"] for r in records] != frame.iloc[indices].label.tolist():
            raise ValueError("Exported labels do not match the frozen snapshot")
        for record, row in zip(records, indices):
            if any(record.get(key) != value for key, value in case_context(frame, row).items()):
                raise ValueError("Exported pair metadata does not match the frozen snapshot")
        recomputed = summarize(records, len(metadata["labels"]))
        if any(canonical(saved[key]) != canonical(value) for key, value in recomputed.items()):
            raise ValueError(f"Saved summary does not reproduce from predictions: {path}")
        summaries.append(saved)
    if not summaries:
        raise ValueError("No completed evaluations to export")
    return summaries


def export_run(root):
    root = Path(root).resolve()
    summaries = collect_summaries(root)
    write_json(root / "summaries.json", summaries)
    columns = ("dataset", "suite", "status", "label_status", "seed", "partition", "model", "panel", "n", "accuracy", "balanced_accuracy",
               "n_pairs", "both_members_correct",
               "macro_f1", "probability_coverage", "brier", "log_loss", "failed", "invalid", "unsupported")
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary.get(key) for key in columns}
            row["model"] = summary.get("baseline") or summary["provider"]["name"]
            row.update({key: summary["statuses"][key] for key in ("failed", "invalid", "unsupported")})
            writer.writerow(row)
    destination = root.with_name(root.name + "_results.zip")
    source_root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            gpu_log = path.suffix == ".log" and path.relative_to(root).parts[0] == "execution"
            if path.is_file() and (path.suffix in (".json", ".jsonl", ".parquet", ".csv") or gpu_log) and not path.name.startswith("."):
                archive.write(path, path.relative_to(root))
        for directory in ("jevbench", "jevbench_v4"):
            for path in sorted((source_root / directory).glob("*.py")):
                archive.write(path, "source/" + path.relative_to(source_root).as_posix())
        for name in ("requirements.txt", "requirements-v4-local.txt"):
            archive.write(source_root / name, "source/" + name)
        for name in ("scripts/__init__.py", "scripts/run_v4.py"):
            archive.write(source_root / name, "source/" + name)
        archive.writestr("README.md", "# V4 run artifacts\n\n"
            "Tables are reconstructed from per-case predictions, with failures retained.\n"
            "From source/, install requirements.txt and run:\n\n"
            "    python -m scripts.run_v4 export --root ..\n\n"
            "V4 adds new tasks; V3 scores remain historical context. Draft synthetic labels require review.\n"
            "Optional continuity runs require --suite continuity when using the CLI.\n")
    return destination
