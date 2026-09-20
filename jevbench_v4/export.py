"""Recompute raw summaries from saved predictions before creating a portable ZIP."""
import csv
import json
from pathlib import Path
import zipfile

from .contracts import canonical
from .data import load_job
from .metrics import summarize
from .storage import read_json, write_json


def export_run(root):
    root = Path(root).resolve()
    summaries = []
    for path in sorted(root.rglob("summary.json")):
        saved = read_json(path)
        frame, metadata, split = load_job(root, saved["dataset"], saved["seed"])
        records = [json.loads(line) for line in path.with_name("predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        job = read_json(path.with_name("job.json"))
        expected = job.get("case_ids", split["partitions"][saved["partition"]]["case_ids"])
        if [r["case_id"] for r in records] != expected:
            raise ValueError("Exported prediction IDs/order do not match the job")
        indices = split["partitions"][saved["partition"]]["indices"][:len(expected)]
        if [r["label"] for r in records] != frame.iloc[indices].label.tolist():
            raise ValueError("Exported labels do not match the frozen snapshot")
        recomputed = summarize(records, len(metadata["labels"]))
        if any(canonical(saved[key]) != canonical(value) for key, value in recomputed.items()):
            raise ValueError(f"Saved summary does not reproduce from predictions: {path}")
        summaries.append(saved)
    if not summaries:
        raise ValueError("No completed evaluations to export")
    write_json(root / "summaries.json", summaries)
    columns = ("dataset", "seed", "partition", "model", "panel", "n", "accuracy", "balanced_accuracy",
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
            if path.is_file() and path.suffix in (".json", ".jsonl", ".parquet", ".csv") and not path.name.startswith("."):
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
            "The manifest distinguishes development pilots from continuity test evaluations.\n")
    return destination
