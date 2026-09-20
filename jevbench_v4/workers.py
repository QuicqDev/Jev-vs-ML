"""Merge outputs from independent workers without rerunning their models."""
import hashlib
from pathlib import Path
import shutil

from .data import load_job
from .export import collect_summaries, export_run
from .storage import read_json


def merge_runs(destination, inputs):
    """Combine independent workers only when their complete frozen study matches."""
    destination = Path(destination).resolve()
    roots = [Path(root).resolve() for root in inputs]
    if len(set(roots)) < 2:
        raise ValueError("Provide at least two distinct run directories")
    if destination.exists():
        raise ValueError("Choose a new merge directory; existing results are never overwritten")
    reference = read_json(roots[0] / "run.json")
    files = {}
    data_inventory = None
    for root in roots:
        if read_json(root / "run.json") != reference:
            raise ValueError("Notebook outputs belong to different frozen studies")
        collect_summaries(root)
        for dataset in reference["config"]["datasets"]:
            for seed in reference["config"]["seeds"]:
                load_job(root, dataset, seed)
        inventory = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or (relative.parts[0] not in ("data", "providers", "baselines")
                                     and relative.as_posix() != "run.json"):
                continue
            if path.suffix not in (".json", ".jsonl", ".parquet", ".csv") or path.name.startswith("."):
                continue
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if relative.parts[0] == "data":
                inventory[relative.as_posix()] = checksum
            if relative in files and files[relative][1] != checksum:
                raise ValueError(f"Conflicting worker output: {relative}")
            files[relative] = (path, checksum)
        if data_inventory is not None and inventory != data_inventory:
            raise ValueError("Worker data snapshots or manifests differ")
        data_inventory = inventory
    # All inputs are checked before creating any output files.
    for relative, (source, _) in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return export_run(destination)
