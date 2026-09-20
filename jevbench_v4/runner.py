"""Resumable continuity evaluations and a development-only compatibility pilot."""
from pathlib import Path

from .client import DecisionClient
from .contracts import canonical, digest
from .data import load_job, request_for
from .metrics import summarize
from .storage import environment, freeze, read_json, write_json


def run_provider(root, provider, dataset, seed, partition="train", limit=50,
                 max_attempts=50000, max_seconds=7200, min_interval=.15):
    if partition not in ("train", "policy", "test"):
        raise ValueError("partition must be train, policy or test")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if partition != "train" and limit is not None:
        raise ValueError("Policy/test evaluation must use the complete frozen partition")
    root = Path(root)
    run = read_json(root / "run.json")
    frame, metadata, split = load_job(root, dataset, seed)
    selected = split["partitions"][partition]
    rows = selected["indices"][:limit] if limit else selected["indices"]
    case_ids = selected["case_ids"][:len(rows)]
    provider_root = root / "providers" / digest(provider.identity)
    job = provider_root / dataset.replace(" ", "_") / str(seed) / partition
    freeze(job / "job.json", {"run_id": run["run_id"], "provider": provider.identity,
          "environment": environment(), "dataset": dataset, "seed": seed, "partition": partition,
          "case_ids": case_ids, "snapshot_sha256": metadata["sha256"], "limit": limit})
    client = DecisionClient(provider, provider_root, run["run_id"], max_attempts=max_attempts,
                            max_seconds=max_seconds, min_interval=min_interval)
    records = []
    for row, case_id in zip(rows, case_ids):
        request = request_for(frame, row, metadata)
        path = job / "predictions" / (case_id + ".json")
        request_hash = digest(request.as_dict())
        if path.exists():
            record = read_json(path)
            if record["case_id"] != case_id or record["input_hash"] != request_hash or record["label"] != int(frame.iloc[row].label):
                raise ValueError("Saved prediction does not match the frozen case")
        else:
            result = client.call(request, context={"dataset": dataset, "seed": seed, "partition": partition})
            record = {**result, "case_id": case_id, "input_hash": request_hash,
                      "label": int(frame.iloc[row].label)}
            write_json(path, record)
        records.append(record)
    summary = {"dataset": dataset, "seed": seed, "partition": partition, "provider": provider.identity,
               "panel": "raw", "status": "development-pilot" if partition == "train" else "continuity-evaluation",
               **summarize(records, len(metadata["labels"]))}
    write_json(job / "summary.json", summary)
    # Portable per-case export with exact order/IDs for later paired comparisons.
    (job / "predictions.jsonl").write_text("".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    print(f"{provider.identity['name']} / {dataset} / {partition}: {summary['statuses']}", flush=True)
    return summary
