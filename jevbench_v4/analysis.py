"""Paired Jev-versus-comparator tables reconstructed from saved predictions."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .storage import read_json


def _model(summary):
    return summary.get("baseline") or summary["provider"]["name"]


def _interval(values, seed=20260922, draws=4000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return map(float, np.quantile(samples, [.025, .975]))


def build_comparisons(root, summaries):
    root = Path(root)
    grouped = defaultdict(dict)
    for path in root.rglob("summary.json"):
        summary = read_json(path)
        if summary.get("partition") != "test" or summary.get("dataset") == "Iterative Support":
            continue
        if not path.with_name("predictions.jsonl").exists():
            continue
        records = [json.loads(line) for line in path.with_name("predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        grouped[(summary["dataset"], summary["seed"])][_model(summary)] = records
    rows = []
    for (dataset, seed), models in sorted(grouped.items()):
        if "jev" not in models:
            continue
        reference = {record["case_id"]: record for record in models["jev"]}
        for model, records in sorted(models.items()):
            if model == "jev":
                continue
            candidate = {record["case_id"]: record for record in records}
            if set(candidate) != set(reference):
                raise ValueError(f"Unpaired comparison for {dataset}/{seed}/{model}")
            differences = []
            clusters = defaultdict(list)
            for case_id, jev in reference.items():
                other = candidate[case_id]
                difference = float(jev["prediction"] == jev["label"]) - float(other["prediction"] == other["label"])
                key = jev.get("pair_id") or (jev.get("timestamp", "")[:10] if dataset == "Bike Demand" else case_id)
                clusters[key].append(difference)
            cluster_values = [np.mean(value) for value in clusters.values()]
            low, high = _interval(cluster_values)
            rows.append({"dataset": dataset, "seed": seed, "comparator": model,
                         "n_cases": len(reference), "n_resampling_clusters": len(cluster_values),
                         "jev_minus_comparator_accuracy": float(np.mean(cluster_values)),
                         "ci95_low": low, "ci95_high": high,
                         "resampling_unit": "policy pair" if dataset == "Support Policy" else
                         "calendar day block" if dataset == "Bike Demand" else "case"})
    destination = root / "comparisons_vs_jev.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        columns = ("dataset", "seed", "comparator", "n_cases", "n_resampling_clusters",
                   "jev_minus_comparator_accuracy", "ci95_low", "ci95_high", "resampling_unit")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_iterative_comparisons(root):
    root = Path(root)
    grouped = defaultdict(dict)
    for path in root.glob("providers/*/Iterative_Support/*/test/summary.json"):
        summary = read_json(path)
        records = [json.loads(line) for line in path.with_name("episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        grouped[summary["condition"]][summary["provider"]["name"]] = records
    rows = []
    for condition, models in sorted(grouped.items()):
        if "jev" not in models:
            continue
        reference = {record["episode_id"]: record for record in models["jev"]}
        for model, records in sorted(models.items()):
            if model == "jev":
                continue
            candidate = {record["episode_id"]: record for record in records}
            if set(candidate) != set(reference):
                raise ValueError(f"Unpaired iterative comparison for {condition}/{model}")
            success = [float(reference[key]["success"]) - float(candidate[key]["success"]) for key in reference]
            utility = [reference[key]["net_utility"] - candidate[key]["net_utility"] for key in reference]
            success_low, success_high = _interval(success)
            utility_low, utility_high = _interval(utility)
            rows.append({"condition": condition, "comparator": model, "n_episodes": len(reference),
                         "jev_minus_comparator_success": float(np.mean(success)),
                         "success_ci95_low": success_low, "success_ci95_high": success_high,
                         "jev_minus_comparator_utility": float(np.mean(utility)),
                         "utility_ci95_low": utility_low, "utility_ci95_high": utility_high})
    destination = root / "iterative_comparisons_vs_jev.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        columns = ("condition", "comparator", "n_episodes", "jev_minus_comparator_success",
                   "success_ci95_low", "success_ci95_high", "jev_minus_comparator_utility",
                   "utility_ci95_low", "utility_ci95_high")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows
