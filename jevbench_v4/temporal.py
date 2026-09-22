"""Leakage-resistant Bike Sharing future-prediction windows for V4."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

from .contracts import canonical, digest
from .storage import freeze

DATASET = "Bike Demand"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def _download(root: Path) -> tuple[pd.DataFrame, str]:
    source = root / "sources" / "uci_bike_sharing.zip"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        response = requests.get(SOURCE_URL, timeout=(15, 120))
        response.raise_for_status()
        source.write_bytes(response.content)
    payload = source.read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        frame = pd.read_csv(archive.open("hour.csv"))
    return frame, hashlib.sha256(payload).hexdigest()


def build_examples(raw: pd.DataFrame) -> pd.DataFrame:
    """Use only information available at hour t to predict demand at t+1."""
    required = {"dteday", "hr", "cnt", "season", "yr", "mnth", "holiday", "weekday",
                "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Bike data is missing columns: {sorted(missing)}")
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw.dteday) + pd.to_timedelta(raw.hr, unit="h")
    if raw.timestamp.duplicated().any():
        raise ValueError("Bike timestamps must be unique")
    raw = raw.sort_values("timestamp").set_index("timestamp", drop=False)
    lookup = set(raw.index)
    rows = []
    for timestamp, current in raw.iterrows():
        timestamp = pd.Timestamp(timestamp)
        wanted = (timestamp - pd.Timedelta(hours=1), timestamp - pd.Timedelta(hours=24),
                  timestamp - pd.Timedelta(hours=168), timestamp + pd.Timedelta(hours=1))
        if any(value not in lookup for value in wanted):
            continue
        previous, day, week, target = (raw.loc[value] for value in wanted)
        features = {
            "timestamp": timestamp.isoformat(), "target_timestamp": wanted[-1].isoformat(),
            "hour_next": int(target.hr), "weekday_next": int(target.weekday),
            "month_next": int(target.mnth), "year_next": int(target.yr),
            "holiday_next": int(target.holiday), "workingday_next": int(target.workingday),
            "season_next": int(target.season), "weather_now": int(current.weathersit),
            "temp_now": float(current.temp), "feels_like_now": float(current.atemp),
            "humidity_now": float(current.hum), "windspeed_now": float(current.windspeed),
            "count_now": int(current.cnt), "count_lag_1": int(previous.cnt),
            "count_lag_24": int(day.cnt), "count_lag_168": int(week.cnt),
        }
        rows.append({**features, "target_count": int(target.cnt)})
    result = pd.DataFrame(rows)
    if result.empty or not result.timestamp.is_monotonic_increasing:
        raise ValueError("No ordered, exact-hour future examples were constructed")
    return result


def _future_windows(frame: pd.DataFrame, config: dict) -> dict[int, dict[str, list[int]]]:
    test_n, validation_n, policy_n = (config[key] for key in
                                      ("temporal_test", "temporal_validation", "temporal_policy"))
    train_cap, gap, purge = config["temporal_train_cap"], config["temporal_gap_hours"], 1
    minimum = train_cap + validation_n + policy_n + gap + 3 * purge
    latest = len(frame) - test_n
    windows = config["temporal_windows"]
    if latest < minimum + (windows - 1) * test_n:
        raise ValueError("Bike snapshot is too small for the requested forward windows")
    starts = np.linspace(minimum, latest, windows, dtype=int).tolist()
    if any(b - a < test_n for a, b in zip(starts, starts[1:])):
        raise ValueError("Forward test windows overlap")
    result = {}
    for number, start in enumerate(starts):
        policy_end = start - gap - purge
        policy_start = policy_end - policy_n
        validation_end = policy_start - purge
        validation_start = validation_end - validation_n
        train_end = validation_start - purge
        train_start = max(0, train_end - train_cap)
        parts = {"train": list(range(train_start, train_end)),
                 "validation": list(range(validation_start, validation_end)),
                 "policy": list(range(policy_start, policy_end)),
                 "test": list(range(start, start + test_n))}
        if (pd.Timestamp(frame.iloc[start].timestamp)
                - pd.Timestamp(frame.iloc[policy_end - 1].target_timestamp) < pd.Timedelta(hours=gap)):
            raise ValueError("The forward test embargo is shorter than configured")
        result[number] = parts
    return result


def _random_diagnostic(frame: pd.DataFrame, config: dict, labels: np.ndarray) -> dict[str, list[int]]:
    """A deliberately random split, reported only beside the forward-window result."""
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(frame))
    train, test = train_test_split(indices, test_size=config["temporal_test"], random_state=20260922,
                                   stratify=labels)
    train, policy = train_test_split(train, test_size=config["temporal_policy"], random_state=20260923,
                                     stratify=labels[train])
    train, validation = train_test_split(train, test_size=config["temporal_validation"], random_state=20260924,
                                         stratify=labels[train])
    if len(train) > config["temporal_train_cap"]:
        train, _ = train_test_split(train, train_size=config["temporal_train_cap"], random_state=20260925,
                                    stratify=labels[train])
    return {"train": sorted(map(int, train)), "validation": sorted(map(int, validation)),
            "policy": sorted(map(int, policy)), "test": sorted(map(int, test))}


def prepare(root: Path, config: dict) -> tuple[pd.DataFrame, dict, dict[int, dict]]:
    raw, archive_sha = _download(root)
    frame = build_examples(raw)
    windows = _future_windows(frame, config)
    reference = windows[0]["train"]
    threshold = int(np.quantile(frame.iloc[reference].target_count, .75, method="higher"))
    frame["label"] = (frame.target_count > threshold).astype(int)
    labels = frame.label.to_numpy()
    windows[config["temporal_random_seed"]] = _random_diagnostic(frame, config, labels)
    feature_columns = [column for column in frame if column not in
                       {"label", "target_count", "timestamp", "target_timestamp", "text"}]
    frame["text"] = frame.apply(lambda row: canonical(json.loads(row[feature_columns].to_json())), axis=1)
    frame["timestamp"] = frame.timestamp.astype(str)
    frame["target_timestamp"] = frame.target_timestamp.astype(str)
    metadata = {"name": DATASET, "kind": "tabular", "features": feature_columns,
                "labels": [f"Demand at t+1 is at most {threshold}", f"Demand at t+1 exceeds {threshold}"],
                "task": "Predict the next hour's total bike demand using only the supplied present and past facts.",
                "source": SOURCE_URL, "source_archive_sha256": archive_sha, "threshold": threshold,
                "threshold_rule": "75th percentile of forward window 0 training targets; strictly greater is positive",
                "suite": "future", "label_status": "public-dataset-derived",
                "excluded_columns": ["casual", "registered", "future weather", "future count"]}
    for seed, parts in windows.items():
        for name, rows in parts.items():
            if frame.iloc[rows].label.nunique() != 2:
                raise ValueError(f"Bike {seed}/{name} does not contain both classes")
    return frame, metadata, windows


def freeze_windows(root: Path, frame: pd.DataFrame, metadata: dict, windows: dict[int, dict]) -> None:
    folder = root / "data" / DATASET.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / "snapshot.parquet"
    if snapshot.exists() and not pd.read_parquet(snapshot).equals(frame):
        raise ValueError("Existing Bike snapshot differs from the downloaded source")
    if not snapshot.exists():
        frame.to_parquet(snapshot, index=False)
    metadata["sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    freeze(folder / "metadata.json", metadata)
    ids = [digest({"dataset": DATASET,
                   "state": canonical(json.loads(frame.iloc[i][metadata["features"]].to_json()))})
           for i in range(len(frame))]
    hashes = {}
    for seed, parts in windows.items():
        protocol = "random-holdout-diagnostic" if seed < 0 else "expanding-forward-window"
        manifest = {"dataset": DATASET, "seed": seed, "protocol": protocol,
                    "snapshot_sha256": metadata["sha256"], "partitions": {},
                    "relationship_to_v3": "additional-future-prediction-task"}
        for name, rows in parts.items():
            manifest["partitions"][name] = {"indices": rows, "case_ids": [ids[i] for i in rows],
                "class_counts": {str(k): int(v) for k, v in frame.iloc[rows].label.value_counts().items()}}
        freeze(folder / f"split_{seed}.json", manifest)
        hashes[str(seed)] = digest(manifest)
    freeze(folder / "manifest.json", {"metadata_hash": digest(metadata), "splits": hashes,
                                      "representation": "full-input; no future leakage"})
