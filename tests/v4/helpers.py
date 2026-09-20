import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from jevbench_v4.contracts import DecisionRequest, digest
from jevbench_v4.data import configuration, dataset_folder, freeze_splits
from jevbench_v4.storage import source_fingerprint, write_json


def request(choices=2):
    return DecisionRequest("The payment was refunded yesterday.", "Choose the action.",
                           tuple((f"C{i}", f"Action {i}") for i in range(choices)))


def body(req, choice="C0"):
    return {"model": "fake-model", "answers": {"classification": {
        "choice": choice, "confidence": -1234,
        "probabilities": {key: 1.0 if key == choice else 0.0 for key, _ in req.choices}}}}


class FakeProvider:
    def __init__(self, name="fake"):
        self.identity = {"name": name, "revision": "test-only"}
        self.calls = 0

    def audit(self, req):
        return {"choice_count": len(req.choices), "truncated": False}

    def predict(self, req, audit):
        self.calls += 1
        return body(req)


def prepared_fixture(root, dataset="IMDb", seeds=(2027, 2028)):
    root = Path(root)
    config = configuration()
    config.update(datasets=[dataset], seeds=list(seeds), train_cap=80, validation_cap=30,
                  policy_cap=30, test_cap=40, banking_test_cap=40)
    if dataset == "IMDb":
        frame = pd.DataFrame({"text": [f"{'good' if i % 2 else 'bad'} story sample{i}" for i in range(240)],
                              "label": np.arange(240) % 2,
                              "_official_split": ["train"] * 180 + ["test"] * 60})
        kind, features = "text", ["text"]
    else:
        frame = pd.DataFrame({"balance": np.arange(240, dtype=float), "job": ["a", "b"] * 120,
                              "label": np.arange(240) % 2})
        kind, features = "tabular", ["balance", "job"]
    folder = dataset_folder(root, dataset)
    folder.mkdir(parents=True)
    snapshot = folder / "snapshot.parquet"
    frame.to_parquet(snapshot, index=False)
    metadata = {"name": dataset, "kind": kind, "features": features, "labels": ["no", "yes"],
                "task": "Classify the synthetic development fixture.", "source": "test-fixture-only",
                "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    write_json(folder / "metadata.json", metadata)
    source = source_fingerprint()
    write_json(root / "run.json", {"config": config, "source_fingerprint": source,
                                   "run_id": digest({"config": config, "source": source})})
    freeze_splits(root, frame, metadata, config)
    return frame, metadata, config
