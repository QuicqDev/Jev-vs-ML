import tempfile
from pathlib import Path
import unittest
import json
import zipfile

from jevbench_v4.baselines import run_baseline
from jevbench_v4.data import dataset_folder, freeze_splits, load_job, request_for
from jevbench_v4.runner import run_provider
from jevbench_v4.export import export_run
from jevbench_v4.storage import read_json, write_json
from tests.v4.helpers import FakeProvider, prepared_fixture


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[2] / "generated"
        directory.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame, self.metadata, self.config = prepared_fixture(self.root)

    def test_shared_holdout_disjoint_partitions_and_official_boundaries(self):
        _, _, first = load_job(self.root, "IMDb", 2027)
        _, _, second = load_job(self.root, "IMDb", 2028)
        self.assertEqual(first["partitions"]["test"], second["partitions"]["test"])
        groups = [set(p["case_ids"]) for p in first["partitions"].values()]
        self.assertEqual(sum(map(len, groups)), len(set.union(*groups)))
        for partition, data in first["partitions"].items():
            expected = "test" if partition == "test" else "train"
            self.assertTrue(self.frame.iloc[data["indices"]]["_official_split"].eq(expected).all())

    def test_long_input_is_preserved_and_gold_never_enters_request(self):
        frame = self.frame.copy()
        frame.loc[0, "text"] = "x" * 9000
        req = request_for(frame, 0, self.metadata)
        self.assertEqual(len(req.state), 9000)
        self.assertNotIn("label", req.as_dict())
        self.assertNotIn("_official_split", req.state)

    def test_snapshot_and_split_tampering_are_detected(self):
        folder = dataset_folder(self.root, "IMDb")
        split = read_json(folder / "split_2027.json")
        split["partitions"]["train"]["indices"][0] = split["partitions"]["test"]["indices"][0]
        write_json(folder / "split_2027.json", split)
        with self.assertRaises(ValueError):
            load_job(self.root, "IMDb", 2027)
        (folder / "snapshot.parquet").write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            load_job(self.root, "IMDb", 2028)

    def test_pilot_is_development_only_and_resume_does_not_call_again(self):
        provider = FakeProvider()
        summary = run_provider(self.root, provider, "IMDb", 2027, limit=5, min_interval=0)
        self.assertEqual(summary["status"], "development-pilot")
        self.assertEqual(summary["partition"], "train")
        self.assertEqual(provider.calls, 5)
        self.assertEqual(run_provider(self.root, provider, "IMDb", 2027, limit=5, min_interval=0), summary)
        self.assertEqual(provider.calls, 5)
        with self.assertRaises(ValueError):
            run_provider(self.root, provider, "IMDb", 2027, partition="test", limit=5)

    def test_real_text_baseline_and_mock_provider_share_test_cases(self):
        baseline = run_baseline(self.root, "IMDb", 2027, "svm", threads=1)
        provider = run_provider(self.root, FakeProvider(), "IMDb", 2027, "test", None, min_interval=0)
        self.assertEqual(baseline["n"], provider["n"])
        self.assertEqual(baseline["policy_labels_used"], 0)
        self.assertIsNone(baseline["brier"])
        self.assertEqual(len(read_json(self.root / "baselines/svm/IMDb/2027/trials.json")), 4)

    def test_tabular_duration_is_rejected_and_rbf_fit_works(self):
        folder = self.root / "tabular"
        frame, metadata, config = prepared_fixture(folder, "Bank Marketing", (2027,))
        metadata["features"].append("duration")
        with self.assertRaises(ValueError):
            freeze_splits(folder, frame, metadata, config)
        result = run_baseline(folder, "Bank Marketing", 2027, "svm", threads=1)
        self.assertEqual(result["n"], 40)

    def test_real_catboost_and_majority_baselines(self):
        folder = self.root / "tabular"
        prepared_fixture(folder, "Bank Marketing", (2027,))
        for name in ("majority", "catboost"):
            summary = run_baseline(folder, "Bank Marketing", 2027, name, threads=1)
            self.assertEqual(summary["n"], 40)
            self.assertEqual(summary["probability_coverage"], 1)
            self.assertEqual(summary["policy_labels_used"], 0)

    def test_export_recomputes_predictions_and_detects_changed_scores(self):
        run_provider(self.root, FakeProvider(), "IMDb", 2027, limit=5, min_interval=0)
        archive = export_run(self.root)
        self.addCleanup(archive.unlink, missing_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertIn("summary.csv", bundle.namelist())
            self.assertIn("source/jevbench_v4/runner.py", bundle.namelist())
            self.assertFalse(any(".env" in name for name in bundle.namelist()))
        summary_path = next(self.root.rglob("summary.json"))
        summary = read_json(summary_path)
        summary["accuracy"] = 123
        write_json(summary_path, summary)
        with self.assertRaises(ValueError):
            export_run(self.root)
