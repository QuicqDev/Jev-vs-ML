from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from jevbench_v4.data import configuration, prepare
from jevbench_v4.export import collect_summaries
from jevbench_v4.policy import DATASET
from jevbench_v4.runner import run_provider
from jevbench_v4.storage import read_json, write_json
from jevbench_v4.workers import merge_runs
from scripts.run_v4 import main
from tests.v4.helpers import FakeProvider


class WorkerTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[2] / "generated"
        directory.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "study"
        with patch("jevbench_v4.data.prepare_data", side_effect=AssertionError("No old datasets")):
            prepare(self.root)

    def test_default_cli_prepares_only_new_cases_and_keeps_frozen_splits(self):
        with patch("jevbench_v4.data.prepare_data", side_effect=AssertionError("No old datasets")):
            main(["prepare", "--root", str(self.root)])
            main(["verify", "--root", str(self.root)])
        self.assertEqual(read_json(self.root / "run.json")["config"]["datasets"], [DATASET])
        with self.assertRaises(ValueError):
            configuration("continuity")
        self.assertEqual(configuration("continuity", suite="continuity")["datasets"],
                         ["IMDb", "Banking77", "Bank Marketing"])

    def test_two_workers_merge_without_rerunning_and_reject_conflicts(self):
        worker = self.root.with_name("second")
        shutil.copytree(self.root, worker)
        jev, local = FakeProvider("fake-jev"), FakeProvider("fake-local")
        run_provider(self.root, jev, DATASET, 2027, limit=4, min_interval=0)
        run_provider(worker, local, DATASET, 2027, limit=4, min_interval=0)
        execution = worker / "execution/local"
        write_json(execution / "plan.json", {"test_gpu_assignment": [0, 1]})
        (execution / "von.log").write_text("test worker log", encoding="utf-8")
        destination = self.root.with_name("combined")
        archive = merge_runs(destination, [self.root, worker])
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as bundle:
            self.assertIn("execution/local/plan.json", bundle.namelist())
            self.assertEqual(bundle.read("execution/local/von.log"), b"test worker log")
        self.assertEqual(len(collect_summaries(destination)), 2)
        self.assertEqual((jev.calls, local.calls), (4, 4))
        with self.assertRaises(ValueError):
            merge_runs(destination, [self.root, worker])
        run = read_json(worker / "run.json")
        run["run_id"] = "another-study"
        write_json(worker / "run.json", run)
        rejected = self.root.with_name("rejected")
        with self.assertRaises(ValueError):
            merge_runs(rejected, [self.root, worker])
        self.assertFalse(rejected.exists())

    def test_merge_rejects_conflicting_provider_artifacts_before_writing(self):
        run_provider(self.root, FakeProvider(), DATASET, 2027, limit=4, min_interval=0)
        worker = self.root.with_name("duplicate-worker")
        shutil.copytree(self.root, worker)
        manifest = next((worker / "providers").rglob("provider.json"))
        identity = read_json(manifest)
        identity["different_worker_setting"] = True
        write_json(manifest, identity)
        destination = self.root.with_name("conflicting-merge")
        with self.assertRaises(ValueError):
            merge_runs(destination, [self.root, worker])
        self.assertFalse(destination.exists())
