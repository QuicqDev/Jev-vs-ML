import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jevbench_v4.baselines import run_baseline
from jevbench_v4.data import configuration, load_job, prepare, request_for
from jevbench_v4.export import collect_summaries
from jevbench_v4.policy import DATASET, generate, refund_label
from jevbench_v4.runner import run_provider
from jevbench_v4.storage import read_json, write_json
from tests.v4.helpers import FakeProvider


class PolicyTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[2] / "generated"
        directory.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "study"
        with patch("jevbench_v4.data.prepare_data", side_effect=AssertionError("No old datasets")):
            prepare(self.root)

    def test_oracle_cancellation_precedence_exception_and_day_boundary(self):
        base = {"age_days": 30, "cancel_before_dispatch": False, "defective": False,
                "unopened": True, "final_sale": False}
        self.assertEqual(refund_label(base), 1)
        self.assertEqual(refund_label({**base, "age_days": 31}), 0)
        self.assertEqual(refund_label({**base, "final_sale": True}), 0)
        self.assertEqual(refund_label({**base, "final_sale": True, "defective": True}), 1)
        self.assertEqual(refund_label({**base, "final_sale": True, "defective": True, "age_days": 31}), 0)
        self.assertEqual(refund_label({**base, "final_sale": True, "age_days": 31,
                                       "cancel_before_dispatch": True}), 1)

    def test_deterministic_complete_pairs_and_held_out_compositions(self):
        config = configuration("study")
        first, metadata = generate(config)
        second, _ = generate(config)
        self.assertTrue(first.equals(second))
        self.assertFalse(first.text.duplicated().any())
        self.assertEqual(len(first[first._partition == "test"]), 1200)
        test = first[first._partition == "test"]
        self.assertEqual(test._composition.value_counts().to_dict(), {"familiar": 600, "held-out": 600})
        self.assertTrue(first[first._partition != "test"]._composition.eq("familiar").all())
        for _, group in first.groupby("_pair_id"):
            self.assertEqual(len(group), 2)
            self.assertEqual(group._partition.nunique(), 1)
            self.assertEqual(group._family.nunique(), 1)
            expected = "preserve" if group.label.nunique() == 1 else "flip"
            self.assertTrue(group._pair_relation.eq(expected).all())
            for _, row in group.iterrows():
                self.assertEqual(row.label, refund_label(json.loads(row._facts)))
        self.assertEqual(metadata["label_status"], "synthetic-draft-needs-human-review")

    def test_hidden_labels_and_pair_metadata_never_enter_requests(self):
        frame, metadata, _ = load_job(self.root, DATASET, 2027)
        req = request_for(frame, 0, metadata)
        self.assertEqual(req.state, frame.iloc[0].text)
        self.assertIn(metadata["task"], req.instructions)
        for field in ("_facts", "_pair_id", "_partition", "label", "pair_relation"):
            self.assertNotIn(field, req.as_dict())
        with self.assertRaises(ValueError):
            run_provider(self.root, FakeProvider(), DATASET, 2027, limit=3, min_interval=0)

    def test_new_task_svm_and_provider_export_pair_metrics(self):
        baseline = run_baseline(self.root, DATASET, 2027, "svm", threads=1)
        provider = run_provider(self.root, FakeProvider(), DATASET, 2027, "test", None, min_interval=0)
        self.assertEqual(baseline["n_pairs"], 32)
        self.assertEqual(provider["n_pairs"], 32)
        self.assertEqual(provider["suite"], "policy")
        self.assertEqual(len(collect_summaries(self.root)), 2)
        # Pair groups must remain verifiable against the saved inputs.
        predictions = next((self.root / "providers").rglob("predictions.jsonl"))
        records = [json.loads(line) for line in predictions.read_text().splitlines()]
        records[0]["pair_id"] = "changed"
        predictions.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        with self.assertRaises(ValueError):
            collect_summaries(self.root)

    def test_invalid_pair_counts_are_rejected(self):
        config = configuration()
        config["pairs_per_partition"]["test"] = 7
        with self.assertRaises(ValueError):
            generate(config)

    def test_export_rejects_case_id_edits_even_when_job_and_predictions_agree(self):
        run_provider(self.root, FakeProvider(), DATASET, 2027, limit=4, min_interval=0)
        job_path = next((self.root / "providers").rglob("job.json"))
        job = read_json(job_path)
        job["case_ids"][0] = "substituted-case"
        write_json(job_path, job)
        predictions = job_path.with_name("predictions.jsonl")
        records = [json.loads(line) for line in predictions.read_text().splitlines()]
        records[0]["case_id"] = "substituted-case"
        predictions.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        with self.assertRaises(ValueError):
            collect_summaries(self.root)
