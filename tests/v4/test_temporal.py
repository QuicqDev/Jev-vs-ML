import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from jevbench_v4.temporal import _future_windows, build_examples
from jevbench_v4.data import configuration, job_ids, load_job, prepare


class TemporalTests(unittest.TestCase):
    @staticmethod
    def raw(hours=1800):
        timestamp = pd.date_range("2024-01-01", periods=hours, freq="h")
        return pd.DataFrame({"dteday": timestamp.date.astype(str), "hr": timestamp.hour,
            "cnt": 100 + (np.arange(len(timestamp)) % 24), "season": 1, "yr": 0,
            "mnth": timestamp.month, "holiday": 0, "weekday": timestamp.weekday,
            "workingday": 1, "weathersit": 1, "temp": .5, "atemp": .5,
            "hum": .5, "windspeed": .2})

    def test_exact_lags_forward_windows_and_embargo(self):
        raw = self.raw()
        examples = build_examples(raw)
        first = examples.iloc[0]
        self.assertEqual(pd.Timestamp(first.target_timestamp) - pd.Timestamp(first.timestamp), pd.Timedelta(hours=1))
        self.assertEqual(first.count_lag_168, raw.iloc[0].cnt)
        config = {"temporal_test": 80, "temporal_validation": 80, "temporal_policy": 60,
                  "temporal_train_cap": 300, "temporal_gap_hours": 24, "temporal_windows": 3}
        windows = _future_windows(examples, config)
        tests = [set(window["test"]) for window in windows.values()]
        self.assertTrue(tests[0].isdisjoint(tests[1]) and tests[1].isdisjoint(tests[2]))
        for parts in windows.values():
            self.assertTrue(all(set(a).isdisjoint(b) for i, a in enumerate(parts.values())
                                for b in list(parts.values())[i + 1:]))
            test_start, policy_end = parts["test"][0], parts["policy"][-1]
            gap = pd.Timestamp(examples.iloc[test_start].timestamp) - pd.Timestamp(examples.iloc[policy_end].target_timestamp)
            self.assertGreaterEqual(gap, pd.Timedelta(hours=24))

    def test_full_suite_freezes_and_reloads_mocked_temporal_jobs(self):
        generated = Path(__file__).resolve().parents[2] / "generated"
        generated.mkdir(exist_ok=True)
        config = configuration("pilot", "full")
        config.update(temporal_train_cap=300, temporal_validation=80,
                      temporal_policy=60, temporal_test=80)
        with tempfile.TemporaryDirectory(dir=generated) as temporary, \
                patch("jevbench_v4.temporal._download", return_value=(self.raw(), "0" * 64)):
            root = Path(temporary) / "full"
            prepare(root, config)
            for seed in job_ids(config, "Bike Demand"):
                frame, metadata, split = load_job(root, "Bike Demand", seed)
                self.assertEqual(metadata["excluded_columns"],
                                 ["casual", "registered", "future weather", "future count"])
                self.assertEqual(len(split["partitions"]["test"]["indices"]), 80)
                self.assertFalse(frame[metadata["features"]].isna().any().any())
