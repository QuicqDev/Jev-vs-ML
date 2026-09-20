import unittest

from jevbench_v4.metrics import summarize


class MetricTests(unittest.TestCase):
    def test_pair_success_counts_failed_members_and_rejects_partial_pairs(self):
        context = dict(family="negation", composition="familiar", pair_relation="flip")
        rows = [dict(pair_id=pair, label=label, prediction=label, status="ok", **context)
                for pair in ("a", "b") for label in (0, 1)]
        rows[-1].update(prediction=-1, status="unsupported")
        result = summarize(rows, 2)
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["both_members_correct"], .5)
        with self.assertRaises(ValueError):
            summarize(rows[:-1], 2)

    def test_failures_remain_in_accuracy_and_probability_coverage_is_explicit(self):
        rows = [dict(label=0, prediction=0, status="ok", probabilities=[1, 0], latency_ms=10),
                dict(label=1, prediction=-1, status="failed", probabilities=None, latency_ms=30),
                dict(label=1, prediction=-1, status="unsupported", probabilities=None, latency_ms=1)]
        result = summarize(rows, 2)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertAlmostEqual(result["probability_coverage"], 1 / 3)
        self.assertEqual(result["brier"], 0)
        self.assertEqual(result["statuses"]["unsupported"], 1)

    def test_svm_without_native_probabilities_does_not_invent_them(self):
        rows = [dict(label=i, prediction=i, status="ok", probabilities=None) for i in [0, 1]]
        result = summarize(rows, 2)
        self.assertEqual(result["accuracy"], 1)
        self.assertIsNone(result["brier"])
        self.assertEqual(result["probability_coverage"], 0)

    def test_cached_latency_is_excluded_and_absent_classes_are_visible(self):
        rows = [dict(label=0, prediction=0, status="ok", probabilities=[1, 0],
                     latency_ms=1, cache_hit=True)]
        result = summarize(rows, 2)
        self.assertIsNone(result["balanced_accuracy"])
        self.assertIsNone(result["per_class_recall"][1])
        self.assertEqual(result["uncached_timing_rows"], 0)
