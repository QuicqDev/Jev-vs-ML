import tempfile
import unittest
from pathlib import Path

from jevbench_v4.client import BudgetExceeded, DecisionClient
from jevbench_v4.contracts import PermanentProviderError, RetryableProviderError, UnsupportedInput
from tests.v4.helpers import FakeProvider, request


class ClientTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[2] / "generated"
        directory.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.provider = FakeProvider()

    def client(self, **kwargs):
        return DecisionClient(self.provider, self.root, "run-one", min_interval=0, sleep=lambda _: None, **kwargs)

    def test_resume_uses_cache_and_does_not_count_cached_latency(self):
        first = self.client().call(request())
        cached = self.client().call(request())
        self.assertEqual(self.provider.calls, 1)
        self.assertTrue(cached["cache_hit"])
        self.assertIsNone(cached["latency_ms"])
        self.assertEqual(cached["attempts"], 0)
        self.assertEqual(first["probabilities"], cached["probabilities"])

    def test_trajectory_changes_cache_and_repeated_control_executes_fresh(self):
        client = self.client()
        client.call(request(), context={"trajectory": []})
        client.call(request(), context={"trajectory": ["inspect_payment"]})
        client.call(request(), use_cache=False)
        client.call(request(), use_cache=False)
        self.assertEqual(self.provider.calls, 4)

    def test_provider_and_run_identity_cannot_change_on_resume(self):
        self.client()
        self.provider.identity["revision"] = "another"
        with self.assertRaises(ValueError):
            self.client()
        with self.assertRaises(ValueError):
            DecisionClient(FakeProvider(), self.root, "another-run")

    def test_attempt_ceiling_survives_restart(self):
        self.client(max_attempts=1).call(request(), use_cache=False)
        with self.assertRaises(BudgetExceeded):
            self.client(max_attempts=1).call(request(), use_cache=False)
        self.assertEqual(self.provider.calls, 1)

    def test_elapsed_ceiling_stops_before_call(self):
        now = [0.0]
        client = self.client(max_seconds=1, clock=lambda: now[0])
        now[0] = 2
        with self.assertRaises(BudgetExceeded):
            client.call(request())
        self.assertEqual(self.provider.calls, 0)

    def test_nonfinite_and_fractional_budgets_are_rejected(self):
        for options in ({"max_attempts": 1.5}, {"max_attempts": True},
                        {"max_seconds": float("nan")}, {"max_seconds": float("inf")},
                        {"max_retries": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.client(**options)

    def test_only_transient_errors_retry_and_invalid_results_are_not_cached(self):
        original = self.provider.predict
        attempts = []
        def temporary(req, audit):
            attempts.append(1)
            if len(attempts) < 3:
                raise RetryableProviderError("http_429")
            return original(req, audit)
        self.provider.predict = temporary
        self.assertEqual(self.client().call(request())["attempts"], 3)
        self.provider.predict = lambda *args: {"answers": {}}
        client = self.client()
        for _ in range(2):
            result = client.call(request(), context={"invalid": True})
            self.assertEqual(result["status"], "invalid")
            self.assertEqual(result["attempts"], 1)
        self.assertEqual(len(list((self.root / "cache").glob("*.json"))), 1)

    def test_unsupported_input_never_reaches_model(self):
        def reject(req):
            raise UnsupportedInput("too_long")
        self.provider.audit = reject
        result = self.client().call(request())
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(self.provider.calls, 0)

    def test_permanent_error_stops_client(self):
        def fail(*args):
            raise PermanentProviderError("HTTP 401")
        self.provider.predict = fail
        client = self.client()
        with self.assertRaises(PermanentProviderError):
            client.call(request())
        with self.assertRaises(PermanentProviderError):
            client.call(request())
        self.assertEqual(client.attempts, 1)

    def test_failed_calls_are_counted_without_persisting_exception_text(self):
        def fail(*args):
            raise RuntimeError("sensitive backend detail")
        self.provider.predict = fail
        result = self.client().call(request())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["prediction"], -1)
        self.assertNotIn("sensitive", (self.root / "attempts.jsonl").read_text())
