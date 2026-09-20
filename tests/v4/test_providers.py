import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jevbench_v4.contracts import PermanentProviderError, RetryableProviderError
from jevbench_v4.providers import JevProvider, LayaProvider, VonProvider, audit_device, verify_source
from tests.v4.helpers import body, request


class ProviderTests(unittest.TestCase):
    def test_gpu_fallback_and_misplaced_model_tensors_are_rejected(self):
        model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cuda:0")]),
                                buffers=lambda: iter([SimpleNamespace(device="cuda:0")]))
        self.assertEqual(audit_device("cuda:0", "cuda:0", model)["actual_device"], "cuda:0")
        with self.assertRaises(PermanentProviderError):
            audit_device("cuda:0", "cpu", model)
        model.buffers = lambda: iter([SimpleNamespace(device="cpu")])
        with self.assertRaises(PermanentProviderError):
            audit_device("cuda:0", "cuda:0", model)

    def test_laya_inference_cpu_fallback_cannot_produce_a_gpu_score(self):
        provider = LayaProvider("cuda:0")
        router = Mock()
        router.predict.return_value = {**body(request()), "routing": {"model": "english"}}
        router.load.return_value = SimpleNamespace(device="cpu")
        provider.router = router
        with self.assertRaisesRegex(PermanentProviderError, "fallback"):
            provider.predict(request(), {"checkpoint": "english"})

    def test_jev_sends_full_state_and_77_choices(self):
        req = request(77)
        response = body(req)
        response["model"] = "jev-1.13.0"
        session = Mock()
        session.post.return_value = SimpleNamespace(status_code=200, json=lambda: response)
        provider = JevProvider(session=session, key="FAKE-KEY")
        provider.predict(req, provider.audit(req))
        sent = session.post.call_args.kwargs["json"]
        self.assertEqual(sent["state"], req.state)
        self.assertEqual(list(sent["questions"]["classification"]["criteria"]), [f"C{i}" for i in range(77)])
        self.assertNotIn("FAKE-KEY", json.dumps(provider.identity))

    def test_http_errors_are_sanitized_and_model_aliases_rejected(self):
        with self.assertRaises(ValueError):
            JevProvider("jev-latest")
        session = Mock()
        for status, error in [(401, PermanentProviderError), (429, RetryableProviderError)]:
            session.post.return_value = SimpleNamespace(status_code=status, text="SECRET BODY")
            with self.assertRaises(error) as caught:
                JevProvider(session=session, key="FAKE").predict(request(), {})
            self.assertNotIn("SECRET BODY", str(caught.exception))
        session.post.return_value = SimpleNamespace(status_code=200, json=lambda: {"model": "another"})
        with self.assertRaises(PermanentProviderError):
            JevProvider(session=session, key="FAKE").predict(request(), {})

    def test_wrong_sdk_revision_fails_before_model_load(self):
        distribution = Mock()
        distribution.read_text.return_value = json.dumps({"vcs_info": {"commit_id": "old"}})
        with patch("importlib.metadata.distribution", return_value=distribution):
            with self.assertRaises(PermanentProviderError):
                verify_source("laya", "required")

    def test_von_invokes_explicit_default_backend(self):
        provider = VonProvider("cpu")
        result = Mock()
        result.model_dump.return_value = body(request())
        backend = Mock()
        backend.evaluate.return_value = result
        provider.backend = backend
        provider.predict(request(), {})
        self.assertEqual(backend.evaluate.call_args.kwargs["model"], "von-1.0.0")
        self.assertEqual(provider.identity["backend"], "sdk-default-pairwise")

    def test_laya_records_selected_checkpoint_and_rejects_route_change(self):
        provider = LayaProvider("cpu")
        router = Mock()
        router.predict.return_value = {**body(request()), "routing": {"model": "english"}}
        router.load.return_value = SimpleNamespace(device="cpu")
        provider.router = router
        result = provider.predict(request(), {"checkpoint": "english"})
        self.assertEqual(result["routing"]["weights_revision"], provider.identity["weights_revision"])
        with self.assertRaises(PermanentProviderError):
            provider.predict(request(), {"checkpoint": "multilingual"})
