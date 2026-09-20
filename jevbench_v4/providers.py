"""Pinned Jev, Von, and routed Laya adapters; heavy imports are lazy."""
import importlib.metadata
import json
import os
from pathlib import Path

import requests

from .audits import audit_laya, audit_von
from .contracts import PermanentProviderError, RetryableProviderError

VON_SOURCE = "bed7e7337791c5124a557eb59ad918cb6747b276"
VON_WEIGHTS = "94f220771d119201634d1fdd734ce7382b2c185b"
LAYA_SOURCE = "42626c348753fbb17572a813127df2278a1ec527"
LAYA_WEIGHTS = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


def verify_source(distribution, revision):
    metadata = importlib.metadata.distribution(distribution)
    direct = json.loads(metadata.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != revision:
        raise PermanentProviderError(f"Install {distribution} from its pinned Git commit in requirements-v4-local.txt")


class JevProvider:
    def __init__(self, model="jev-1.13.0", session=None, key=None):
        if not model.startswith("jev-") or any(alias in model for alias in ("latest", "preview")):
            raise ValueError("Use a pinned Jev model identifier")
        self.model = model
        self.session = session or requests.Session()
        self._key = key
        self.identity = {"name": "jev", "model": model, "endpoint": "https://api.typesafe.ai/v1/systemone"}

    def audit(self, request):
        # Server-side tokenizer details are unavailable; do not invent token counts.
        return {"representation": "full-json-request", "choice_count": len(request.choices),
                "client_truncated": False, "server_tokenization": "unverified"}

    def predict(self, request, audit):
        if not self._key:
            self._key = os.environ.get("TYPESAFE_API_KEY", "").strip()
            if not self._key:
                try:
                    from kaggle_secrets import UserSecretsClient
                    self._key = UserSecretsClient().get_secret("TYPESAFE_API_KEY").strip()
                except Exception:
                    raise PermanentProviderError("Set TYPESAFE_API_KEY or enable the Kaggle secret") from None
        if not self._key:
            raise PermanentProviderError("TYPESAFE_API_KEY is empty")
        try:
            response = self.session.post(self.identity["endpoint"],
                headers={"Authorization": "Bearer " + self._key},
                json={"model": self.model, "state": request.state, "questions": request.questions},
                timeout=(15, 90))
        except requests.RequestException:
            raise RetryableProviderError("transport_error") from None
        if response.status_code in (408, 429, 500, 502, 503, 504):
            raise RetryableProviderError(f"http_{response.status_code}")
        if response.status_code != 200:
            raise PermanentProviderError(f"Jev HTTP {response.status_code}; response body omitted")
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Jev returned a non-object response")
        if body.get("model") != self.model:
            raise PermanentProviderError("Jev resolved model differs from the pinned model")
        return body


class VonProvider:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.backend = None
        self.identity = {"name": "von", "backend": "sdk-default-pairwise", "source_revision": VON_SOURCE,
                         "weights_repo": "wfzyx/von-1.0", "weights_revision": VON_WEIGHTS,
                         "requested_device": device}

    def _load(self):
        if self.backend is None:
            verify_source("von-sdk", VON_SOURCE)
            from huggingface_hub import snapshot_download
            from von.backends.berta_backend import BertaBackend
            snapshot = snapshot_download("wfzyx/von-1.0", revision=VON_WEIGHTS,
                allow_patterns=["config.json", "model.safetensors", "tokenizer*.json", "calibration.json"])
            self.backend = BertaBackend(variant="von-1.0", device=self.device)
            self.backend.model_id = str(snapshot)
            self.backend._get_model_and_tok()
        return self.backend

    def audit(self, request):
        backend = self._load()
        _, tokenizer = backend._get_model_and_tok()
        return {**audit_von(tokenizer, request), "actual_device": str(backend.device),
                "checkpoint_revision": VON_WEIGHTS}

    def predict(self, request, audit):
        result = self._load().evaluate(state=request.state, questions=request.questions, model="von-1.0.0")
        return result.model_dump()


class LayaProvider:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.router = None
        self.identity = {"name": "laya", "backend": "default-router", "source_revision": LAYA_SOURCE,
                         "weights_repo": "convaiinnovations/laya", "weights_revision": LAYA_WEIGHTS,
                         "auto_task_detection": False, "requested_device": device}

    def _load_router(self):
        if self.router is None:
            verify_source("laya", LAYA_SOURCE)
            from huggingface_hub import snapshot_download
            from laya import Router
            snapshot = Path(snapshot_download("convaiinnovations/laya", revision=LAYA_WEIGHTS,
                allow_patterns=["*.json", "*.safetensors"]))
            # All routes point into one pinned snapshot; no SDK download of latest.
            models = {"english": str(snapshot), "multilingual": str(snapshot / "multilingual"),
                      "typed-decisions": str(snapshot / "typed-decisions")}
            for model in models.values():
                for required in ("rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "model.safetensors"):
                    if not (Path(model) / required).is_file():
                        raise PermanentProviderError("Pinned Laya snapshot is incomplete")
            self.router = Router(models=models, device=self.device, max_loaded=1, auto_task_detection=False)
        return self.router

    def audit(self, request):
        router = self._load_router()
        from laya.common import build_sequence, render_options
        route = router.route(request.state, request.questions)
        agent = router.load(route["model"])
        result = audit_laya(agent.tok, request, agent.cfg, build_sequence, render_options)
        return {**result, "checkpoint": route["model"], "checkpoint_revision": LAYA_WEIGHTS,
                "actual_device": str(agent.device), "routing_reason": route["reason"]}

    def predict(self, request, audit):
        result = self._load_router().predict(request.state, request.questions)
        if result["routing"]["model"] != audit["checkpoint"]:
            raise PermanentProviderError("Laya routing changed between audit and inference")
        result["routing"]["weights_revision"] = LAYA_WEIGHTS
        agent = self.router.load(audit["checkpoint"])
        result["routing"]["actual_device"] = str(agent.device)
        return result


def make_provider(name, device="cuda:0", jev_model="jev-1.13.0"):
    if name == "jev":
        return JevProvider(jev_model)
    if name == "von":
        return VonProvider(device)
    if name == "laya":
        return LayaProvider(device)
    raise ValueError(f"Unknown decision provider: {name}")
