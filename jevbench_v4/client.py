"""Sequential bounded evaluation, exact-request cache, and durable attempt ledger."""
import time
import math
from pathlib import Path

from .contracts import (PermanentProviderError, RetryableProviderError, UnsupportedInput,
                        canonical, digest, validate_answer)
from .storage import freeze, read_json, write_json


class BudgetExceeded(RuntimeError):
    pass


class DecisionClient:
    def __init__(self, provider, root, run_id, max_attempts=50000, max_seconds=7200,
                 max_retries=2, min_interval=0.15, clock=time.monotonic, sleep=time.sleep):
        if (type(max_attempts) is not int or max_attempts < 1 or type(max_retries) is not int
                or max_retries < 0 or not math.isfinite(max_seconds) or max_seconds <= 0
                or not math.isfinite(min_interval) or min_interval < 0):
            raise ValueError("Invalid request budget")
        self.provider, self.root, self.run_id = provider, Path(root), run_id
        self.clock, self.sleep = clock, sleep
        self.max_attempts, self.max_seconds = max_attempts, max_seconds
        self.max_retries, self.min_interval = max_retries, min_interval
        freeze(self.root / "provider.json", {"run_id": run_id, "provider": provider.identity,
            "max_attempts": max_attempts, "max_seconds": max_seconds, "max_retries": max_retries,
            "min_interval": min_interval})
        self.ledger = self.root / "attempts.jsonl"
        prior = []
        if self.ledger.exists():
            import json
            prior = [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines()]
        self.attempts = sum(row["event"] == "start" for row in prior)
        self.elapsed_before = max((row.get("elapsed_seconds", 0) for row in prior), default=0)
        self.started = clock()
        self.last_attempt = None
        self.stopped = False

    def _elapsed(self):
        return self.elapsed_before + self.clock() - self.started

    def _log(self, **row):
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(canonical({**row, "elapsed_seconds": self._elapsed()}) + "\n")

    def _reserve(self, key):
        if self.last_attempt is not None:
            self.sleep(max(0, self.min_interval - (self.clock() - self.last_attempt)))
        if self.attempts >= self.max_attempts or self._elapsed() >= self.max_seconds:
            raise BudgetExceeded("Request or elapsed-time ceiling reached")
        self.attempts += 1
        self.last_attempt = self.clock()
        self._log(event="start", attempt=self.attempts, request_hash=key)

    def call(self, request, context=None, use_cache=True):
        if self.stopped:
            raise PermanentProviderError("Provider stopped after a permanent error")
        key = digest({"run_id": self.run_id, "provider": self.provider.identity,
                      "request": request.as_dict(), "context": context or {}})
        cache = self.root / "cache" / (key + ".json")
        if use_cache and cache.exists():
            saved = read_json(cache)
            if saved.get("request_hash") != key:
                raise ValueError("Cache fingerprint mismatch")
            return {**saved, "cache_hit": True, "attempts": 0, "latency_ms": None}
        started = self.clock()
        try:
            audit = self.provider.audit(request)
        except UnsupportedInput as exc:
            return {"status": "unsupported", "prediction": -1, "probabilities": None,
                    "reason": str(exc), "attempts": 0, "cache_hit": False, "request_hash": key,
                    "latency_ms": (self.clock() - started) * 1000}
        audit_ms = (self.clock() - started) * 1000
        for attempt in range(self.max_retries + 1):
            self._reserve(key)
            try:
                body = self.provider.predict(request, audit)
                answer = validate_answer(body, request)
            except PermanentProviderError:
                self.stopped = True
                self._log(event="end", request_hash=key, status="permanent_error")
                raise
            except RetryableProviderError as exc:
                status, reason = "failed", str(exc)
                self._log(event="end", request_hash=key, status=status, reason=reason)
                if attempt < self.max_retries:
                    self.sleep(min(2 ** attempt, 30))
                    continue
            except (KeyError, TypeError, ValueError):
                status, reason = "invalid", "invalid_response"
                self._log(event="end", request_hash=key, status=status)
            except RuntimeError:
                status, reason = "failed", "local_runtime_error"
                self._log(event="end", request_hash=key, status=status, reason=reason)
            else:
                result = {**answer, "status": "ok", "audit": audit, "audit_ms": audit_ms,
                          "latency_ms": (self.clock() - started) * 1000, "attempts": attempt + 1,
                          "cache_hit": False, "request_hash": key}
                self._log(event="end", request_hash=key, status="ok")
                if use_cache:
                    write_json(cache, result)
                return result
            return {"status": status, "prediction": -1, "probabilities": None, "reason": reason,
                    "audit": audit, "audit_ms": audit_ms, "latency_ms": (self.clock() - started) * 1000,
                    "attempts": attempt + 1, "cache_hit": False, "request_hash": key}
