"""Shared requests and strict response validation for decision providers."""
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionRequest:
    state: str
    instructions: str
    choices: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not isinstance(self.state, str) or not self.state.strip():
            raise ValueError("state must be nonempty serialized text")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("instructions must be nonempty text")
        if len(self.choices) < 2 or len(dict(self.choices)) != len(self.choices):
            raise ValueError("At least two distinct choices are required")
        if any(not isinstance(v, str) or not v.strip() for pair in self.choices for v in pair):
            raise ValueError("Choice IDs and descriptions must be nonempty strings")

    @property
    def questions(self):
        return {"classification": {"type": "choice", "instructions": self.instructions,
                                   "criteria": dict(self.choices)}}

    def as_dict(self):
        # Preserve option order in fingerprints: order is part of the experiment.
        return {"state": self.state, "instructions": self.instructions, "choices": list(self.choices)}


class UnsupportedInput(ValueError):
    """The full request cannot be represented by this provider."""


class PermanentProviderError(RuntimeError):
    """Stop the run after an authentication, configuration, or model identity error."""


class RetryableProviderError(RuntimeError):
    """A transient transport/service error; safe error codes only, no response body."""


def validate_answer(body, request):
    if not isinstance(body, dict):
        raise ValueError("Response must be an object")
    answer = body["answers"]["classification"]
    choice = answer["choice"]
    labels = [key for key, _ in request.choices]
    if choice not in labels:
        raise ValueError("Unknown returned choice")
    probabilities = answer["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        raise ValueError("Returned probabilities must cover exactly the supplied choices")
    values = [probabilities[key] for key in labels]
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid probability value")
    total = sum(values)
    # SDKs round each class to four decimals, including 77-way tasks.
    if not math.isclose(total, 1.0, abs_tol=0.005):
        raise ValueError("Probabilities do not sum to one")
    return {"prediction": labels.index(choice), "probabilities": [p / total for p in values],
            "resolved_model": body.get("model"), "routing": body.get("routing"),
            "probability_sum_before_normalization": total}
