"""Operational metrics keep failures and unsupported inputs in the denominator."""
import numpy as np
from sklearn.metrics import f1_score, log_loss


def summarize(records, n_classes):
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    y = np.array([r["label"] for r in records])
    pred = np.array([r["prediction"] for r in records])
    if any(label not in range(n_classes) for label in y):
        raise ValueError("Unknown ground-truth class")
    recalls = [float((pred[y == label] == label).mean()) if (y == label).any() else None
               for label in range(n_classes)]
    valid = [r for r in records if r["status"] == "ok" and r.get("probabilities") is not None]
    timing = [r["latency_ms"] for r in records if not r.get("cache_hit")
              and r["status"] != "unsupported" and r.get("latency_ms") is not None]
    result = {"n": len(records), "accuracy": float((pred == y).mean()),
              "balanced_accuracy": float(np.mean(recalls)) if all(r is not None for r in recalls) else None,
              "macro_f1": float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
              "per_class_recall": recalls,
              "statuses": {status: sum(r["status"] == status for r in records)
                           for status in ("ok", "failed", "invalid", "unsupported")},
              "probability_rows": len(valid), "probability_coverage": len(valid) / len(records),
              "brier": None, "log_loss": None, "uncached_timing_rows": len(timing),
              "latency_p50_ms": float(np.quantile(timing, .5)) if timing else None,
              "latency_p95_ms": float(np.quantile(timing, .95)) if timing else None,
              "cache_hits": sum(bool(r.get("cache_hit")) for r in records)}
    if valid:
        probabilities = np.array([r["probabilities"] for r in valid])
        labels = np.array([r["label"] for r in valid])
        if probabilities.shape != (len(valid), n_classes) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid probability matrix")
        if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("Invalid probability distribution")
        result["brier"] = float(np.mean(np.sum((probabilities - np.eye(n_classes)[labels]) ** 2, axis=1)))
        result["log_loss"] = float(log_loss(labels, probabilities, labels=list(range(n_classes))))
    return result
