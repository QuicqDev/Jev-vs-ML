"""Deterministic support simulator for the V4 iterative-decision question."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .client import DecisionClient
from .contracts import DecisionRequest, canonical, digest
from .storage import freeze, read_json, write_json

CONDITIONS = ("one_shot", "adaptive_feedback", "repeated_without_new_evidence", "fixed_evidence_schedule")
TERMINALS = ("refund", "deny", "escalate")


def correct_terminal(episode):
    return "refund" if episode["payment_captured"] and not episode["delivered"] else "deny"


def generate(config):
    rng = random.Random(config["iterative_seed"])
    counts = {"train": config["iterative_train_episodes"],
              "policy": config["iterative_policy_episodes"], "test": config["iterative_test_episodes"]}
    result = {}
    cursor = 0
    for partition, count in counts.items():
        episodes = []
        for index in range(count):
            payment = bool((cursor + index) % 2)
            delivered = bool(((cursor + index) // 2) % 2)
            # Rotate which fact is initially known so useful inspection plans vary.
            reveal = (cursor + index) % 3
            episode = {"episode_id": digest({"seed": config["iterative_seed"], "n": cursor + index}),
                       "order_age_days": rng.randint(1, 45), "payment_captured": payment,
                       "delivered": delivered, "initial_payment": payment if reveal == 0 else None,
                       "initial_delivery": delivered if reveal == 1 else None}
            episode["correct_terminal"] = correct_terminal(episode)
            episodes.append(episode)
        result[partition] = episodes
        cursor += count
    return result


def prepare(root, config):
    root = Path(root)
    payload = {"protocol": "deterministic-support-simulator-v1", "conditions": list(CONDITIONS),
               "max_steps": config["iterative_max_steps"], "partitions": generate(config)}
    freeze(root / "iterative" / "episodes.json", payload)
    controls = []
    for partition in ("policy", "test"):
        episodes = payload["partitions"][partition]
        for condition in CONDITIONS:
            for name in ("fully_informed_oracle", "executable_observed_policy"):
                records = []
                for episode in episodes:
                    if name == "fully_informed_oracle" or condition in ("adaptive_feedback", "fixed_evidence_schedule"):
                        success, escalated = True, False
                        steps = 1 if name == "fully_informed_oracle" else 1 + sum(
                            value is None for value in (episode["initial_payment"], episode["initial_delivery"]))
                    else:
                        enough = (episode["initial_payment"] is False or episode["initial_delivery"] is True or
                                  (episode["initial_payment"] is True and episode["initial_delivery"] is False))
                        success, escalated, steps = enough, not enough, 1
                    records.append({"success": success, "escalated": escalated, "critical_error": False,
                                    "steps": steps, "net_utility": (1.0 - .05 * (steps - 1)) if success else -.25,
                                    "terminal": episode["correct_terminal"] if success else "escalate"})
                controls.append({"dataset": "Iterative Support", "partition": partition,
                                 "condition": condition, "baseline": name, **summarize(records)})
    freeze(root / "iterative" / "control_summaries.json", controls)
    return payload


def render_state(episode, known, history):
    payment = "unknown" if known["payment"] is None else ("captured" if known["payment"] else "not captured")
    delivery = "unknown" if known["delivery"] is None else ("delivered" if known["delivery"] else "not delivered")
    return canonical({"request": "Resolve this refund request", "order_age_days": episode["order_age_days"],
                      "payment_status": payment, "delivery_status": delivery, "history": history})


def choices_for(condition, known):
    actions = []
    if condition == "adaptive_feedback":
        if known["payment"] is None:
            actions.append(("inspect_payment", "Inspect the payment record and receive its result"))
        if known["delivery"] is None:
            actions.append(("inspect_delivery", "Inspect delivery tracking and receive its result"))
    actions += [("refund", "Approve and issue the refund"), ("deny", "Deny the refund"),
                ("escalate", "Escalate to a human reviewer")]
    return tuple((f"C{i}", description) for i, (_, description) in enumerate(actions)), [name for name, _ in actions]


def _request(episode, known, history, condition):
    choices, actions = choices_for(condition, known)
    instruction = ("Choose the next action. Refund only when payment was captured and delivery did not occur. "
                   "Deny when payment was not captured or delivery occurred. Inspect an unknown fact when useful; "
                   "escalate if safe automation is impossible.")
    return DecisionRequest(render_state(episode, known, history), instruction, choices), actions


def run_episode(client, episode, condition, max_steps=4):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    known = {"payment": episode["initial_payment"], "delivery": episode["initial_delivery"]}
    history, calls, terminal = [], [], None
    repeats = max_steps if condition == "repeated_without_new_evidence" else max_steps
    for step in range(repeats):
        request, actions = _request(episode, known, history, condition)
        result = client.call(request, context={"episode_id": episode["episode_id"], "condition": condition,
                                               "step": step}, use_cache=False)
        action = actions[result["prediction"]] if result["status"] == "ok" else "escalate"
        calls.append({**result, "step": step + 1, "action": action})
        if condition == "repeated_without_new_evidence":
            terminal = action
            continue
        if condition == "fixed_evidence_schedule":
            terminal = action
            if step == 0:
                known["payment"] = episode["payment_captured"]
                history.append("Scheduled evidence: payment status revealed.")
                continue
            if step == 1:
                known["delivery"] = episode["delivered"]
                history.append("Scheduled evidence: delivery status revealed.")
                continue
            break
        if action == "inspect_payment":
            known["payment"] = episode["payment_captured"]
            history.append("Payment inspection completed.")
        elif action == "inspect_delivery":
            known["delivery"] = episode["delivered"]
            history.append("Delivery inspection completed.")
        else:
            terminal = action
            break
        if condition == "one_shot":
            break
    terminal = terminal or "escalate"
    correct = terminal == episode["correct_terminal"]
    critical = terminal in ("refund", "deny") and not correct
    utility = (1.0 if correct else (-2.0 if critical else -.25)) - .05 * max(0, len(calls) - 1)
    return {"episode_id": episode["episode_id"], "condition": condition, "terminal": terminal,
            "correct_terminal": episode["correct_terminal"], "success": correct,
            "critical_error": critical, "escalated": terminal == "escalate", "steps": len(calls),
            "net_utility": utility, "calls": calls}


def summarize(records):
    n = len(records)
    latencies = [sum(call.get("latency_ms") or 0 for call in r.get("calls", [])) for r in records if r.get("calls")]
    return {"n_episodes": n, "success_rate": sum(r["success"] for r in records) / n,
            "mean_net_utility": sum(r["net_utility"] for r in records) / n,
            "critical_error_rate": sum(r["critical_error"] for r in records) / n,
            "escalation_rate": sum(r["escalated"] for r in records) / n,
            "mean_steps": sum(r["steps"] for r in records) / n,
            "episode_latency_p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
            "episode_latency_p95_ms": sorted(latencies)[min(len(latencies) - 1, int(.95 * len(latencies)))] if latencies else None,
            "terminal_counts": dict(Counter(r["terminal"] for r in records))}


def run_provider(root, provider, partition="test", max_attempts=50000, max_seconds=7200, min_interval=.15):
    root = Path(root)
    run = read_json(root / "run.json")
    payload = read_json(root / "iterative" / "episodes.json")
    provider_root = root / "providers" / digest(provider.identity)
    client = DecisionClient(provider, provider_root, run["run_id"], max_attempts=max_attempts,
                            max_seconds=max_seconds, min_interval=min_interval)
    summaries = []
    for condition in CONDITIONS:
        job = provider_root / "Iterative_Support" / condition / partition
        freeze(job / "job.json", {"run_id": run["run_id"], "provider": provider.identity,
                                   "condition": condition, "partition": partition})
        records = []
        for episode in payload["partitions"][partition]:
            path = job / "episodes" / (episode["episode_id"] + ".json")
            record = read_json(path) if path.exists() else run_episode(
                client, episode, condition, payload["max_steps"])
            if not path.exists():
                write_json(path, record)
            records.append(record)
        summary = {"dataset": "Iterative Support", "partition": partition, "condition": condition,
                   "provider": provider.identity, **summarize(records)}
        (job / "episodes.jsonl").write_text("".join(canonical(record) + "\n" for record in records), encoding="utf-8")
        write_json(job / "summary.json", summary)
        summaries.append(summary)
    return summaries
