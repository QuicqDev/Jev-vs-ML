"""Run a TypeSafe experiment using Python's standard library only."""
import argparse
import datetime
import http.client
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent


def api_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and name.strip() == "TYPESAFE_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].rstrip()
                if value:
                    return value
    raise SystemExit("Set TYPESAFE_API_KEY in the environment or project .env.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", type=Path, default=ROOT / "examples/purchase_intent.json")
    parser.add_argument("--repeat", type=int, default=1, help="Number of sequential paid API calls (1-100)")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100:
        parser.error("--repeat must be between 1 and 100")
    key = api_key()
    payload = json.loads(args.request.read_text(encoding="utf-8-sig"))
    body = json.dumps(payload).encode("utf-8")
    connection = http.client.HTTPSConnection("api.typesafe.ai", timeout=60)
    records = []
    failed = False
    try:
        for i in range(args.repeat):
            start = time.perf_counter()
            try:
                connection.request("POST", "/v1/systemone", body=body, headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                })
                response = connection.getresponse()
                raw = response.read().decode("utf-8", errors="replace").replace(key, "<redacted>")
                elapsed = (time.perf_counter() - start) * 1000
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw": raw[:4000]}
                records.append({"run": i + 1, "http_status": response.status, "latency_ms": round(elapsed, 2), "response": data})
                print(json.dumps(records[-1], indent=2))
                if response.status != 200:
                    failed = True
                    break  # No automatic retries or extra charges on errors.
            except (OSError, http.client.HTTPException) as exc:
                records.append({"run": i + 1, "error": str(exc).replace(key, "<redacted>")})
                print(json.dumps(records[-1]))
                failed = True
                break
    finally:
        connection.close()
    latencies = [r["latency_ms"] for r in records if r.get("http_status") == 200]
    summary = {"successful_calls": len(latencies), "attempted_calls": len(records)}
    if latencies:
        summary.update(min_ms=min(latencies), median_ms=round(statistics.median(latencies), 2), max_ms=max(latencies))
    output = ROOT / "results" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"request": payload, "summary": summary, "runs": records}, indent=2), encoding="utf-8")
    print("Summary:", json.dumps(summary))
    print("Saved:", output)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
