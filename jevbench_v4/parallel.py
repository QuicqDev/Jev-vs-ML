"""One isolated model process per GPU; no shared CUDA contexts or model memory."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .data import load_job
from .storage import freeze, read_json, write_json


def run_notebook_command(command, cwd):
    """Forward a Kaggle notebook interrupt so its launcher can stop GPU children."""
    process = subprocess.Popen(command, cwd=cwd)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=5)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def discover_gpus():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the local-model requirements and a CUDA-enabled PyTorch build") from exc
    count = torch.cuda.device_count()
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = [value.strip() for value in mask.split(",")] if mask is not None else list(map(str, range(count)))
    if count > len(selectors):
        raise RuntimeError("CUDA device inventory does not match CUDA_VISIBLE_DEVICES")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "selector": selectors[index], "name": properties.name,
                        "total_memory_bytes": properties.total_memory})
    return devices


def assign_gpus(inventory, indices=None, min_gpus=1):
    if type(min_gpus) is not int or min_gpus < 1:
        raise ValueError("min_gpus must be a positive integer")
    available = {gpu["index"]: gpu for gpu in inventory}
    selected = list(available) if indices is None else list(indices)
    if not selected or len(selected) != len(set(selected)) or any(i not in available for i in selected):
        raise ValueError("Select distinct, available CUDA GPU indices; no CPU fallback is permitted")
    if len(selected) < min_gpus:
        raise ValueError(f"This run requires {min_gpus} GPUs, but only {len(selected)} were selected")
    selected = selected[:2]  # The current local roster contains exactly two models.
    if len({available[i]["selector"] for i in selected}) != len(selected):
        raise ValueError("GPU selectors must refer to distinct devices")
    return [{"provider": provider, "gpu": available[selected[index % len(selected)]],
             "device_in_worker": "cuda:0", "concurrent_workers": len(selected)}
            for index, provider in enumerate(("von", "laya"))]


def worker_environment(assignment, cpu_threads):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = assignment["gpu"]["selector"]
    environment["JEVBENCH_V4_GPU_WORKER"] = json.dumps(assignment)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(cpu_threads)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def gpu_worker_context(provider):
    """Executed in the child after its CUDA visibility is set, before model loading."""
    import torch
    assignment = json.loads(os.environ["JEVBENCH_V4_GPU_WORKER"])
    if assignment["provider"] != provider:
        raise RuntimeError("GPU worker/provider mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != assignment["gpu"]["selector"]:
        raise RuntimeError("GPU worker visibility changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("An isolated GPU worker must see exactly one working CUDA device")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if (properties.name != assignment["gpu"]["name"]
            or properties.total_memory != assignment["gpu"]["total_memory_bytes"]):
        raise RuntimeError("The worker GPU differs from the selected parent device")
    # Exercise actual FP16 allocation and math, including T4's inference dtype.
    sample = torch.ones((32, 32), device="cuda:0", dtype=torch.float16)
    result = sample @ sample
    torch.cuda.synchronize(0)
    if not bool(torch.all(result == 32).item()):
        raise RuntimeError("CUDA FP16 smoke test failed")
    del sample, result
    torch.cuda.reset_peak_memory_stats(0)
    return {**assignment, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "smoke_test": "fp16-matmul-passed"}


def run_local(root, phase="pilot", indices=None, min_gpus=1, cpu_threads=2,
              suite="policy", dataset=None, max_attempts=50000, max_seconds=7200, min_interval=0):
    if phase not in ("pilot", "evaluate"):
        raise ValueError("phase must be pilot or evaluate")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    root = Path(root).resolve()
    run = read_json(root / "run.json")
    if run["config"].get("suite", "continuity") != suite:
        raise ValueError("Run suite differs from requested suite")
    for name in ([dataset] if dataset else run["config"]["datasets"]):
        for seed in run["config"]["seeds"]:
            load_job(root, name, seed)
    assignments = assign_gpus(discover_gpus(), indices, min_gpus)
    directory = root / "execution" / "local"
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".lock"
    try:
        owner = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("A local launcher owns this run; check it before removing execution/local/.lock") from exc
    active, handles, results = {}, [], []
    status_started = False
    pending = list(assignments)
    status_path = directory / phase / "status.json"
    try:
        owner.write(str(os.getpid()))
        owner.close()
        freeze(directory / "plan.json", {"run_id": run["run_id"], "assignments": assignments,
                                         "cpu_threads_per_worker": cpu_threads})
        write_json(status_path, {"run_id": run["run_id"], "phase": phase, "state": "running", "workers": []})
        status_started = True
        while pending or active:
            for assignment in list(pending):
                gpu = assignment["gpu"]["index"]
                if gpu in active:
                    continue
                log = directory / phase / (assignment["provider"] + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open("a", encoding="utf-8")
                handles.append(stream)
                command = [sys.executable, "-m", "scripts.run_v4", phase, "--root", str(root),
                           "--suite", suite, "--provider", assignment["provider"], "--device", "cuda:0",
                           "--isolated-gpu", "--max-attempts", str(max_attempts),
                           "--max-seconds", str(max_seconds), "--min-interval", str(min_interval)]
                if dataset:
                    command += ["--dataset", dataset]
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                    env=worker_environment(assignment, cpu_threads), stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                active[gpu] = (process, assignment, log, time.monotonic())
                pending.remove(assignment)
                print(f"Started {assignment['provider']} on GPU {gpu} ({assignment['gpu']['name']}); log: {log}", flush=True)
            for gpu, (process, assignment, log, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                results.append({**assignment, "exit_code": code, "elapsed_seconds": time.monotonic() - started,
                                "log": log.relative_to(root).as_posix()})
                del active[gpu]
                write_json(status_path, {"run_id": run["run_id"], "phase": phase, "workers": results})
                if code:
                    raise RuntimeError(f"{assignment['provider']} GPU worker failed (exit {code}); inspect {log}")
            if active:
                time.sleep(.1)
        return results
    finally:
        # A failure or notebook interrupt must not leave the other GPU worker running.
        for process, _, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, assignment, log, started in active.values():
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=5)
            results.append({**assignment, "exit_code": code, "status": "interrupted",
                            "elapsed_seconds": time.monotonic() - started,
                            "log": log.relative_to(root).as_posix()})
        try:
            if status_started:
                complete = not pending and not active and len(results) == len(assignments) and all(r["exit_code"] == 0 for r in results)
                write_json(status_path, {"run_id": run["run_id"], "phase": phase,
                                        "state": "complete" if complete else "incomplete", "workers": results})
        finally:
            for stream in handles:
                stream.close()
            owner.close()
            lock.unlink(missing_ok=True)
