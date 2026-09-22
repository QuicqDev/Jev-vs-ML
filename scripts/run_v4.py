"""Run additional V4 tasks and combine independent Jev/local notebook outputs."""
import argparse
from pathlib import Path

from jevbench_v4.data import DATASETS, configuration, job_ids, load_job, prepare
from jevbench_v4.providers import make_provider
from jevbench_v4.storage import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "pilot", "evaluate", "local", "all-providers", "baselines", "export", "merge"))
    parser.add_argument("--root", type=Path, default=Path("results/v4_pilot"))
    parser.add_argument("--suite", choices=("full", "policy", "continuity"), default="policy")
    parser.add_argument("--preset", choices=("pilot", "study", "continuity"), default="pilot")
    parser.add_argument("--inputs", type=Path, nargs="+", help="Extracted run directories to merge")
    parser.add_argument("--provider", choices=("jev", "von", "laya"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=("pilot", "evaluate"), default="pilot", help="Phase for the local GPU launcher")
    parser.add_argument("--gpus", type=int, nargs="+", help="Visible GPU indices; default uses up to two available GPUs")
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2, help="CPU threads per GPU worker")
    parser.add_argument("--automl-minutes", type=int, default=30)
    parser.add_argument("--isolated-gpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--max-attempts", type=int, default=50000)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--min-interval", type=float, help="Seconds between attempts; defaults to 0 for local models, 0.15 for Jev")
    args = parser.parse_args(argv)
    if args.command == "merge":
        if not args.inputs or len(args.inputs) < 2:
            parser.error("merge requires --inputs with at least two extracted run directories")
        from jevbench_v4.workers import merge_runs
        print(merge_runs(args.root, args.inputs))
        return
    if args.command == "prepare":
        prepare(args.root, configuration(args.preset, args.suite))
        return
    run = read_json(args.root / "run.json")
    if run["config"].get("suite", "continuity") != args.suite:
        parser.error("Run suite differs from --suite; old-dataset diagnostics require --suite continuity")
    datasets = [args.dataset] if args.dataset else run["config"]["datasets"]
    if any(dataset not in run["config"]["datasets"] for dataset in datasets):
        parser.error("Requested dataset is not part of the frozen run")
    if args.command == "verify":
        for dataset in datasets:
            for seed in job_ids(run["config"], dataset):
                load_job(args.root, dataset, seed)
        print(f"Verified {run['run_id']}")
        return
    if args.command == "local":
        from jevbench_v4.parallel import run_local
        run_local(args.root, phase=args.phase, indices=args.gpus, min_gpus=args.min_gpus,
                  cpu_threads=args.cpu_threads, suite=args.suite, dataset=args.dataset,
                  max_attempts=args.max_attempts, max_seconds=args.max_seconds,
                  min_interval=args.min_interval if args.min_interval is not None else 0)
        return
    if args.command == "all-providers":
        from jevbench_v4.parallel import run_all_providers
        run_all_providers(args.root, phase=args.phase, indices=args.gpus, min_gpus=args.min_gpus,
                          cpu_threads=args.cpu_threads, suite=args.suite,
                          max_attempts=args.max_attempts, max_seconds=args.max_seconds)
        return
    if args.command in ("pilot", "evaluate"):
        if args.provider is None:
            parser.error("--provider is required for pilot/evaluate")
        from jevbench_v4.runner import run_provider
        interval = args.min_interval if args.min_interval is not None else (.15 if args.provider == "jev" else 0)
        execution = None
        if args.isolated_gpu:
            if args.provider == "jev" or args.device != "cuda:0":
                parser.error("Isolated GPU workers require a local provider on their only visible cuda:0")
            from jevbench_v4.parallel import gpu_worker_context
            execution = gpu_worker_context(args.provider)
        provider = make_provider(args.provider, device=args.device, jev_model=args.jev_model)
        for dataset in datasets:
            for seed in job_ids(run["config"], dataset):
                partitions = ["train"] if args.command == "pilot" else ["policy", "test"]
                for partition in partitions:
                    run_provider(args.root, provider, dataset, seed, partition,
                        limit=50 if partition == "train" else None, max_attempts=args.max_attempts,
                        max_seconds=args.max_seconds, min_interval=interval, execution=execution)
        if run["config"].get("suite") == "full" and args.command == "evaluate":
            from jevbench_v4.iterative import run_provider as run_iterative
            run_iterative(args.root, provider, max_attempts=args.max_attempts,
                          max_seconds=args.max_seconds, min_interval=interval)
        if execution is not None:
            import torch
            from jevbench_v4.storage import write_json
            write_json(args.root / "execution/local" / args.command / (args.provider + "_gpu.json"),
                       {**execution, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)})
    elif args.command == "baselines":
        from jevbench_v4.baselines import run_baseline
        for dataset in datasets:
            for seed in job_ids(run["config"], dataset):
                names = (["majority", "svm", "embedding"] if dataset == "Support Policy" else
                         ["persistence", "seasonal", "svm", "catboost"] if dataset == "Bike Demand" else
                         ["majority", "svm"] + (["catboost"] if dataset == "Bank Marketing" else []))
                for name in names:
                    run_baseline(args.root, dataset, seed, name)
                if dataset == "Bike Demand":
                    from jevbench_v4.baselines import run_automl
                    run_automl(args.root, dataset, seed, threads=args.cpu_threads,
                               time_limit=args.automl_minutes * 60)
    elif args.command == "export":
        from jevbench_v4.export import export_run
        print(export_run(args.root))


if __name__ == "__main__":
    main()
