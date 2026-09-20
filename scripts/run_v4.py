"""Run additional V4 tasks and combine independent Jev/local notebook outputs."""
import argparse
from pathlib import Path

from jevbench_v4.data import DATASETS, configuration, load_job, prepare
from jevbench_v4.providers import make_provider
from jevbench_v4.storage import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "pilot", "evaluate", "baselines", "export", "merge"))
    parser.add_argument("--root", type=Path, default=Path("results/v4_pilot"))
    parser.add_argument("--suite", choices=("policy", "continuity"), default="policy")
    parser.add_argument("--preset", choices=("pilot", "study", "continuity"), default="pilot")
    parser.add_argument("--inputs", type=Path, nargs="+", help="Extracted run directories to merge")
    parser.add_argument("--provider", choices=("jev", "von", "laya"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--max-attempts", type=int, default=50000)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--min-interval", type=float, default=.15)
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
            for seed in run["config"]["seeds"]:
                load_job(args.root, dataset, seed)
        print(f"Verified {run['run_id']}")
        return
    if args.command in ("pilot", "evaluate"):
        if args.provider is None:
            parser.error("--provider is required for pilot/evaluate")
        from jevbench_v4.runner import run_provider
        provider = make_provider(args.provider, device=args.device, jev_model=args.jev_model)
        for dataset in datasets:
            for seed in run["config"]["seeds"]:
                partitions = ["train"] if args.command == "pilot" else ["policy", "test"]
                for partition in partitions:
                    run_provider(args.root, provider, dataset, seed, partition,
                        limit=50 if partition == "train" else None, max_attempts=args.max_attempts,
                        max_seconds=args.max_seconds, min_interval=args.min_interval)
    elif args.command == "baselines":
        from jevbench_v4.baselines import run_baseline
        for dataset in datasets:
            for seed in run["config"]["seeds"]:
                names = ["majority", "svm"] + (["catboost"] if dataset == "Bank Marketing" else [])
                for name in names:
                    run_baseline(args.root, dataset, seed, name)
    elif args.command == "export":
        from jevbench_v4.export import export_run
        print(export_run(args.root))


if __name__ == "__main__":
    main()
