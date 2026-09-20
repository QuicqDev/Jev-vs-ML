"""Check actual process scheduling with notebook-defined functions and no GPU/API."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid


def main():
    root_dir = Path(__file__).resolve().parents[1]
    # Notebook functions are not importable from a Python module. Exercise that
    # cloudpickle path explicitly rather than only testing module-level functions.
    namespace = {'__name__': 'notebook_validation_session'}
    source = (root_dir / 'legacy' / 'kaggle_benchmark.py').read_text(encoding='utf-8')
    exec(compile(source, '<notebook-definitions>', 'exec'), namespace)
    b = SimpleNamespace(**namespace)
    cfg = b.config('quick')
    cfg.update(datasets=['Iris', 'Breast Cancer'], trees=4, train_cap=100,
               validation_cap=50, test_cap=20, latency_sample_rows=1, threads=1)
    root = root_dir / 'results' / 'notebook_tests' / ('parallel_' + uuid.uuid4().hex[:8])
    root = b.init_run(cfg, root)
    prepared, _ = b.prepare_suite(root, cfg)
    plan = b.plan_lanes(list(prepared), root, cfg, [0, 1])
    assert len(plan) == min(2, os.cpu_count() or 2)
    assert len({lane['gpu_id'] for lane in plan}) == len(plan)
    jobs = [tuple(job) for lane in plan for job in lane['jobs']]
    assert len(jobs) == len(set(jobs)) == 2
    for gpu_id in [0, 1]:
        specs = b.model_specs(cfg, 42, True, False, gpu_id=gpu_id)
        assert specs['XGBoost'][1].get_params()['device'] == f'cuda:{gpu_id}'
        assert specs['CatBoost'][1].get_params()['devices'] == str(gpu_id)
    print('Two-GPU routing, exclusive lanes and unique job assignment: PASS', flush=True)
    b.run_baselines(prepared, root, cfg, [])  # two real CPU processes, no CUDA needed
    files = sorted((root / 'runs').rglob('ml.json'))
    assert len(files) == 2
    for file in files:
        records = json.loads(file.read_text())
        assert len(records) == 11
        assert all(row['parallel_training_workers'] == len(plan) for row in records)
    before = [p.stat().st_mtime_ns for p in files]
    b.run_baselines(prepared, root, cfg, [])
    assert before == [p.stat().st_mtime_ns for p in files]
    assert (root / 'report.html').exists()
    print('Notebook-defined workers: process execution, all 11 models, report and checkpoint reuse: PASS', flush=True)
    print('Artifacts:', root)


if __name__ == '__main__':
    main()
