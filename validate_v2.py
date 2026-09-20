"""Offline v2 integration checks: no secret loading and no real API calls."""
from pathlib import Path
import json
import uuid
import sys
import numpy as np
import pandas as pd
from jevbench.common import write_json, MODELS, ALL_DATASETS
from jevbench.config import configuration
from jevbench.datasets import prepare_data, make_holdout, make_split, pilot_split
from jevbench.runner import prepare_suite, run_ml, run_jev
from jevbench.training import train_job
from jevbench.decisions import choose_threshold
from jevbench.reporting import render_results
from jevbench.features import Features
from jevbench.api import JevClient, make_payload, few_shot_ids


def main():
    root = Path('results') / ('v2_validation_' + uuid.uuid4().hex[:8])
    cfg = configuration('quick')
    if '--v3' in sys.argv:
        cfg = configuration('v3')
        # Local Windows has no cuML/CUDA: test real CPU fits and device routing separately.
        cfg.update(use_cuml=False, require_gpu=False, seeds=[2027], histogram_iterations=5)
    cfg.update(datasets=['Iris', 'Breast Cancer'], max_trials=4, trees=5, early_stopping=2,
               svd_components=4, selected_features=30, bootstrap_samples=30)
    prepare_suite(root, cfg, 'results/notebook_data_validation')
    # Both presets, all real datasets, every seed: exact common test and pilot exclusion.
    for preset in ['quick', 'benchmark', 'v3']:
        full = configuration(preset)
        for name in ALL_DATASETS:
            df, meta = prepare_data(name, Path('results/notebook_data_validation'), full)
            h = make_holdout(df, full)
            for seed in full['seeds']:
                split = make_split(df, h, full, seed)
                assert np.array_equal(split['test'], h['test'])
                assert set(split['test']).isdisjoint(h['pilot_excluded'])
                assert set(few_shot_ids(df, split, full, seed)).issubset(split['train'])
    print('All real dataset partitions and exclusions: PASS', flush=True)
    run_ml(root, cfg, gpu_ids=[])
    # A text dataset exercises sparse TF-IDF, supervised feature selection, SVD and dense trees.
    text = pd.DataFrame({'text': [('great enjoyable story' if i % 3 == 0 else 'awful boring story' if i % 3 == 1 else 'ordinary neutral story') + f' item{i}' for i in range(240)], 'label': np.arange(240) % 3})
    mixed = pd.DataFrame({'amount': np.arange(240, dtype=float), 'category': ['a', 'b', 'c', 'd'] * 60,
                          'label': (np.arange(240) % 5 == 0).astype(int)})
    mixed.loc[::17, 'amount'] = np.nan
    for name, df, kind, labels in [('Synthetic text', text, 'text', ['good', 'bad', 'neutral']),
                                    ('Synthetic mixed', mixed, 'tabular', ['no', 'yes'])]:
        local = dict(cfg, exclude_pilot_tests=False)
        data_dir = root / 'data' / name.replace(' ', '_')
        data_dir.mkdir(parents=True)
        df.to_parquet(data_dir / 'snapshot.parquet', index=False)
        meta = dict(name=name, kind=kind, labels=labels, task='Classify the input.', features=[c for c in df if c != 'label'])
        write_json(data_dir / 'metadata.json', meta)
        split = make_split(df, make_holdout(df, local), local, cfg['seeds'][0])
        folder = root / 'runs' / name.replace(' ', '_') / str(cfg['seeds'][0])
        write_json(folder / 'split.json', {k: v.tolist() for k, v in split.items()})
        train_job(root, name, cfg['seeds'][0], cfg)
        cfg['datasets'].append(name)
    for name in cfg['datasets']:
        folder = root / 'runs' / name.replace(' ', '_') / str(cfg['seeds'][0])
        split = json.loads((folder / 'split.json').read_text())
        for model in MODELS + ['Majority baseline']:
            result = json.loads((folder / (model + '.json')).read_text())
            assert len(result['predictions']['raw']) == len(split['test'])
            assert 0 <= result['adjusted']['balanced_accuracy'] <= 1
            if model != 'Voting ensemble':
                trials = json.loads((folder / (model + '.trials.json')).read_text())
                assert all('error' not in trial for trial in trials), (name, model, trials)
    print('All 11 models + dummy, four candidates, numeric/categorical/text: PASS', flush=True)
    chosen = choose_threshold([0, 0, 1, 1], [.01, .02, .15, .2], .5)
    assert chosen['selection_balanced_accuracy'] == 1 and chosen['threshold'] < .5

    class MockClient:
        def call(self, payload):
            k = len(payload['questions']['classification']['criteria'])
            p = [.8] + [.2 / (k - 1)] * (k - 1)
            return dict(ok=True, prediction=0, probabilities=p, resolved_model='mock-only', cache_hit=False)
    run_jev(root, cfg, MockClient())
    tables = render_results(root, cfg)
    assert len(tables) == 6
    assert (root / 'paired_bootstrap_intervals.csv').stat().st_size > 0
    assert not tables['raw_balanced_accuracy'].eq('pending').any().any()
    assert train_job(root, 'Iris', cfg['seeds'][0], cfg)['status'] == 'cached'

    # Actual adapter serialization/cache tested with a fake transport, never network.
    import jevbench.api as api
    class Response:
        status_code, headers = 200, {}
        def json(self):
            return dict(model='mock-only', answers={'classification': {'choice': 'C0', 'probabilities': {'C0': .8, 'C1': .2}}})
    class Session:
        calls = 0
        def post(self, *args, **kwargs):
            Session.calls += 1
            return Response()
    original = api.requests.Session
    api.requests.Session = Session
    try:
        client = JevClient(root, dict(cfg, min_request_interval=0), 'FAKE-SECRET')
        payload = dict(questions={'classification': {'criteria': {'C0': 'no', 'C1': 'yes'}}})
        assert client.call(payload)['ok']
        assert client.call(payload)['cache_hit'] and Session.calls == 1
        assert all('FAKE-SECRET' not in p.read_text() for p in root.rglob('*.json'))
    finally:
        api.requests.Session = original
    print('Mock API, tqdm, threshold rules, caches, six tables, paired intervals: PASS', flush=True)
    print(root)


if __name__ == '__main__':
    main()
