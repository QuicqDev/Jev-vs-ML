"""Offline integration checks; no real TypeSafe requests or secret loading."""
import json
from pathlib import Path
import uuid

import nbformat
import numpy as np
import pandas as pd
import kaggle_benchmark as b

ROOT = Path(__file__).resolve().parent
nb = nbformat.read(ROOT / 'jev_classification_benchmark.ipynb', as_version=4)
nbformat.validate(nb)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'notebook_cell_{i}', 'exec')
    assert not cell.get('outputs'), 'Deliver a clean notebook without saved outputs'
print('Notebook schema and all code cells: PASS', flush=True)

cfg = b.config('quick')
cfg.update(datasets=['Iris', 'Synthetic text test'], candidates=2, trees=8,
           latency_sample_rows=2, svd_components=4, min_request_interval=0, max_retries=0)
output = ROOT / 'results' / 'notebook_tests'
output.mkdir(parents=True, exist_ok=True)
root = output / ('validation_' + uuid.uuid4().hex[:8])
root.mkdir()
df, meta = b.prepare_data('Iris', root, cfg)
split = b.make_split(df, cfg, 42)
assert set(split['train']).isdisjoint(split['test'])
assert set(split['validation']).isdisjoint(split['test'])
examples = b.few_shot_ids(df, split, cfg, 42)
assert set(examples).issubset(split['train'])
payload = b.make_payload(df, int(split['test'][0]), meta, examples, cfg)
assert 'label' not in payload['state']['input']
assert len(payload['state']['labeled_training_examples']) == 3
folder = root / 'runs' / 'Iris' / '42'
folder.mkdir(parents=True)
b.write_json(folder / 'split.json', {'positions': {k: v.tolist() for k, v in split.items()}})
with b.threadpool_limits(limits=2):
    records = b.run_ml(df, meta, split, cfg, folder, False)
assert len(records) == 11
assert all(len(r['prediction']) == len(split['test']) for r in records)
assert all(0 <= r['balanced_accuracy'] <= 1 for r in records)
print('All 11 tabular models, two candidates, refit and latency: PASS', flush=True)

# Exercise sparse TF-IDF and dense SVD branches, not just toy tabular models.
texts = ['excellent pleasant enjoyable movie', 'terrible boring awful movie', 'ordinary average neutral movie']
text_df = pd.DataFrame({'text': [texts[i % 3] + f' story detail sample{i}' for i in range(120)],
                       'label': [i % 3 for i in range(120)], '_row_id': range(120)})
text_meta = dict(kind='text', features=['text'], labels=['positive', 'negative', 'neutral'], task='Classify sentiment.')
text_split = b.make_split(text_df, cfg, 42)
text_folder = root / 'runs' / 'Synthetic_text_test' / '42'
text_folder.mkdir(parents=True)
with b.threadpool_limits(limits=2):
    text_records = b.run_ml(text_df, text_meta, text_split, cfg, text_folder, False)
assert len(text_records) == 11
print('All 11 text pipelines: PASS', flush=True)

mixed = pd.DataFrame({'number': [1., np.nan, 3., 4.], 'category': ['a', 'b', 'a', 'b']})
mixed_meta = dict(kind='tabular')
rep = b.Representations(mixed_meta, cfg, 42).fit(mixed)
held = pd.DataFrame({'number': [np.nan], 'category': ['unseen']})
assert np.isfinite(rep.transform(held)['dense']).all()
print('Mixed numeric/categorical preprocessing with missing and unseen values: PASS', flush=True)

class Response:
    status_code = 200
    headers = {}
    def json(self):
        return {'model': 'mock-model', 'answers': {'classification': {
            'choice': 'C0', 'probabilities': {'C0': .7, 'C1': .2, 'C2': .1}, 'confidence': .5}},
            'usage': {'input_tokens': 100, 'output_tokens': 10}}

class Session:
    calls = 0
    def post(self, url, **kwargs):
        assert url == 'https://api.typesafe.ai/v1/systemone'
        Session.calls += 1
        return Response()

original_session = b.requests.Session
b.requests.Session = Session
try:
    client = b.JevClient(root, cfg, 'FAKE-TEST-KEY-NEVER-SAVE')
    first = client.call(payload)
    again = client.call(payload)
    assert first['ok'] and again['cache_hit'] and Session.calls == 1
    results = b.run_jev(df, meta, split, cfg, folder, client)
    assert len(results) == 2 and all(r['failures'] == 0 for r in results)
    assert all(r['n_test'] == len(split['test']) for r in results)
    table = b.render_results(root, cfg)
    assert 'Jev zero-shot' in table.columns and 'XGBoost' in table.columns
    assert (root / 'report.html').exists() and (root / 'balanced_accuracy_heatmap.png').exists()
    for file in root.rglob('*.json'):
        assert 'FAKE-TEST-KEY-NEVER-SAVE' not in file.read_text()
    assert b.probability_metrics(np.array([0, 1]), [[1, 0], [0, 1]])['brier'] == 0
    assert b.probability_metrics(np.array([0, 1]), [[1, 0], [0, 1]])['ece_10_bin'] == 0
    # Invalid probabilities must never enter the cache or be scored as a valid class.
    class InvalidResponse(Response):
        def json(self):
            result = super().json()
            result['answers']['classification']['probabilities']['C0'] = -1
            return result
    client.local.session = type('InvalidSession', (), {'post': lambda *a, **kw: InvalidResponse()})()
    invalid_payload = {**payload, 'state': {'input': 'invalid-vector-test'}}
    invalid = client.call(invalid_payload)
    assert not invalid['ok'] and invalid['prediction'] == -1
    assert not (client.cache / (b.digest(invalid_payload) + '.json')).exists()
    assert b.metrics(np.array([0, 1]), np.array([0, -1]), ['a', 'b'])['accuracy'] == .5
    # Permanent errors halt the client rather than consuming the entire request budget.
    class Unauthorized(Response):
        status_code = 401
    client.local.session = type('BadSession', (), {'post': lambda *a, **kw: Unauthorized()})()
    try:
        client.call({**payload, 'state': {'input': 'unauthorized-test'}})
        raise AssertionError('401 must stop the API client')
    except RuntimeError as exc:
        assert '401' in str(exc)
    assert client.stopped
    print('Mock Jev, cache, training-only demonstrations, secret exclusion, metrics and reporting: PASS', flush=True)
finally:
    b.requests.Session = original_session

print('Validation artifacts:', root)
