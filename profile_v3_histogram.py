"""Measure the revised Banking77 bottleneck on training data, never test scores."""
from pathlib import Path
import time
from jevbench.config import configuration
from jevbench.datasets import prepare_data, make_holdout, make_split
from jevbench.features import Features
from jevbench.models import candidates, fit_candidate
from jevbench.common import threadpool_limits

cfg = configuration('v3')
cfg.update(use_cuml=False, require_gpu=False)
df, meta = prepare_data('Banking77', Path('results/notebook_data_validation'), cfg)
split = make_split(df, make_holdout(df, cfg), cfg, 2027)
x, y = df.iloc[split['train']][meta['features']], df.iloc[split['train']].label.to_numpy()
with threadpool_limits(limits=2):
    start = time.perf_counter()
    f = Features(meta, cfg, 2027).fit(x, y)
    features = f.transform(x)
    print(f'Training feature preparation: {time.perf_counter() - start:.1f}s', flush=True)
    rep, model, params, weighted = candidates('Hist gradient boost', cfg, True, 2027)[0]
    start = time.perf_counter()
    model, _ = fit_candidate('Hist gradient boost', model, features[rep], y, weighted, cfg)
    print(f'Banking77: {len(y)} rows, {features[rep].shape[1]} features, '
          f'{model.n_iter_} iterations, 77 classes; one candidate fit: {time.perf_counter() - start:.1f}s', flush=True)
