"""Regression checks for capped stratified subsets and both complete presets."""
from pathlib import Path
import numpy as np
import pandas as pd
import kaggle_benchmark as b


def check(y, cap, seed=42):
    ids = np.arange(len(y))
    chosen = b.cap_indices(ids, y, cap, seed)
    assert len(chosen) == min(len(y), cap)
    assert len(set(chosen)) == len(chosen)
    assert set(chosen).issubset(ids)
    assert set(y[chosen]) == set(y)
    assert np.array_equal(chosen, b.cap_indices(ids, y, cap, seed))


# Exact failure: 8,002 rows, 77 classes, only 2 discarded rows.
y = np.arange(8002) % 77
try:
    b.train_test_split(np.arange(len(y)), train_size=8000, stratify=y, random_state=42)
except ValueError as exc:
    assert 'test_size = 2' in str(exc)
else:
    raise AssertionError('Original failure was not reproduced')
for cap in [77, 100, 7999, 8000, 8001, 8002, 9000]:
    check(y, cap)
for counts in [[1, 99], [1, 1, 100], [2, 2, 96], [1, 3, 9, 21], [30, 30, 30]]:
    labels = np.repeat(np.arange(len(counts)), counts)
    for cap in range(len(counts), len(labels) + 1):
        check(labels, cap)
try:
    b.cap_indices(np.arange(100), np.arange(100) % 10, 9, 42)
except ValueError:
    pass
else:
    raise AssertionError('Too-small cap should fail')
print('Original failure reproduced; proportional capping edge cases: PASS')

root = Path(__file__).resolve().parent / 'results' / 'notebook_data_validation' / 'data'
for preset in ['quick', 'benchmark']:
    cfg = b.config(preset)
    for name in cfg['datasets']:
        frame = pd.read_parquet(root / name.replace(' ', '_') / 'snapshot.parquet')
        for seed in cfg['seeds']:
            split = b.make_split(frame, cfg, seed)
            a, v, t = (set(split[k]) for k in ['train', 'validation', 'test'])
            assert not (a & v or a & t or v & t)
            assert all(set(frame.iloc[indices].label) == set(frame.label) for indices in split.values())
    print(f'All eight actual dataset snapshots, {preset} preset, all seeds: PASS')
