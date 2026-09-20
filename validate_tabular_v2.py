"""Diagnostic on OLD pilot holdouts only; does not inspect the new v2 test set."""
from pathlib import Path
import json
import shutil
import uuid
import numpy as np
from jevbench.config import configuration
from jevbench.common import write_json, train_test_split
from jevbench.datasets import prepare_data, pilot_split, cap_indices
from jevbench.training import train_job

root = Path('results') / ('v2_old_holdout_diagnostic_' + uuid.uuid4().hex[:8])
cfg = configuration()
root.mkdir(parents=True)
for name in ['Bank Marketing', 'Online Shoppers']:
    source = Path('results/notebook_data_validation/data') / name.replace(' ', '_')
    dest = root / 'data' / name.replace(' ', '_')
    shutil.copytree(source, dest)
    df, meta = prepare_data(name, root, cfg)
    old = pilot_split(df, dict(train_cap=8000, validation_cap=1000, test_cap=300), 42)
    development = np.r_[old['train'], old['validation']]
    y = df.label.to_numpy()
    remaining, policy = train_test_split(development, test_size=.2, stratify=y[development], random_state=2027)
    train, validation = train_test_split(remaining, test_size=.25, stratify=y[remaining], random_state=2027)
    split = dict(train=cap_indices(train, y, cfg['train_cap'], 2027),
                 validation=cap_indices(validation, y, cfg['validation_cap'], 2027),
                 policy=cap_indices(policy, y, cfg['policy_cap'], 2027), test=old['test'])
    folder = root / 'runs' / name.replace(' ', '_') / '2027'
    write_json(folder / 'split.json', {k: v.tolist() for k, v in split.items()})
    train_job(root, name, 2027, cfg, model_names=['Logistic regression', 'XGBoost', 'CatBoost'])
    for model in ['Logistic regression', 'XGBoost', 'CatBoost']:
        r = json.loads((folder / (model + '.json')).read_text())
        print(name, model, 'raw BA', round(r['raw']['balanced_accuracy'], 4),
              'adjusted BA', round(r['adjusted']['balanced_accuracy'], 4), flush=True)
print(root)
