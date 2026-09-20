"""Local diagnostic only: explain low balanced accuracy, without changing the benchmark."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from legacy import kaggle_benchmark as b

root = Path(__file__).resolve().parents[1] / 'results' / 'notebook_data_validation' / 'data'
cfg = b.config('benchmark')
results = []
with b.threadpool_limits(limits=2):
    for name in ['Bank Marketing', 'Online Shoppers']:
        folder = root / name.replace(' ', '_')
        df = pd.read_parquet(folder / 'snapshot.parquet')
        meta = json.loads((folder / 'metadata.json').read_text())
        split = b.make_split(df, cfg, 42)
        tr, va, te = [split[k] for k in ['train', 'validation', 'test']]
        y = df.label.to_numpy()
        prep = b.Representations(meta, cfg, 42).fit(df.iloc[tr][meta['features']])
        a, v = prep.transform(df.iloc[tr][meta['features']]), prep.transform(df.iloc[va][meta['features']])
        specs = b.model_specs(cfg, 42, False, False)
        for name_model in ['Logistic regression', 'XGBoost']:
            rep, estimator, options = specs[name_model]
            trials = []
            for params in options:
                model = clone(estimator).set_params(**params).fit(a[rep], y[tr])
                p = model.predict_proba(v[rep])[:, 1]
                trials.append((balanced_accuracy_score(y[va], model.predict(v[rep])), params, p))
            _, params, p_val = max(trials, key=lambda t: t[0])
            thresholds = np.linspace(.05, .95, 19)
            threshold = max(thresholds, key=lambda t: balanced_accuracy_score(y[va], p_val >= t))
            joined = np.concatenate([tr, va])
            final_prep = b.Representations(meta, cfg, 42).fit(df.iloc[joined][meta['features']])
            fit_x = final_prep.transform(df.iloc[joined][meta['features']])[rep]
            test_x = final_prep.transform(df.iloc[te][meta['features']])[rep]
            model = clone(estimator).set_params(**params).fit(fit_x, y[joined])
            p = model.predict_proba(test_x)[:, 1]
            pred = model.predict(test_x)
            record = dict(dataset=name, model=name_model, test_n=len(te), positives=int(y[te].sum()),
                          predicted_positives=int(pred.sum()), confusion=confusion_matrix(y[te], pred).tolist(),
                          default_balanced_accuracy=float(balanced_accuracy_score(y[te], pred)),
                          auc=float(roc_auc_score(y[te], p)), validation_selected_threshold=float(threshold),
                          diagnostic_threshold_balanced_accuracy=float(balanced_accuracy_score(y[te], p >= threshold)))
            results.append(record)
            print(json.dumps(record), flush=True)
b.write_json(root.parent / 'threshold_diagnostic.json', results)
