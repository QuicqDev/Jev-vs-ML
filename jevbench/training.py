"""Validation search, refit, independent policy calibration, then test evaluation."""
from .common import *
from .features import Features
from .models import candidates, fit_candidate
from .backends import route
from .decisions import outputs, choose_threshold, score_result
from sklearn.dummy import DummyClassifier
import warnings


def serial_output(out):
    def convert(v):
        if isinstance(v, np.ndarray):
            return np.where(np.isfinite(v), v, None).tolist()
        return v
    return {k: convert(v) for k, v in out.items()}


def finish_result(name, y_policy, y_test, policy, test, classes, cfg, details):
    policy = {k: np.asarray(v) if isinstance(v, list) else v for k, v in policy.items()}
    test = {k: np.asarray(v) if isinstance(v, list) else v for k, v in test.items()}
    rule = None
    adjusted = test['raw'].copy()
    # If API policy calls failed, retain raw choices and disclose that tuning failed.
    if classes == 2 and name != 'Majority baseline':
        if np.isfinite(policy['score']).all():
            rule = choose_threshold(y_policy, policy['score'], policy['default_threshold'], cfg['threshold_quantiles'])
            valid = np.isfinite(test['score'])
            adjusted[valid] = (test['score'][valid] >= rule['threshold']).astype(int)
        else:
            rule = dict(status='not tuned: policy API failures; raw choices retained')
    return dict(model=name, **details, threshold=rule,
                raw=score_result(y_test, test['raw'], classes, test['score'], test['probability']),
                adjusted=score_result(y_test, adjusted, classes, test['score'], test['probability']),
                predictions=dict(raw=test['raw'].tolist(), adjusted=adjusted.tolist()),
                policy_output=serial_output(policy), test_output=serial_output(test))


def train_job(root, dataset, seed, cfg, gpu_id=None, model_names=None):
    root = Path(root)
    folder = root / 'runs' / dataset.replace(' ', '_') / str(seed)
    df = pd.read_parquet(root / 'data' / dataset.replace(' ', '_') / 'snapshot.parquet')
    meta = json.loads((root / 'data' / dataset.replace(' ', '_') / 'metadata.json').read_text())
    split = json.loads((folder / 'split.json').read_text())
    y = df.label.to_numpy()
    x = df[meta['features']]
    classes = len(meta['labels'])
    names = model_names or MODELS + ['Majority baseline']
    if cfg.get('speed_profile') and model_names is None:
        names = ['XGBoost', 'CatBoost'] + [n for n in names if n not in ['XGBoost', 'CatBoost']]
    missing = [n for n in names if not (folder / (n + '.json')).exists()]
    if not missing:
        return dict(dataset=dataset, seed=seed, status='cached')
    with threadpool_limits(limits=cfg['threads']):
        print(f'{dataset}, seed {seed}: preparing features; assigned GPU={gpu_id}', flush=True)
        selection_features = Features(meta, cfg, seed).fit(x.iloc[split['train']], y[split['train']])
        train = selection_features.transform(x.iloc[split['train']])
        validation = selection_features.transform(x.iloc[split['validation']])
        refit_ids = split['train'] + split['validation']
        final_features = Features(meta, cfg, seed).fit(x.iloc[refit_ids], y[refit_ids])
        refit = final_features.transform(x.iloc[refit_ids])
        policy = final_features.transform(x.iloc[split['policy']])
        test = final_features.transform(x.iloc[split['test']])
        for name in missing:
            started = time.perf_counter()
            print(f'{dataset}, seed {seed}: START {name}', flush=True)
            if name == 'Voting ensemble':
                members = ['Logistic regression', 'Random forest', 'XGBoost']
                results = [json.loads((folder / (member + '.json')).read_text()) for member in members]
                outs = []
                for key in ['policy_output', 'test_output']:
                    p = np.mean([r[key]['probability'] for r in results], axis=0)
                    outs.append(dict(raw=p.argmax(axis=1), score=p[:, 1] if classes == 2 else p,
                                     probability=p, default_threshold=.5))
                result = finish_result(name, y[split['policy']], y[split['test']], *outs, classes, cfg,
                                       dict(members=members, method='equal-weight probability average'))
            else:
                trials, best = [], None
                if name == 'Majority baseline':
                    choices = [('dense', DummyClassifier(strategy='most_frequent'), {}, False)]
                else:
                    choices = candidates(name, cfg, meta['kind'] == 'text', seed, gpu_id)
                for trial, (rep, model, params, weighted) in enumerate(choices):
                    rep, model, backend, reason = route(name, rep, model, weighted, meta['kind'] == 'text', classes, cfg, gpu_id)
                    print(f'  {name}: candidate {trial + 1}/{len(choices)}, {backend}, features={train[rep].shape[1]}' +
                          (f' ({reason})' if reason else ''), flush=True)
                    entry = dict(trial=trial, params=params, balanced_weights=weighted, backend=backend,
                                 backend_reason=reason, estimator_class=type(model).__module__ + '.' + type(model).__name__)
                    tick = time.perf_counter()
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            model, rounds = fit_candidate(name, model, train[rep], y[split['train']], weighted, cfg,
                                (validation[rep], y[split['validation']]), getattr(selection_features, 'cats', []))
                            out = outputs(model, validation[rep], classes)
                        entry['warnings'] = sorted(set(str(w.message) for w in caught))
                        prediction = out['raw']
                        if classes == 2 and name != 'Majority baseline':
                            provisional = choose_threshold(y[split['validation']], out['score'], out['default_threshold'], cfg['threshold_quantiles'])
                            prediction = (out['score'] >= provisional['threshold']).astype(int)
                            entry['provisional_validation_threshold'] = provisional['threshold']
                        value = float(balanced_accuracy_score(y[split['validation']], prediction))
                        entry.update(validation_balanced_accuracy=value, rounds=rounds)
                        if best is None or value > best['value']:
                            best = dict(value=value, trial=trial, rounds=rounds)
                    except Exception as exc:
                        entry['error'] = f'{type(exc).__name__}: {exc}'
                    entry['seconds'] = time.perf_counter() - tick
                    trials.append(entry)
                    write_json(folder / (name + '.trials.json'), trials)
                    print(f'  {name}: candidate {trial + 1} finished in {entry["seconds"]:.1f}s' +
                          (' — FAILED: ' + entry['error'] if 'error' in entry else ''), flush=True)
                if best is None:
                    raise RuntimeError(f'{dataset}/{seed}/{name}: all candidates failed; inspect trial log')
                if name == 'Majority baseline':
                    rep, model, params, weighted = choices[0]
                else:
                    rep, model, params, weighted = candidates(name, cfg, meta['kind'] == 'text', seed, gpu_id)[best['trial']]
                rep, model, backend, reason = route(name, rep, model, weighted, meta['kind'] == 'text', classes, cfg, gpu_id)
                if best['rounds'] is not None:
                    if name == 'Hist gradient boost':
                        model.set_params(max_iter=best['rounds'], early_stopping=False)
                    else:
                        model.set_params(**{('iterations' if name == 'CatBoost' else 'n_estimators'): best['rounds']})
                print(f'  {name}: refitting selected candidate', flush=True)
                model, _ = fit_candidate(name, model, refit[rep], y[refit_ids], weighted, cfg,
                                          cat_columns=getattr(final_features, 'cats', []))
                policy_out = outputs(model, policy[rep], classes)
                test_out = outputs(model, test[rep], classes)
                result = finish_result(name, y[split['policy']], y[split['test']], policy_out, test_out, classes, cfg,
                    dict(selected_trial=best['trial'], params=params, balanced_weights=weighted,
                         candidate_trials=len(trials), failed_trials=sum('error' in t for t in trials),
                         boosting_rounds=best['rounds'], representation=final_features.description(rep),
                         backend=backend, backend_reason=reason, estimator_class=type(model).__module__ + '.' + type(model).__name__,
                         selection_balanced_accuracy=best['value'], gpu_id=gpu_id if backend in ['CUDA', 'cuML GPU'] else None))
            result.update(dataset=dataset, seed=seed, training_search_refit_seconds=time.perf_counter() - started,
                          assigned_visible_device=cfg.get('assigned_visible_device'),
                          test_ids=split['test'], test_labels=y[split['test']].tolist())
            write_json(folder / (name + '.json'), result)
            print(f'{dataset}, seed {seed}: {name} saved', flush=True)
    return dict(dataset=dataset, seed=seed, status='complete')


def train_lane(jobs, root, cfg, gpu_id):
    # One sequential worker per GPU prevents simultaneous fits on the same device.
    if cfg.get('use_cuml'):
        import cupy as cp
        with cp.cuda.Device(gpu_id):
            return [train_job(root, name, seed, cfg, gpu_id) for name, seed in jobs]
    return [train_job(root, name, seed, cfg, gpu_id) for name, seed in jobs]
