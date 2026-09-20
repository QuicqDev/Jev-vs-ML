"""Explicit cuML routing. No monkey-patching or hidden CPU fallbacks."""
import numpy as np


def route(name, rep, model, weighted, text, classes, cfg, gpu_id):
    backend = ('CUDA' if gpu_id is not None else 'native CPU') if name in ['XGBoost', 'CatBoost'] else 'sklearn CPU'
    reason = ''
    if not cfg.get('use_cuml'):
        return rep, model, backend, reason
    if gpu_id is None:
        raise RuntimeError('cuML enabled but worker has no assigned GPU')
    params = model.get_params()
    if name == 'Logistic regression':
        from cuml.linear_model import LogisticRegression
        model = LogisticRegression(C=params['C'], max_iter=params['max_iter'],
                                   tol=params['tol'], output_type='numpy')
    elif name == 'Random forest' and not weighted:
        from cuml.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**{k: params[k] for k in
            ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'random_state']},
            n_bins=128, n_streams=1, output_type='numpy')
        rep = 'dense'  # Same selected TF-IDF values; cuML RF requires dense input.
    elif name == 'k-NN':
        from cuml.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'],
                                     algorithm='brute', output_type='numpy')
    elif name == 'SVM' and not text and classes == 2:
        from cuml.svm import SVC
        model = SVC(C=params['C'], kernel='rbf', gamma='scale', cache_size=512,
                    tol=params['tol'], output_type='numpy')
    else:
        if name == 'Random forest':
            reason = 'Balanced sample weights retained: sklearn weighted RF candidate'
        elif name == 'SVM':
            reason = 'Sparse linear text SVM / multiclass tabular SVM retained in sklearn'
        elif name not in ['XGBoost', 'CatBoost']:
            reason = 'No cuML substitution selected for this estimator'
        return rep, model, backend, reason
    return rep, model, 'cuML GPU', ''


def probe_cuml(gpu_id, cfg):
    """Run the actual adapters on each device before expensive dataset jobs."""
    import cupy as cp
    import cuml
    from scipy.sparse import csr_matrix
    from .models import candidates, fit_candidate
    from .decisions import outputs
    rng = np.random.default_rng(912)
    x = rng.normal(size=(120, 8)).astype(np.float32)
    rows = []
    with cp.cuda.Device(gpu_id):
        for name, text, classes in [('Logistic regression', False, 2),
                ('Logistic regression', True, 3), ('Random forest', True, 3),
                ('k-NN', False, 3), ('SVM', False, 2)]:
            y = (np.arange(len(x)) % classes).astype(np.int32)
            for weighted in ([False, True] if name in ['Logistic regression', 'SVM'] else [False]):
                print(f'cuML probe: local GPU {gpu_id}, {name}, text={text}, weighted={weighted}', flush=True)
                local = dict(cfg, trees=3)
                choices = candidates(name, local, text, 912, gpu_id)
                rep, base, _, _ = choices[0]
                rep, model, backend, _ = route(name, rep, base, weighted, text, classes, local, gpu_id)
                data = csr_matrix(x) if name == 'Logistic regression' and text else x
                model, _ = fit_candidate(name, model, data, y, weighted, local)
                out = outputs(model, data[:12], classes)
                assert out['raw'].shape == (12,) and np.isfinite(out['score']).all()
                if out['probability'] is not None:
                    assert out['probability'].shape == (12, classes)
                    assert np.allclose(out['probability'].sum(axis=1), 1, atol=.01)
                rows.append(dict(model=name, text=text, classes=classes, weighted=weighted, backend=backend))
        cp.cuda.get_current_stream().synchronize()
    print(f'GPU {gpu_id}: cuML {cuml.__version__} adapter probes passed', flush=True)
    return dict(gpu_id=gpu_id, cuml=cuml.__version__, probes=rows)
