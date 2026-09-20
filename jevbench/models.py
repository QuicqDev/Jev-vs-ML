"""Four predeclared candidates per family; selected without test access."""
from .common import *
from sklearn.utils.class_weight import compute_sample_weight


def candidates(name, cfg, text, seed, gpu_id=None):
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    n, jobs = cfg['trees'], cfg['threads']
    specs = {
        'Logistic regression': ('linear', LogisticRegression(max_iter=3000, random_state=seed), [{'C': .3}, {'C': 3}]),
        'SVM': ('linear', LinearSVC(dual='auto', max_iter=10000, random_state=seed) if text else SVC(cache_size=512), [{'C': .3}, {'C': 3}]),
        'Decision tree': ('tree', DecisionTreeClassifier(random_state=seed), [{'max_depth': 10, 'min_samples_leaf': 3}, {'max_depth': None, 'min_samples_leaf': 10}]),
        'Random forest': ('tree', RandomForestClassifier(n_estimators=n, n_jobs=jobs, random_state=seed), [{'max_depth': 16, 'min_samples_leaf': 2, 'max_features': 'sqrt'}, {'max_depth': None, 'min_samples_leaf': 1, 'max_features': .5}]),
        'Extra trees': ('tree', ExtraTreesClassifier(n_estimators=n, n_jobs=jobs, random_state=seed), [{'max_depth': 16, 'min_samples_leaf': 2, 'max_features': 'sqrt'}, {'max_depth': None, 'min_samples_leaf': 1, 'max_features': .5}]),
        'Naive Bayes': ('word' if text else 'dense', MultinomialNB() if text else GaussianNB(), [{'alpha': .1}, {'alpha': 1.0}] if text else [{'var_smoothing': 1e-9}, {'var_smoothing': 1e-6}]),
        'Hist gradient boost': ('dense', HistGradientBoostingClassifier(max_iter=n, early_stopping=False, random_state=seed), [{'max_leaf_nodes': 15, 'learning_rate': .05, 'l2_regularization': 1}, {'max_leaf_nodes': 31, 'learning_rate': .1, 'l2_regularization': 5}]),
        'XGBoost': ('tree', XGBClassifier(n_estimators=n, tree_method='hist', device=f'cuda:{gpu_id}' if gpu_id is not None else 'cpu', n_jobs=jobs, random_state=seed), [{'max_depth': 3, 'learning_rate': .05, 'min_child_weight': 3, 'reg_lambda': 3, 'subsample': .9, 'colsample_bytree': .9}, {'max_depth': 6, 'learning_rate': .1, 'min_child_weight': 5, 'reg_lambda': 5, 'subsample': .9, 'colsample_bytree': .9}]),
        'CatBoost': ('cat', CatBoostClassifier(iterations=n, verbose=False, allow_writing_files=False, task_type='GPU' if gpu_id is not None else 'CPU', thread_count=jobs, random_seed=seed, **({'devices': str(gpu_id)} if gpu_id is not None else {})), [{'depth': 4, 'learning_rate': .05, 'l2_leaf_reg': 3}, {'depth': 7, 'learning_rate': .1, 'l2_leaf_reg': 5}]),
    }
    if name == 'k-NN':
        return [('knn', KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=jobs),
                 dict(n_neighbors=k, weights=w), False) for k, w in [(5, 'distance'), (15, 'distance'), (31, 'distance'), (15, 'uniform')]][:cfg['max_trials']]
    rep, base, settings = specs[name]
    if cfg.get('speed_profile'):
        if name in ['Random forest', 'Extra trees']:
            settings = [dict(max_depth=16, min_samples_leaf=2, max_features='sqrt'),
                        dict(max_depth=24, min_samples_leaf=1, max_features=.15)]
        if name == 'Hist gradient boost':
            # Multiclass HGB builds one tree PER CLASS per iteration on CPU.
            rep = 'knn' if text else 'dense'
            base.set_params(max_iter=cfg['histogram_iterations'], max_bins=63,
                            early_stopping=True, n_iter_no_change=cfg['early_stopping'],
                            validation_fraction=.15)
    # Alternation ensures even quick mode compares unweighted and balanced fits.
    return [(rep, clone(base).set_params(**params), params, weighted)
            for params in settings for weighted in [False, True]][:cfg['max_trials']]


def fit_candidate(name, model, x, y, weighted, cfg, validation=None, cat_columns=None):
    kwargs = {}
    if weighted:
        kwargs['sample_weight'] = compute_sample_weight('balanced', y)
    if name == 'CatBoost' and cat_columns:
        kwargs['cat_features'] = cat_columns
    if validation is not None and name == 'XGBoost':
        model.set_params(early_stopping_rounds=cfg['early_stopping'])
        kwargs.update(eval_set=[validation], verbose=False)
    if validation is not None and name == 'CatBoost':
        kwargs.update(eval_set=validation, early_stopping_rounds=cfg['early_stopping'], use_best_model=True)
    model.fit(x, y, **kwargs)
    if name == 'XGBoost' and cfg.get('require_gpu'):
        actual = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        if not actual.startswith('cuda'):
            raise RuntimeError(f'XGBoost unexpectedly used {actual}; GPU required')
    rounds = None
    if validation is not None and name == 'XGBoost':
        rounds = int(model.best_iteration) + 1
    elif validation is not None and name == 'CatBoost':
        rounds = int(model.tree_count_)
    elif validation is not None and name == 'Hist gradient boost' and cfg.get('speed_profile'):
        rounds = int(model.n_iter_)
    return model, rounds
