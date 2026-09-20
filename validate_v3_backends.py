"""Backend routing contracts with constructor doubles; NOT a real GPU test."""
import sys
from types import SimpleNamespace
from unittest.mock import patch
from jevbench.backends import route
from jevbench.config import configuration
from jevbench.models import candidates
from jevbench.runner import run_ml

class Constructor:
    def __init__(self, **kwargs):
        self.parameters = kwargs

cfg = configuration('v3')
assert cfg['jev_workers'] == 8 and cfg['max_trials'] == 4
stubs = {'cuml.linear_model': SimpleNamespace(LogisticRegression=Constructor),
         'cuml.ensemble': SimpleNamespace(RandomForestClassifier=Constructor),
         'cuml.neighbors': SimpleNamespace(KNeighborsClassifier=Constructor),
         'cuml.svm': SimpleNamespace(SVC=Constructor)}
with patch.dict(sys.modules, stubs):
    for text, classes in [(False, 2), (False, 3), (True, 77)]:
        for name in ['Logistic regression', 'Random forest', 'k-NN', 'SVM', 'Extra trees', 'Hist gradient boost']:
            for rep, model, params, weighted in candidates(name, cfg, text, 2027, 1):
                original = model
                rep, model, backend, reason = route(name, rep, model, weighted, text, classes, cfg, 1)
                expected_gpu = (name in ['Logistic regression', 'k-NN'] or
                                name == 'Random forest' and not weighted or
                                name == 'SVM' and not text and classes == 2)
                assert (backend == 'cuML GPU') == expected_gpu
                if expected_gpu:
                    assert model.parameters['output_type'] == 'numpy'
                else:
                    assert model is original and reason
                if name == 'Random forest' and not weighted:
                    assert rep == 'dense' and model.parameters['n_streams'] == 1
                if name == 'Hist gradient boost':
                    assert original.max_iter == 60 and original.early_stopping
                    assert rep == ('knn' if text else 'dense')
try:
    run_ml('.', cfg, gpu_ids=[])
except RuntimeError as exc:
    assert 'No GPU' in str(exc)
else:
    raise AssertionError('GPU-required preset silently accepted CPU')
print('v3 backend routing, retained weights, histogram budget, Jev workers and GPU fail-fast: PASS')
