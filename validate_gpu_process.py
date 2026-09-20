"""Real fresh-process isolation tests without CUDA hardware or API calls."""
import json
import os
from pathlib import Path
import uuid
from unittest.mock import patch
from jevbench import gpu_process as gp

root = Path('results') / ('gpu_process_test_' + uuid.uuid4().hex[:8])
root.mkdir(parents=True)
cfg = dict(threads=2, max_parallel_jobs=2, datasets=['A', 'B'], seeds=[1])
original = os.environ.get('CUDA_VISIBLE_DEVICES')
results = gp.run_phase(root, cfg, ['GPU-test-A', 'GPU-test-B'], [[], []], 'test-environment')
assert {r['visible_device'] for r in results} == {'GPU-test-A', 'GPU-test-B'}
assert len({r['pid'] for r in results}) == 2
assert all(not r['cuda_imported'] and r['pid'] != os.getpid() for r in results)
assert os.environ.get('CUDA_VISIBLE_DEVICES') == original

with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-A,GPU-B'}):
    assert gp.visible_devices() == ['GPU-A', 'GPU-B']
    with patch.object(gp, 'run_phase', side_effect=RuntimeError('probe failed')) as phase:
        try:
            gp.run_isolated(root, cfg)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Probe failure was ignored')
        assert phase.call_count == 1 and phase.call_args.args[-1] == 'probe'
    with patch.object(gp, 'run_phase', side_effect=[[{'ok': True}, {'ok': True}], [[{'job': 1}], [{'job': 2}]]]) as phase:
        assert gp.run_isolated(root, cfg, [1, 0]) == [{'job': 1}, {'job': 2}]
        assert [c.args[-1] for c in phase.call_args_list] == ['probe', 'train']
        assert phase.call_args_list[0].args[2] == ['GPU-B', 'GPU-A']

# Failure must reap every child, not leave a GPU worker running in the background.
real_popen = gp.subprocess.Popen
children = []
def recorded(*args, **kwargs):
    p = real_popen(*args, **kwargs)
    children.append(p)
    return p
with patch.object(gp.subprocess, 'Popen', side_effect=recorded):
    try:
        gp.run_phase(root, cfg, ['0', '1'], [[], []], 'deliberate-error-test')
    except RuntimeError as exc:
        assert 'Isolated GPU' in str(exc)
    else:
        raise AssertionError('Worker error was ignored')
assert len(children) == 2 and all(p.poll() is not None for p in children)
print('Fresh interpreters, pre-import device masks, UUID mapping, probe barrier and failure cleanup: PASS')
