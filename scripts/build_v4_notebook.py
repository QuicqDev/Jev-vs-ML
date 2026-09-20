"""Build separate Jev and local-model notebooks over one frozen additional study."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import zipfile

import nbformat

from jevbench_v4.contracts import digest
from jevbench_v4.data import configuration, prepare
from jevbench_v4.storage import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("jev", "local")


def notebook_cells(role, blob, checksum):
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [markdown(f"""# V4 additional tests — {role} worker

Run this notebook alongside `jev_benchmark_v4_{'local' if role == 'jev' else 'jev'}.ipynb`.
Both embed the **same frozen study**, including identical inputs, splits, source,
and case IDs. They use independent output directories and can run simultaneously.

V3 is complete and remains unchanged. These notebooks do not rerun its datasets or
classical benchmark. This first scaffold covers **new synthetic support-policy pairs**:
negation, exceptions, event order, and paraphrases. Labels are a development draft
pending human review. Temporal/AutoML and iterative-feedback tracks are planned,
not implemented here; see `docs/protocols/BENCHMARK_V4_PLAN.md` after extraction.

Enable Internet. {'A CPU session is sufficient; enable the TYPESAFE_API_KEY Kaggle secret.' if role == 'jev' else 'Use a GPU session for Von/Laya; this notebook needs no Jev secret.'}
All model switches start disabled. Inspect development compatibility and the review
sample before interpreting test results. No output is a finished V4 publication."""),
        code(f'''import base64, hashlib, io, sys, zipfile
from pathlib import Path

WORKER_ROLE = {role!r}
PACKAGE_SHA256 = {checksum!r}
blob = base64.b64decode({base64.b64encode(blob).decode()!r})
assert hashlib.sha256(blob).hexdigest() == PACKAGE_SHA256
CODE_DIR = Path('/kaggle/working') / ('jevbench_v4_code_' + PACKAGE_SHA256[:12])
CODE_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    archive.extractall(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
print(CODE_DIR)
'''), code('''import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE_DIR / 'requirements-dev.txt')])
subprocess.check_call([sys.executable, '-m', 'tests.validate_v4'], cwd=CODE_DIR)
'''), markdown("## Load the common frozen study\n\nNo old dataset download or independent split generation. Both workers copy the same byte-for-byte study package and verify it before inference."),
        code('''import shutil
from jevbench_v4.storage import read_json
from jevbench_v4.parallel import run_notebook_command

ROOT = Path('/kaggle/working') / ('jev_benchmark_v4_' + WORKER_ROLE)
if not ROOT.exists():
    shutil.copytree(CODE_DIR / 'study', ROOT)
if read_json(ROOT / 'run.json') != read_json(CODE_DIR / 'study/run.json'):
    raise ValueError('Different study already exists here; choose a fresh ROOT.')

def command(action, *options):
    run_notebook_command([sys.executable, '-m', 'scripts.run_v4', action,
                          '--root', str(ROOT), *options], cwd=CODE_DIR)

command('verify')
print('Shared run ID:', read_json(ROOT / 'run.json')['run_id'])
print('Review sample:', ROOT / 'data/Support_Policy/review_sample.csv')
''')]
    if role == "jev":
        cells += [markdown("## Jev development compatibility\n\nRuns 50 training-partition cases from the new policy task. Request and time ceilings apply; a running call can finish after the time ceiling."),
            code('''RUN_JEV = False
if RUN_JEV:
    command('pilot', '--provider', 'jev', '--max-attempts', '10000', '--max-seconds', '7200')
'''), markdown("## Additional policy evaluation\n\nAfter development checks, this evaluates only the frozen new policy/test partitions. Synthetic labels still require review before publication."),
            code('''RUN_EVALUATION = False
if RUN_JEV and RUN_EVALUATION:
    command('evaluate', '--provider', 'jev', '--max-attempts', '10000', '--max-seconds', '7200')
''')]
    else:
        cells += [markdown("## Local development compatibility — two T4 GPUs\n\nSelect Kaggle's two-T4 accelerator. The launcher requires two visible GPUs, runs Von on the first and Laya on the second simultaneously, and limits each process to its assigned GPU. Each worker performs an FP16 CUDA smoke test and rejects SDK CPU fallback. The Jev notebook runs independently."),
            code('''RUN_LOCAL_MODELS = False
if RUN_LOCAL_MODELS:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE_DIR / 'requirements-v4-local.txt')])
    command('local', '--phase', 'pilot', '--gpus', '0', '1', '--min-gpus', '2')
'''), markdown("## Comparators on the new task only\n\nMajority and TF-IDF SVM use the new task's development labels. This does not repeat V3. Embeddings and temporal AutoML remain planned and must be added before claiming those Reddit questions are answered."),
            code('''RUN_EVALUATION = False
RUN_NEW_TASK_BASELINES = False
if RUN_LOCAL_MODELS and RUN_EVALUATION:
    command('local', '--phase', 'evaluate', '--gpus', '0', '1', '--min-gpus', '2')
if RUN_NEW_TASK_BASELINES and RUN_EVALUATION:
    command('baselines')
''')]
    cells += [markdown("## Export this worker\n\nSave both worker ZIPs. GPU assignment, preflight, peak-memory records, and per-provider logs are included for local runs. Extract each ZIP into a separate directory, then merge with `python -m scripts.run_v4 merge --root results/v4_combined --inputs path/to/jev path/to/local`. The merge rejects different studies, mismatched data, and conflicting outputs. It needs no inference or API key."),
        code('''from IPython.display import FileLink, display
if list(ROOT.rglob('summary.json')):
    command('export')
    display(FileLink(str(ROOT.with_name(ROOT.name + '_results.zip'))))
else:
    print('No model jobs completed yet; enable a worker cell above.')
''')]
    return cells


def build(output_dir, preset="pilot"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {role: output_dir / f"jev_benchmark_v4_{role}.ipynb" for role in ROLES}
    for destination in destinations.values():
        if destination.exists():
            previous = nbformat.read(destination, as_version=4)
            if any(cell.get("outputs") for cell in previous.cells):
                raise ValueError("Refusing to overwrite an executed notebook")
    config = configuration(preset, suite="policy")
    study = output_dir / ("v4_inputs_" + digest({"config": config, "source": source_fingerprint()})[:16])
    prepare(study, config)
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in ("jevbench", "jevbench_v4", "tests/v4"):
            for path in sorted((ROOT / directory).glob("*.py")):
                archive.write(path, path.relative_to(ROOT))
        for name in ("scripts/__init__.py", "scripts/run_v4.py", "scripts/build_v4_notebook.py",
                     "tests/__init__.py", "tests/validate_v4.py", "requirements.txt", "requirements-dev.txt",
                     "requirements-v4-local.txt", "docs/v4.md", "docs/protocols/BENCHMARK_V4_PLAN.md"):
            archive.write(ROOT / name, name)
        for path in sorted(study.rglob("*")):
            if path.is_file():
                archive.write(path, "study/" + path.relative_to(study).as_posix())
    blob = source.getvalue()
    checksum = hashlib.sha256(blob).hexdigest()
    for role, destination in destinations.items():
        cells = notebook_cells(role, blob, checksum)
        notebook = nbformat.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "none" if role == "jev" else "gpu", "isInternetEnabled": True}})
        nbformat.validate(notebook)
        for cell in cells:
            if cell.cell_type == "code":
                compile(cell.source, "v4-notebook-cell", "exec")
        nbformat.write(notebook, destination)
        print(destination)
    (output_dir / "jev_benchmark_v4_source.zip").write_bytes(blob)
    return destinations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated")
    parser.add_argument("--preset", choices=("pilot", "study"), default="pilot")
    args = parser.parse_args()
    build(args.output_dir, args.preset)
