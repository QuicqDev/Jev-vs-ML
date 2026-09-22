"""Build the single self-contained V4 Kaggle notebook."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def _bundle():
    names = []
    for directory in ("jevbench", "jevbench_v4", "tests/v4"):
        names.extend(path for path in (ROOT / directory).glob("*.py"))
    for name in ("scripts/__init__.py", "scripts/run_v4.py", "scripts/build_v4_notebook.py",
                 "tests/__init__.py", "tests/validate_v4.py", "requirements.txt",
                 "requirements-dev.txt", "requirements-v4-local.txt", "requirements-v4-kaggle.txt",
                 "docs/v4.md", "docs/protocols/BENCHMARK_V4_PLAN.md"):
        names.append(ROOT / name)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(names, key=lambda value: value.relative_to(ROOT).as_posix()):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 9, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return output.getvalue()


def notebook_cells(blob, checksum):
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    encoded = base64.b64encode(blob).decode()
    return [
        markdown("""# Jev Benchmark V4 — Reddit questions

This is one self-contained Kaggle notebook for the **new V4 experiments only**. It does not rerun V3.

It compares Jev with Von and Laya on the same frozen cases, adds classical and frozen-embedding controls, tests AutoGluon on leakage-resistant future prediction, and measures whether feedback helps in an iterative simulator. During provider evaluation, Jev runs concurrently with Von on T4 0 and Laya on T4 1. The notebook refuses CPU fallback for either local model.

Before running, select Kaggle's **2× T4** accelerator, enable Internet, and add `TYPESAFE_API_KEY` as a Kaggle secret. `PRESET = "study"` is the full registered design; use `"pilot"` only for a pipeline check."""),
        code(f'''# First executable cell: restore every source file and pinned requirement.
import base64, hashlib, io, sys, zipfile
from pathlib import Path

PACKAGE_SHA256 = {checksum!r}
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == PACKAGE_SHA256
CODE_DIR = Path('/kaggle/working') / ('jevbench_v4_code_' + PACKAGE_SHA256[:12])
CODE_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
print('Restored verified V4 source:', CODE_DIR)
'''),
        markdown("## Configuration"),
        code('''PRESET = "study"       # "study" = registered full sizes; "pilot" = pipeline check
RUN_PROVIDERS = True     # Jev + Von + Laya on the same frozen cases
RUN_BASELINES = True     # majority/SVM/embedding + temporal controls + AutoGluon
AUTOML_MINUTES = 30      # per temporal split, including the random-split diagnostic
ROOT = Path('/kaggle/working') / ('jev_benchmark_v4_' + PRESET)
print(ROOT)
'''),
        markdown("## Install the pinned environment and verify the packaged harness"),
        code('''import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                       str(CODE_DIR / 'requirements-v4-kaggle.txt')])
subprocess.check_call([sys.executable, '-m', 'tests.validate_v4'], cwd=CODE_DIR)
'''),
        markdown("""## Freeze the additional V4 study

This downloads UCI Bike Sharing once, derives only past/present features, fixes the high-demand threshold from the first training block, creates three forward windows with a 24-hour embargo, and creates a separately labelled random-holdout diagnostic. It also freezes paired policy cases and deterministic iterative episodes."""),
        code('''def command(action, *options):
    subprocess.check_call([sys.executable, '-m', 'scripts.run_v4', action,
                           '--root', str(ROOT), '--suite', 'full', *options], cwd=CODE_DIR)

if not (ROOT / 'run.json').exists():
    command('prepare', '--preset', PRESET)
command('verify')
'''),
        markdown("## Inspect the frozen design"),
        code('''import json, pandas as pd
run = json.loads((ROOT / 'run.json').read_text())
display(pd.DataFrame([
    {'track': 'Policy pairs', 'jobs': str(run['config']['jobs']['Support Policy']),
     'test observations': 2 * run['config']['pairs_per_partition']['test']},
    {'track': 'Future bike demand', 'jobs': '3 forward + 1 random diagnostic',
     'test observations': run['config']['temporal_test']},
    {'track': 'Iterative support', 'jobs': '4 controlled conditions',
     'test observations': run['config']['iterative_test_episodes']},
]))
display(pd.read_csv(ROOT / 'data/Support_Policy/review_sample.csv').head(12))
'''),
        markdown("""## Jev, Von, and Laya — concurrent provider run

The parent process starts Jev alongside two isolated CUDA workers. Von sees only physical GPU 0; Laya sees only physical GPU 1. Each local worker performs an FP16 CUDA test and verifies model tensors remain on its assigned card. All three evaluate the same frozen policy, temporal, and iterative inputs."""),
        code('''if RUN_PROVIDERS:
    command('all-providers', '--phase', 'evaluate', '--gpus', '0', '1', '--min-gpus', '2',
            '--max-attempts', '100000', '--max-seconds', '43200')
else:
    print('Provider evaluation disabled in Configuration.')
'''),
        markdown("""## New-task controls and AutoML

These are additions to V4. They do not rerun the V3 ML benchmark. The forward tasks use persistence, same-hour-last-week, RBF SVM, CatBoost, and AutoGluon. AutoGluon receives explicit past-to-future tuning data; random bagging, stacking, dynamic stacking, and threshold calibration are disabled. The policy task adds majority, TF-IDF SVM, and a frozen sentence-embedding logistic model."""),
        code('''if RUN_BASELINES:
    command('baselines', '--cpu-threads', '4', '--automl-minutes', str(AUTOML_MINUTES))
else:
    print('Baseline evaluation disabled in Configuration.')
'''),
        markdown("## Rebuild summaries and download the auditable result bundle"),
        code('''from IPython.display import FileLink, display
if list(ROOT.rglob('summary.json')):
    command('export')
    display(pd.read_csv(ROOT / 'summary.csv'))
    display(pd.read_csv(ROOT / 'comparisons_vs_jev.csv'))
    display(pd.read_csv(ROOT / 'iterative_comparisons_vs_jev.csv'))
    display(FileLink(str(ROOT.with_name(ROOT.name + '_results.zip'))))
else:
    print('No completed evaluations are available yet.')
''')]


def build(output=ROOT / "notebooks" / "jev_benchmark_v4.ipynb"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = nbformat.read(output, as_version=4)
        if any(cell.get("outputs") for cell in previous.cells):
            raise ValueError("Refusing to overwrite an executed notebook")
    blob = _bundle()
    checksum = hashlib.sha256(blob).hexdigest()
    cells = notebook_cells(blob, checksum)
    for index, cell in enumerate(cells):
        cell["id"] = f"v4-{index:02d}"
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "gpuType": "T4 x2"}})
    nbformat.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code":
            compile(cell.source, "jev-benchmark-v4-cell", "exec")
    nbformat.write(notebook, output)
    print(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks" / "jev_benchmark_v4.ipynb")
    args = parser.parse_args()
    build(args.output)
