"""Build a self-contained Kaggle launcher plus an editable Python module bundle."""
from pathlib import Path
import base64
import hashlib
import io
import zipfile
import nbformat as nbf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--v3', action='store_true')
parser.add_argument('--output-dir', default='generated',
                    help='Destination for fresh notebooks and bundles; published runs are never overwritten.')
args = parser.parse_args()
notebook_name = 'jev_benchmark_v3.ipynb' if args.v3 else 'jev_benchmark_v2.ipynb'
bundle_name = 'jev_benchmark_v3_bundle.zip' if args.v3 else 'jev_benchmark_v2_bundle.zip'

ROOT = Path(__file__).resolve().parent
OUTPUT = (ROOT / args.output_dir).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
destination = OUTPUT / notebook_name
if destination.exists():
    existing = nbf.read(destination, as_version=4)
    if any(c.get('outputs') for c in existing.cells):
        raise SystemExit('Refusing to overwrite an executed notebook. Choose another --output-dir.')
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((ROOT / 'jevbench').glob('*.py')):
        archive.writestr('jevbench/' + path.name, path.read_bytes())
    for name in ['requirements-v2.txt', 'BENCHMARK_V2.md', 'BENCHMARK_V3.md']:
        archive.writestr(name, (ROOT / name).read_bytes())
blob = buffer.getvalue()
encoded, checksum = base64.b64encode(blob).decode(), hashlib.sha256(blob).hexdigest()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))
md('''# Jev versus eleven conventional classification pipelines — v2

Use a **fresh Kaggle session**, Internet enabled, **T4 × 2**, and an enabled secret named `TYPESAFE_API_KEY`.

This notebook extracts its bundled Python modules; no separate code upload is required. The full editable bundle is also available. All ML tuning uses development data. One fixed test set is shared across models and training seeds. The previous run should be retained as a pilot.

**Run the ML cells first.** The separate Jev cell makes paid API requests. Full mode plans about 52,410 requests before cache reuse/retries on the current snapshots. Check the displayed counts and your pricing. The estimate guard is not a billing cap.

This is a modest-budget comparison, not a claim of best achievable ML performance.''')
code(f'''# Extract the frozen source package. Run before importing numerical libraries.
import base64, hashlib, io, sys, zipfile
from pathlib import Path

# Optional: point at an edited bundle uploaded as a Kaggle Dataset.
# The directory must contain jevbench/, requirements-v2.txt, and BENCHMARK_V2.md.
EXTERNAL_CODE_DIR = None
PACKAGE_SHA256 = {checksum!r}
if EXTERNAL_CODE_DIR:
    CODE_DIR = Path(EXTERNAL_CODE_DIR)
else:
    bundled = base64.b64decode({encoded!r})
    assert hashlib.sha256(bundled).hexdigest() == PACKAGE_SHA256
    CODE_DIR = Path('/kaggle/working') / ('jevbench_code_' + PACKAGE_SHA256[:12])
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(bundled)) as archive:
        archive.extractall(CODE_DIR)
assert (CODE_DIR / 'jevbench' / 'runner.py').exists()
sys.path.insert(0, str(CODE_DIR))
print('Source modules:', CODE_DIR)''')
code('''import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE_DIR / 'requirements-v2.txt')])
# If you imported numpy/sklearn before installation, restart the session and rerun.
''')
md('''## Freeze the run configuration

`benchmark` uses three seeds, four candidates, up to 400 trees, 12,000 training rows, 1,500 model-selection rows, and 500 decision-policy rows. `quick` is for smoke testing only. Test caps are 1,000 per dataset and 1,500 for Banking77; small datasets retain fewer rows.

Default pilot exclusion assumes seeds 42/43/44, test cap 300 and matching snapshots. If your pilot differed, use its actual saved IDs in `datasets.py` before running. Changing the configuration, source or environment requires a new `ROOT`.''')
code('''from IPython.display import display, Markdown, FileLink
from jevbench.config import configuration
from jevbench.runner import prepare_suite, run_ml, run_jev
from jevbench.reporting import render_results

CFG = configuration('benchmark')
ROOT = Path('/kaggle/working/jev_benchmark_v2')
# Optional existing pilot output directory: reuses data snapshots, not pilot scores.
PILOT_ROOT = None
# Example: PILOT_ROOT = Path('/kaggle/input/my-pilot-results/jev_benchmark_parallel')

# Two T4s: independent worker per GPU; CPU models use two threads per worker.
CFG['max_parallel_jobs'] = 2
CFG['threads'] = 2
CFG['jev_model'] = 'jev-1.13.0'
display(CFG)
display(Markdown((CODE_DIR / 'BENCHMARK_V2.md').read_text()))''')
code('''OVERVIEW = prepare_suite(ROOT, CFG, pilot_root=PILOT_ROOT)
display(OVERVIEW)
''')
md('''## Fit and tune conventional models

GPU probes precede training. Successful per-model checkpoints allow rerunning this cell after interruption. Individual trial warnings/errors are saved under `runs/`; inspect them before publishing. Most scikit-learn models use CPUs. Larger text forests can take hours.''')
code('''ML_STATUS = run_ml(ROOT, CFG)
display(ML_STATUS)
TABLES = render_results(ROOT, CFG)
display(TABLES['raw_balanced_accuracy'])
display(TABLES['adjusted_balanced_accuracy'])''')
md('''## Run Jev — paid API calls

The secret is read from Kaggle Secrets. tqdm shows each test/policy partition; exact successful requests are cached. Permanent API errors stop execution; exhausted transient retries count as failed predictions. Model IDs, request attempts and cache hits are recorded.

Binary **adjusted Jev uses labeled policy data to select a threshold**. Only the raw zero-shot panel is an end-to-end zero-shot baseline. Few-shot examples are sampled from training, one per class. Budget estimates exclude output-token charges; verify current pricing and account limits before executing this cell.''')
code('''API_STATUS = run_jev(ROOT, CFG)
display(API_STATUS)''')
md('''## Compare and export

Rows are datasets; columns are models. Values are mean ± sample SD across training seeds, **not confidence intervals**. Both panels use the same test cases. The report includes accuracy/macro-F1 and paired bootstrap differences; JSON records include minority recall, confusion matrices, thresholds and raw probabilities. Publish both raw and adjusted panels with the protocol.

Save/download the output ZIP to preserve caches and resume later. It contains benchmark artifacts and dataset snapshots, never the API key.''')
code('''TABLES = render_results(ROOT, CFG)
display(Markdown('### Raw decision rules'))
display(TABLES['raw_balanced_accuracy'])
display(Markdown('### Binary thresholds selected on policy data'))
display(TABLES['adjusted_balanced_accuracy'])
display(FileLink(str(ROOT / 'report.html')))
import shutil
# Include the source/protocol used in the result download for reproducibility.
shutil.copytree(CODE_DIR / 'jevbench', ROOT / 'source' / 'jevbench', dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__'))
for filename in ['BENCHMARK_V2.md', 'requirements-v2.txt']:
    shutil.copy2(CODE_DIR / filename, ROOT / filename)
archive = shutil.make_archive(str(ROOT) + '_results', 'zip', root_dir=ROOT.parent, base_dir=ROOT.name)
display(FileLink(archive))''')
if args.v3:
    cells.insert(1, nbf.v4.new_markdown_cell('''## Fast preset — reduced CPU workloads
All 11 models and four candidates remain. Tree ceilings, text dimensions and training caps are reduced; accuracy may change. GPU models run first. Other sklearn models use CPUs. There is no measured 30-minute guarantee.
cuML accelerates compatible estimators; remaining cases have explicit CPU labels.
Read BENCHMARK_V3.md displayed below for the exact differences. Use a NEW output directory and preserve the old results.'''))
    for cell in cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace("configuration('benchmark')", "configuration('v3')")
            cell.source = cell.source.replace("Path('/kaggle/working/jev_benchmark_v2')", "Path('/kaggle/working/jev_benchmark_v3_1')")
            cell.source = cell.source.replace('PILOT_ROOT = None', "PILOT_ROOT = Path('/kaggle/working/jev_benchmark_v3')\nif not PILOT_ROOT.exists():\n    PILOT_ROOT = None")
            cell.source = cell.source.replace("display(Markdown((CODE_DIR / 'BENCHMARK_V2.md').read_text()))", "display(Markdown((CODE_DIR / 'BENCHMARK_V3.md').read_text()))")
            cell.source = cell.source.replace("['BENCHMARK_V2.md', 'requirements-v2.txt']", "['BENCHMARK_V2.md', 'BENCHMARK_V3.md', 'requirements-v2.txt']")
            if "CFG = configuration('v3')" in cell.source:
                cell.source = cell.source.replace("CFG['threads'] = 2", "CFG['threads'] = 2\nCFG['jev_workers'] = 8  # baked-in default\nCFG['min_request_interval'] = 0.15")
            if 'subprocess.check_call' in cell.source:
                cell.source += "\n# CUDA imports and actual GPU checks happen ONLY in isolated child processes.\nimport importlib.util\nif importlib.util.find_spec('cuml') is None:\n    raise RuntimeError('cuML missing: select Kaggle latest GPU environment.')\n"
        else:
            cell.source = cell.source.replace('— v2', '— v3')
            cell.source = cell.source.replace('`benchmark` uses three seeds, four candidates, up to 400 trees, 12,000 training rows, 1,500 model-selection rows, and 500 decision-policy rows.', '`v3` uses three seeds, four candidates, up to 150 trees (60 histogram iterations), 8,000 training rows, 1,000 model-selection rows, and 500 decision-policy rows.')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True}})
nbf.validate(nb)
for i, cell in enumerate(cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'cell_{i}', 'exec')
nbf.write(nb, destination)
with zipfile.ZipFile(OUTPUT / bundle_name, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((ROOT / 'jevbench').glob('*.py')):
        archive.write(path, 'jevbench/' + path.name)
    for name in [notebook_name, 'requirements-v2.txt', 'BENCHMARK_V2.md', 'BENCHMARK_V3.md',
                 'build_modular_notebook.py', 'validate_v2.py', 'validate_tabular_v2.py', 'VALIDATION_V2.md',
                 'validate_v3_backends.py', 'profile_v3_histogram.py', 'VALIDATION_V3.md', 'validate_gpu_process.py']:
        archive.write(destination if name == notebook_name else ROOT / name, name)
print(destination)
print('Embedded package SHA256:', checksum)
