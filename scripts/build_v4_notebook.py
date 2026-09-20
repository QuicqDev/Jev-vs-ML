"""Build a self-contained Kaggle notebook for the V4 continuity pilot."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "jev_benchmark_v4_pilot.ipynb"
    if destination.exists():
        previous = nbformat.read(destination, as_version=4)
        if any(cell.get("outputs") for cell in previous.cells):
            raise ValueError("Refusing to overwrite an executed notebook")
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in ("jevbench", "jevbench_v4", "tests/v4"):
            for path in sorted((ROOT / directory).glob("*.py")):
                archive.write(path, path.relative_to(ROOT))
        for name in ("scripts/__init__.py", "scripts/run_v4.py", "scripts/build_v4_notebook.py",
                     "tests/__init__.py", "tests/validate_v4.py", "requirements.txt", "requirements-dev.txt",
                     "requirements-v4-local.txt", "docs/v4.md", "docs/protocols/BENCHMARK_V4_PLAN.md"):
            archive.write(ROOT / name, name)
    blob = source.getvalue()
    checksum = hashlib.sha256(blob).hexdigest()
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [markdown("""# Jev V4 — continuity compatibility pilot

This is the first V4 implementation stage, with **no saved benchmark results**.
It covers IMDb, Banking77, and Bank Marketing. Read `docs/v4.md` after extraction
for completed work and remaining study components.

Use a fresh Kaggle session with Internet and a GPU. Local providers run in separate
processes on one GPU at a time. Start with development-only pilots; inspect unsupported
inputs before freezing a final study. Complete inputs and every class description are
preserved, so long inputs can be unsupported by a local checkpoint.

The Jev cells require an enabled `TYPESAFE_API_KEY` Kaggle secret and make API calls
only when their explicit switch is enabled. The notebook initially enables preparation
and offline validation; model execution is selected in later cells."""),
        code(f'''import base64, hashlib, io, sys, zipfile
from pathlib import Path

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
'''), markdown("## Prepare fresh splits\n\nThis downloads public datasets and freezes full input snapshots. It does not call a model. Use a new root when changing code or configuration."),
        code('''ROOT = Path('/kaggle/working/jev_benchmark_v4_pilot')

def command(action, *options):
    subprocess.check_call([sys.executable, '-m', 'scripts.run_v4', action,
                           '--root', str(ROOT), *options], cwd=CODE_DIR)

command('prepare', '--preset', 'pilot')
'''), markdown("## Local decision-model compatibility\n\nPins the inspected Von and Laya SDK source commits and weight revisions. GPU compatibility and package versions still need to be recorded on Kaggle. Each provider gets 50 development cases per dataset; this is not a test result."),
        code('''RUN_LOCAL_MODELS = False
if RUN_LOCAL_MODELS:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE_DIR / 'requirements-v4-local.txt')])
    for provider in ('von', 'laya'):
        command('pilot', '--provider', provider, '--device', 'cuda:0')
'''), markdown("## Jev compatibility — API calls\n\nRequest and elapsed-time ceilings apply. No inherited V3 monetary estimate cap is used. A running request may finish after the elapsed-time ceiling."),
        code('''RUN_JEV = False
if RUN_JEV:
    command('pilot', '--provider', 'jev', '--max-attempts', '1000', '--max-seconds', '1800')
'''), markdown("## Reference baselines\n\nRaw majority and SVM comparisons, plus CatBoost on Bank Marketing. Four SVM/CatBoost candidates use selection data; policy/test data are excluded from fitting. AutoML, embeddings, adjusted panels, and paired inference remain later work."),
        code('''RUN_BASELINES = False
if RUN_BASELINES:
    command('baselines')
'''), markdown("## Export completed jobs\n\nSaved summaries are recomputed from per-case predictions before export. Download the ZIP before ending the Kaggle session. No output from this pilot should be presented as a completed V4 study."),
        code('''from IPython.display import FileLink, display
if list(ROOT.rglob('summary.json')):
    command('export')
    display(FileLink(str(ROOT.with_name(ROOT.name + '_results.zip'))))
else:
    print('No model jobs completed yet; enable a pilot cell above.')
''')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}})
    nbformat.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code":
            compile(cell.source, "v4-notebook-cell", "exec")
    nbformat.write(notebook, destination)
    (output_dir / "jev_benchmark_v4_source.zip").write_bytes(blob)
    print(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated")
    build(parser.parse_args().output_dir)
