"""Offline validation of the published run, source, bundle, and saved tables."""
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import nbformat
from export_published_results import saved_tables

ROOT = Path(__file__).resolve().parent
path = ROOT / 'jev_benchmark_v3.ipynb'
n = nbformat.read(path, as_version=4)
nbformat.validate(n)
code = '\n'.join(c.source for c in n.cells if c.cell_type == 'code')
assert "configuration('v3')" in code and "CFG['jev_workers'] = 8" in code
cell = next(c for c in n.cells if c.cell_type == 'code' and 'b64decode' in c.source)
tree = ast.parse(cell.source)
blob = base64.b64decode(next(t.args[0].value for t in ast.walk(tree)
    if isinstance(t, ast.Call) and isinstance(t.func, ast.Attribute) and t.func.attr == 'b64decode'))
checksum = next(t.value.value for t in tree.body if isinstance(t, ast.Assign)
    and any(isinstance(a, ast.Name) and a.id == 'PACKAGE_SHA256' for a in t.targets))
assert hashlib.sha256(blob).hexdigest() == checksum
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    assert archive.testzip() is None
    for name in archive.namelist():
        assert archive.read(name) == (ROOT / name).read_bytes(), name
for c in n.cells:
    if c.cell_type == 'code':
        compile(c.source, 'cell', 'exec')
        assert not any(o.output_type == 'error' for o in c.get('outputs', []))
for panel, expected in saved_tables(n):
    with (ROOT / 'published_results' / f'{panel}_balanced_accuracy.csv').open(encoding='utf-8', newline='') as f:
        assert list(csv.reader(f)) == expected
manifest = json.loads((ROOT / 'published_results/manifest.json').read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']
with zipfile.ZipFile(ROOT / 'jev_benchmark_v3_bundle.zip') as archive:
    assert archive.testzip() is None
    assert archive.read(path.name) == path.read_bytes()
    assert not any('.env' in name or '__pycache__' in name for name in archive.namelist())
    for name in archive.namelist():
        assert archive.read(name) == (ROOT / name).read_bytes(), name
print('PASS: executed notebook, source hashes, complete final panels, CSVs, manifest, and release bundle')
