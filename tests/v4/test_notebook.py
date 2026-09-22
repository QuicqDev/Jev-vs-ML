import ast
import base64
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import nbformat

from scripts.build_v4_notebook import build


class NotebookTests(unittest.TestCase):
    def test_single_notebook_embeds_complete_runnable_source(self):
        generated = Path(__file__).resolve().parents[2] / "generated"
        generated.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=generated) as temporary:
            path = build(Path(temporary) / "jev_benchmark_v4.ipynb")
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            self.assertEqual(path.name, "jev_benchmark_v4.ipynb")
            for cell in notebook.cells:
                self.assertFalse(cell.get("outputs"))
                if cell.cell_type == "code":
                    compile(cell.source, "v4-cell", "exec")
            executable = [cell for cell in notebook.cells if cell.cell_type == "code"]
            self.assertIn("b64decode", executable[0].source)
            tree = ast.parse(executable[0].source)
            encoded = next(node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode")
            source = "\n".join(cell.source for cell in executable[1:])
            self.assertIn("PRESET = \"study\"", source)
            self.assertIn("command('all-providers'", source)
            self.assertIn("'0', '1', '--min-gpus', '2'", source)
            self.assertIn("command('baselines'", source)
            self.assertNotIn("jev_benchmark_v4_local", source)
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as bundle:
                names = set(bundle.namelist())
                for required in ("jevbench_v4/temporal.py", "jevbench_v4/iterative.py",
                                 "requirements-v4-kaggle.txt", "scripts/run_v4.py"):
                    self.assertIn(required, names)
                self.assertFalse(any(name.startswith("study/") for name in names))
                for name in names:
                    if name.endswith(".py"):
                        compile(bundle.read(name), name, "exec")
            executable[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="retained")]
            nbformat.write(notebook, path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                build(path)
            self.assertEqual(path.read_bytes(), before)
