import ast
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import nbformat

from scripts.build_v4_notebook import build


class NotebookTests(unittest.TestCase):
    def test_embedded_bundle_compiles_and_never_overwrites_saved_outputs(self):
        directory = Path(__file__).resolve().parents[2] / "generated"
        directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            paths = build(temporary)
            path = paths["jev"]
            notebook = nbformat.read(path, as_version=4)
            local = nbformat.read(paths["local"], as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                self.assertFalse(cell.get("outputs"))
                if cell.cell_type == "code":
                    compile(cell.source, "v4-cell", "exec")
            extraction = next(cell for cell in notebook.cells if "b64decode" in cell.source)
            tree = ast.parse(extraction.source)
            encoded = next(node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode")
            local_tree = ast.parse(next(cell.source for cell in local.cells if "b64decode" in cell.source))
            local_encoded = next(node.args[0].value for node in ast.walk(local_tree) if isinstance(node, ast.Call)
                                 and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode")
            self.assertEqual(encoded, local_encoded)
            local_commands = "\n".join(cell.source for cell in local.cells if cell.cell_type == "code" and "b64decode" not in cell.source)
            jev_commands = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code" and "b64decode" not in cell.source)
            self.assertNotIn("'--provider', 'jev'", local_commands)
            self.assertNotIn("command('baselines')", jev_commands)
            self.assertNotIn("command('prepare'", jev_commands + local_commands)
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as bundle:
                self.assertIn("jevbench_v4/providers.py", bundle.namelist())
                self.assertIn("tests/validate_v4.py", bundle.namelist())
                self.assertNotIn(".env", bundle.namelist())
                run = json.loads(bundle.read("study/run.json"))
                self.assertEqual(run["config"]["datasets"], ["Support Policy"])
                self.assertIn("study/data/Support_Policy/snapshot.parquet", bundle.namelist())
                for name in bundle.namelist():
                    if name.endswith(".py"):
                        compile(bundle.read(name), name, "exec")
            extraction.outputs = [nbformat.v4.new_output("stream", name="stdout", text="retained")]
            nbformat.write(notebook, path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                build(temporary)
            self.assertEqual(path.read_bytes(), before)
            # Either saved worker must block regeneration of the whole pair.
            extraction.outputs = []
            nbformat.write(notebook, path)
            local.cells[1].outputs = [nbformat.v4.new_output("stream", name="stdout", text="retained")]
            nbformat.write(local, paths["local"])
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                build(temporary)
            self.assertEqual(path.read_bytes(), before)
