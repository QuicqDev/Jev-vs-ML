"""Small atomic artifacts and an immutable run identity."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import uuid

from .contracts import canonical, digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if canonical(read_json(path)) != canonical(value):
            raise ValueError(f"Frozen manifest changed: {path}. Use a new run directory.")
    else:
        write_json(path, value)
    return value


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "requests", "torch", "transformers",
                 "von-sdk", "laya", "catboost", "autogluon.tabular", "sentence-transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "jevbench_v4").glob("*.py")) + sorted((root / "jevbench").glob("*.py"))
    return digest({p.relative_to(root).as_posix(): p.read_bytes().hex() for p in paths})
