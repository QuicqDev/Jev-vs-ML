"""Map the frozen Kaggle package names to the current repository layout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Notebook packages retain their original flat names so saved runs stay portable.
FROZEN_INPUTS = {
    'requirements-v2.txt': ROOT / 'requirements.txt',
    'BENCHMARK_V2.md': ROOT / 'docs/protocols/BENCHMARK_V2.md',
    'BENCHMARK_V3.md': ROOT / 'docs/protocols/BENCHMARK_V3.md',
}


def frozen_source_path(name):
    return FROZEN_INPUTS.get(name, ROOT / name)
