# Development guide

[Repository overview](../README.md) · [Documentation index](README.md)

Run commands from the repository root. Python 3.12 is the version recorded in the completed notebook. Install the development dependencies with `python -m pip install -r requirements-dev.txt`; `requirements.txt` contains the benchmark runtime dependencies.

## Common tasks

For the new benchmark, see the [V4 pilot guide](v4.md). Run `python -m tests.validate_v4` for its offline suite and `python -m scripts.build_v4_notebook` to build the Kaggle pilot. V4 lives in `jevbench_v4/` so the frozen V3 source can still be validated.

| Task | Command |
|---|---|
| Validate the published notebook, source, tables, and bundle | `python -m tests.validate_v3_artifact` |
| Check backend routing with doubles | `python -m tests.validate_v3_backends` |
| Check isolated worker processes | `python -m tests.validate_gpu_process` |
| Extract the saved result tables | `python -m scripts.export_published_results` |
| Rebuild the report website | `python -m scripts.build_site` |
| Build a fresh V3 notebook and source bundle | `python -m scripts.build_modular_notebook --v3` |
| Build a fresh V2 notebook and source bundle | `python -m scripts.build_modular_notebook` |
| Build a fresh V1 notebook | `python -m scripts.build_notebook` |

The first three checks require neither a GPU nor API access. Backend checks use doubles; process checks start real child interpreters without CUDA computation. They do not replace Kaggle GPU probes.

Builders write into ignored `generated/` by default and accept `--output-dir`. They refuse to replace a notebook containing saved outputs. Exporting results reads saved notebook outputs without running cells or making API requests.

## Finding the implementation

| Module in `jevbench/` | Responsibility |
|---|---|
| `config.py` | Presets and experiment budgets |
| `datasets.py` | Dataset loading, snapshots, holdouts, and splits |
| `features.py` | Text and tabular feature preparation |
| `models.py`, `training.py` | Model candidates, fitting, and selection |
| `backends.py`, `gpu_process.py` | CPU/GPU routing and isolated workers |
| `api.py` | Jev requests, prompts, caching, and retries |
| `decisions.py` | Decision thresholds and metrics |
| `runner.py` | Suite preparation and execution |
| `reporting.py` | Tables and experiment reports |
| `common.py` | Shared utilities and device probing |

The original V1 implementation is retained in [legacy/kaggle_benchmark.py](../legacy/kaggle_benchmark.py). The small [API example](../examples/experiment.py) can be run with `python -m examples.experiment`; it uses [purchase_intent.json](../examples/purchase_intent.json) and makes paid requests. It is separate from the published benchmark.

## Additional validation and diagnostics

| Command | Scope and prerequisites |
|---|---|
| `python -m tests.validate_v2` | V2 integration, all dataset splits, real CPU fits, and mocked API |
| `python -m tests.validate_v2 --v3` | V3 integration using CPU fits and reduced test budgets |
| `python -m tests.validate_tabular_v2` | Diagnostics on old pilot holdouts; requires local Bank Marketing and Online Shoppers snapshots |
| `python -m tests.validate_notebook` | V1 notebook schema, model fits, mocked API, and reporting |
| `python -m tests.validate_parallel` | V1 two-process scheduling and checkpoint reuse |
| `python -m tests.validate_sampling` | V1 sampling regressions; requires all eight local snapshots |
| `python -m scripts.audit_tabular` | V1 threshold diagnostics on local tabular snapshots |
| `python -m scripts.profile_v3_histogram` | Banking77 histogram training profile |

Dataset-dependent checks can download missing public datasets. Cached snapshots are read from `results/notebook_data_validation/`; diagnostics that explicitly require those snapshots do not create them. Legacy V1 report checks also need `matplotlib` (`python -m pip install matplotlib`). None of these validation commands makes real Jev requests.

## Published artifacts and provenance

The notebooks in [notebooks/](../notebooks/README.md), ZIPs in [bundles/](../bundles/README.md), and historical validation records are retained from their original releases. V1/V2 protocols and results should not be mixed with the published V3 tables.

The frozen notebook package still uses the original names `requirements-v2.txt`, `BENCHMARK_V2.md`, and `BENCHMARK_V3.md`. [scripts/paths.py](../scripts/paths.py) maps these to the current repository locations. The runtime dependencies, protocols, and `jevbench/` source retain their original bytes. Original bundles keep their original internal layout and build tools; newly generated bundles use the current repository layout.

The [artifact check](../tests/validate_v3_artifact.py) verifies the notebook and release ZIP hashes, the embedded source against the repository and bundle, the notebook schema, saved output errors, and exported tables. Editing the benchmark source will intentionally fail its comparison with the published run; use a fresh generated notebook and a new results directory for a changed experiment.

Historical records under [validation/](validation/VALIDATION_V3.md) may show commands and filenames from the old layout. Use the commands on this page for the current checkout.

`.env`, virtual environments, local results, request caches, generated builds, and generated result archives are excluded from Git. Do not put credentials in notebook cells.

## Report website

GitHub Pages serves `main` / `docs`. Edit [report.template.html](report.template.html) and [assets/report.css](assets/report.css), then run `python -m scripts.build_site`. It regenerates `index.html`, chart data, and downloadable CSVs from `published_results/`. Interactive controls live in [assets/report.js](assets/report.js).

The report includes both decision panels, text/tabular filters, methodology, limitations, and downloadable tables. Its initial raw chart and table work without JavaScript. It uses no external JavaScript libraries, analytics, or API keys.

The [release graphic](assets/benchmark-release-blue.png) was made with the built-in image generation tool; [image provenance](image-generation.md) preserves the prompt and verification notes. Its numbers were checked against the raw result panel. Website charts are rendered directly from CSV values.
