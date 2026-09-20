# Jev vs. classical machine learning

**[Interactive report](https://quicqdev.github.io/Jev-vs-ML/)** · [Completed notebook](notebooks/jev_benchmark_v3.ipynb) · [Results and limitations](docs/results.md)

A comparison of **Jev 1.13.0 and 11 classical classification pipelines** across eight datasets, using three training seeds and a shared test set. The published run uses protocol **3.0.1 (V3)**.

Jev's strongest result is IMDb sentiment classification: **96.3% raw balanced accuracy**, compared with **88.4%** for the best classical pipeline in this run. Results elsewhere are mixed; classical pipelines lead on all four tabular datasets. This is a bounded-budget comparison.

## Start here

| If you want to… | Open |
|---|---|
| Explore the scores | [Interactive report](https://quicqdev.github.io/Jev-vs-ML/) |
| Read the findings and caveats | [Results and interpretation](docs/results.md) |
| Inspect the executed run | [V3 notebook](notebooks/jev_benchmark_v3.ipynb) |
| Download the tables | [Raw CSV](published_results/raw_balanced_accuracy.csv) · [Adjusted CSV](published_results/adjusted_balanced_accuracy.csv) |
| Understand the methodology | [V3 protocol](docs/protocols/BENCHMARK_V3.md) · [Common V2 protocol](docs/protocols/BENCHMARK_V2.md) |
| Check provenance | [Manifest and hashes](published_results/manifest.json) · [Validation record](docs/validation/VALIDATION_V3.md) |
| Download the frozen source | [V3 release bundle](bundles/jev_benchmark_v3_bundle.zip) |
| Work on the repository | [Development guide](docs/development.md) |

Threshold-adjusted Jev uses labeled policy data for binary tasks, so those scores are not zero-shot end to end. Seed standard deviations are not confidence intervals. The full Kaggle result archive and per-example diagnostics are unavailable in this release; see the [full limitations](docs/results.md#interpretation-and-limitations).

## Reproduce on Kaggle

1. Upload [the V3 notebook](notebooks/jev_benchmark_v3.ipynb) to a fresh session. Enable Internet and two T4 GPUs; select the latest GPU environment.
2. Add and enable the Kaggle secret `TYPESAFE_API_KEY`.
3. Choose a new output `ROOT` if changing the configuration. The notebook contains the exact source used for the published run.
4. Run extraction, setup, preparation, and classical ML cells first. The separate Jev cell makes paid API calls.
5. Save the generated results ZIP before ending the session. It includes diagnostics and predictions beyond the displayed tables.

The saved run reports 38,922 request attempts. Its approximately $4.19 input-cost value is a configured estimate, **not an invoice or total billed cost**. Counts, retries, cache reuse, prices, and snapshots can change on a new run; use the notebook's budget controls.

## Repository layout

```text
jevbench/           Benchmark implementation used by V2 and V3
notebooks/          Published V3 notebook and historical V1/V2 notebooks
bundles/            Original downloadable source and notebook archives
published_results/  Tables extracted from the completed run and provenance
scripts/            Notebook builders, result export, report build, diagnostics
tests/              Artifact, backend, process, and integration checks
docs/               Report website, protocols, results, and historical notes
legacy/             Original V1 benchmark implementation
examples/           Standalone API smoke test and sample request
```

## Local checks

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m tests.validate_v3_artifact
python -m tests.validate_v3_backends
python -m tests.validate_gpu_process
```

These checks need no API key or GPU. Python 3.12 is recorded in the completed notebook. See the [development guide](docs/development.md) for build commands, additional checks, and the source-module map.

The [notebook index](notebooks/README.md) explains the earlier experiments. All historical artifacts are retained. Fresh builds go into ignored `generated/`; local results and caches stay in ignored `results/`.
