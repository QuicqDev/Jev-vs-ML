# Notebooks

[Repository overview](../README.md)

| Notebook | Status | Protocol |
|---|---|---|
| [V3](jev_benchmark_v3.ipynb) | Completed published run with saved outputs | [V3](../docs/protocols/BENCHMARK_V3.md) |
| [V2](jev_benchmark_v2.ipynb) | Historical experiment | [V2](../docs/protocols/BENCHMARK_V2.md) |
| [V1](jev_classification_benchmark.ipynb) | Original pilot notebook | [Kaggle guide](../docs/history/KAGGLE_README.md) |

All three original notebooks are preserved byte for byte. Each is self-contained for Kaggle. Use V3 to inspect or reproduce the published comparison; earlier versions have different protocols.

Build fresh notebooks into `generated/` using the [development commands](../docs/development.md). The [release bundles](../bundles/README.md) retain the original source downloads.

The [V4 compatibility pilot](../docs/v4.md) is built separately with `python -m scripts.build_v4_notebook`. Its unexecuted notebook goes into `generated/`; there are no published V4 results yet.
