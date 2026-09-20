# Jev vs. classical machine learning

**[Read the interactive benchmark report](https://quicqdev.github.io/Jev-vs-ML/)** · [Download the release graphic](docs/assets/benchmark-release-blue.png)

A completed comparison of **Jev 1.13.0 against 11 conventional classification pipelines** on eight datasets. The published run uses protocol **3.0.1 / V3**, three training seeds, and one fixed test set per dataset.

**Finding:** Jev's strongest result is IMDb sentiment classification. It is competitive on AG News and SMS Spam, trails on Banking77, and does not outperform the strongest tested classical pipelines on the four tabular datasets. This is a bounded-budget comparison, not a comparison against best-possible ML or all modern language models.

## Results and artifacts

- [Completed notebook with saved outputs](notebooks/jev_benchmark_v3.ipynb)
- [Raw balanced-accuracy table](published_results/raw_balanced_accuracy.csv)
- [Threshold-adjusted balanced-accuracy table](published_results/adjusted_balanced_accuracy.csv)
- [Artifact provenance and SHA-256](published_results/manifest.json)
- [V3 source and notebook bundle](bundles/jev_benchmark_v3_bundle.zip)
- [V3 protocol and backend choices](docs/protocols/BENCHMARK_V3.md), with the [common V2 protocol](docs/protocols/BENCHMARK_V2.md)
- [Validation record](docs/validation/VALIDATION_V3.md)

### Threshold-adjusted balanced accuracy (%)

Binary thresholds are selected on separate labeled policy data for both Jev and classical pipelines. Multiclass decisions are unchanged. The table below shows means; the CSVs and notebook include sample SD across seeds.

| Dataset | Jev zero-shot prompts* | Jev few-shot prompts* | Best classical pipeline | Classical score |
|---|---:|---:|---|---:|
| AG News | 87.5 | 86.3 | SVM | 88.4 |
| Banking77 | 78.9 | 81.9 | SVM | 89.7 |
| SMS Spam | 95.9 | 95.8 | Naive Bayes | 96.3 |
| IMDb | **96.1** | 95.5 | Logistic regression | 88.3 |
| Bank Marketing | 59.7 | 59.0 | Voting ensemble | 73.3 |
| Online Shoppers | 51.8 | 53.6 | Voting ensemble | 71.2 |
| Breast Cancer | 88.4 | 92.7 | Logistic regression | 100.0 |
| Iris | 97.0 | 94.5 | Several pipelines tied | 100.0 |

*For binary tasks, threshold-adjusted Jev uses labeled policy data and is **not zero-shot end to end**. Few-shot prompts contain one training example per class. The best classical column is selected retrospectively from the displayed test means; it describes this benchmark, not a model-selection procedure for deployment.

### Raw decisions

Without policy threshold adjustment, Jev zero-shot scores **96.3% on IMDb**, versus **88.4%** for the strongest classical pipeline (+7.9 percentage points). On SMS Spam, its raw score is **96.1% versus 95.0%**; classical models catch up after threshold tuning. Both complete panels are published to avoid presenting only the more favorable decision rule.

## What the experiment measures

- Eight datasets: AG News, Banking77, SMS Spam, IMDb, Bank Marketing, Online Shoppers, Breast Cancer, and Iris.
- Eleven classical families: logistic regression, SVM, decision tree, random forest, extra trees, k-NN, Naive Bayes, histogram gradient boosting, XGBoost, CatBoost, and voting ensemble.
- Training seeds 2027/2028/2029; holdout seed 20260920. All models share each dataset's test cases.
- Up to 8,000 training rows, 1,000 model-selection rows, and 500 independent policy rows; four candidates per classical family. Small datasets use fewer rows.
- Test sizes: 1,000 for most datasets, 1,500 for Banking77, 114 for Breast Cancer, and 30 for Iris.
- Kaggle two-T4 configuration, with explicitly mixed CPU/GPU backends. Classical models learn from task labels; Jev uses a pretrained API model with task and class descriptions.

## Interpretation and limitations

- Means and sample SD across training seeds are **not confidence intervals**. Identical successful zero-shot requests are cached across seeds: zero SD does not establish independent API repeatability.
- The notebook includes Banking77 warnings about predictions outside the true label set. The adapter encodes failed requests as `-1` and counts them as errors. The exported run diagnostics are needed to quantify this effect; it should not be silently attributed entirely to model quality.
- Small Iris and Breast Cancer holdouts limit interpretation of their high scores. Public-dataset pretraining exposure is not established or ruled out here.
- This release contains the executed notebook, exact embedded source, and tables extracted from its saved outputs. The full Kaggle result ZIP, per-example predictions, dataset snapshots, run diagnostics, and computed paired bootstrap intervals were not provided with this notebook and are **not included**. The notebook's `/kaggle/working/...` links are paths from the original session, not downloadable files in this repository.
- No statistical-significance claim is made from the displayed means. No V3 latency or throughput conclusion is claimed here. Earlier V1 latency results belong to a different experiment.
- Few-shot examples help some datasets and hurt others. These results support task-specific evaluation, not a universal superiority claim.

## Reproduce on Kaggle

1. Upload `notebooks/jev_benchmark_v3.ipynb` to a fresh Kaggle session. Enable Internet and two T4 GPUs; use the latest GPU environment.
2. Add and enable the Kaggle secret `TYPESAFE_API_KEY`.
3. Choose a new output `ROOT` if changing the configuration. The notebook already contains the exact source used for the published run.
4. Run extraction, setup, preparation, and classical ML cells first. Run the separate Jev cell when ready to make paid API calls.
5. Save the generated results ZIP before ending the session. It contains detailed diagnostics and predictions beyond the tables displayed in the notebook.

The saved run reports 38,922 request attempts. Its approximately $4.19 input-cost value is a configured estimate, **not an invoice or total billed cost**. Request counts, retries, cache reuse, prices, and dataset snapshots can change on a new run. See the notebook's budget controls.

## Local artifact checks and development

Python 3.12 is the version recorded by the completed notebook. A GPU is not required to inspect saved outputs or run these artifact checks:

```bash
python -m pip install -r requirements-dev.txt
python -m tests.validate_v3_artifact
python -m tests.validate_v3_backends
python -m tests.validate_gpu_process
```

The backend checks use doubles and the process checks do not perform CUDA computation. They do not substitute for the notebook's actual Kaggle GPU probes.

To regenerate the CSVs from the saved outputs, run `python -m scripts.export_published_results`. To build a fresh unexecuted notebook from editable source, run `python -m scripts.build_modular_notebook --v3`. Generated notebooks and bundles go into the ignored `generated/` directory; the builder refuses to overwrite a notebook containing outputs.

`.env`, virtual environments, raw local results, request caches under `results/`, and generated result archives are excluded from Git. Do not put credentials in notebook cells.

## Report website

The GitHub Pages report is served from `main` / `docs`. It includes raw and adjusted comparisons, text/tabular filters, the full score table, methodology, limitations, and downloadable results. No external JavaScript libraries, analytics, or API keys are required.

Edit `docs/report.template.html` and `docs/assets/report.css` for the presentation. Run `python -m scripts.build_site` to regenerate `docs/index.html`, chart data, and downloadable CSVs from `published_results/`. The published HTML includes the raw table and chart even without JavaScript. Interactive controls are in `docs/assets/report.js`.

The release graphic was made with the built-in image generation tool; the exact prompt and verification notes are in [image provenance](docs/image-generation.md). Its numbers were checked against the raw result panel. The website's bar charts are rendered directly from CSV values.

## Earlier experiments

[V1 notebook](notebooks/jev_classification_benchmark.ipynb), [V1 Kaggle notes](docs/history/KAGGLE_README.md), [V2 notebook](notebooks/jev_benchmark_v2.ipynb), and [audit notes](docs/history/BENCHMARK_AUDIT.md) are retained as historical artifacts. Their protocols and results should not be mixed with the published V3 tables. `examples/experiment.py` and `examples/purchase_intent.json` are an earlier API smoke-test runner and synthetic example; running them makes paid requests.
