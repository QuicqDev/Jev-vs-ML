# Run the Jev classification benchmark on Kaggle

Upload **jev_classification_benchmark.ipynb** to a new Kaggle notebook. It is fully self-contained; no other project files or local `.env` are needed.

1. Enable Internet in notebook settings.
2. Select an NVIDIA GPU accelerator.
3. Add a Kaggle Secret named `TYPESAFE_API_KEY` and enable access for this notebook.
4. Choose `PRESET = "quick"` for a first pipeline check, or keep `"benchmark"` for the three-seed experiment.
5. Run All. This makes paid TypeSafe API calls. Download the generated ZIP from the final cell or Output panel.

## Included comparison

Eight datasets: AG News, Banking77, SMS Spam, IMDb, Bank Marketing, Online Shoppers, Breast Cancer Wisconsin, Iris.

Eleven ML columns: logistic regression, SVM, decision tree, random forest, extra trees, k-NN, naive Bayes, histogram gradient boosting, XGBoost, CatBoost, fixed voting ensemble.

Two Jev columns: zero-shot and few-shot (one training example per class by default).

On T4×2, up to two independent dataset/seed workers run in isolated processes, one per GPU. GPU accelerates XGBoost and CatBoost. Other baselines run on CPU with a bounded per-worker thread budget. Jev runs on TypeSafe infrastructure. Each GPU is tested separately; failed devices are excluded and zero usable devices triggers CPU fallback. GPU activity is intermittent during CPU models/preprocessing; a 2× speedup is not guaranteed. Set `CFG['max_parallel_jobs'] = 1` before starting to disable parallel jobs.

The default performs two candidate settings per base model, selected on validation balanced accuracy, with final refitting on train + validation. The fixed ensemble reuses fitted members. Defaults cap training at 8,000, validation at 1,000 and test at 300 examples per dataset/seed, across seeds 42, 43 and 44. Small datasets have fewer rows. Approximate nominal request counts: 11,664 default or 1,460 quick, before cache reuse and retries.

## Outputs

`/kaggle/working/jev_benchmark_parallel_results.zip` contains:

- `report.html` and `balanced_accuracy_heatmap.png`
- accuracy, balanced accuracy and macro-F1 tables (dataset rows, model columns)
- `scores_by_seed.csv` and `predictions.csv`
- individual model settings, validation scores, fit times and latency measurements
- exact splits, dataset snapshots/source hashes and package versions
- resumable Jev response cache and API attempt ledger

Means and standard deviations across seeds are reported; they are not confidence intervals. Rankings only include datasets completed for every model and seed. Dataset examples are included in snapshots; check source licensing before publishing the full bundle. The key is not saved.

## Resume

Rerunning cells in the same session uses completed dataset/seed results and successful Jev response caches. Across sessions, attach a prior output folder from the same notebook version as a Kaggle input and set `RESUME_FROM` to the folder containing `run.json`. Configuration, implementation and package versions must match; changed experiments should use a new output directory. The parallel notebook uses `/kaggle/working/jev_benchmark_parallel` to preserve the earlier sequential notebook's outputs. A ZIP can be extracted in a separate cell before setting that path.

## Scope and validation

The notebook documents all feature representations, dropped fields and training regimes. On text, linear models and naive Bayes use TF-IDF; other classifiers use TF-IDF/SVD. This is a bounded practical benchmark, not maximal tuning. Famous public data may overlap Jev pretraining. API latency and local ML latency are deployment measurements, not equivalent hardware comparisons. ML timings are collected under concurrent training load; use a separate single-worker experiment for isolated timings.

The notebook was validated locally for schema and code compilation, all eight public dataset loaders/splits, all 11 models on tiny text and tabular inputs, and mocked Jev response/caching/report integration. The parallel revision additionally passed real two-process execution with notebook-defined functions, explicit GPU-ID routing checks, and checkpoint reuse. Actual dual-GPU execution has not been tested locally; it is probed when the notebook starts on Kaggle. It has not been executed on Kaggle or through a full paid Jev benchmark. No scores in the notebook are fabricated or pre-filled.

Developer files: `kaggle_benchmark.py` contains the implementation, `build_notebook.py` embeds it into the standalone notebook, `validate_notebook.py` runs offline integration checks, and `validate_parallel.py` exercises the process scheduler. Only the `.ipynb` is needed on Kaggle.
# Current version

Use [jev_benchmark_v3.ipynb](jev_benchmark_v3.ipynb) and [BENCHMARK_V3.md](BENCHMARK_V3.md) for the faster cuML-enabled comparison. The instructions below describe the original pilot notebook.
