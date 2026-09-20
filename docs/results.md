# Results and interpretation

[Repository overview](../README.md) · [Interactive report](https://quicqdev.github.io/Jev-vs-ML/)

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

Both full panels are available as [raw CSV](../published_results/raw_balanced_accuracy.csv) and [adjusted CSV](../published_results/adjusted_balanced_accuracy.csv). The [completed notebook](../notebooks/jev_benchmark_v3.ipynb) retains the saved outputs.
