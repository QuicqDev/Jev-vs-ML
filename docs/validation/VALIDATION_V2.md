# Local v2 verification — 2026-09-19

No real TypeSafe requests were made during v2 development. API tests used fake responses. Actual two-T4 execution still requires Kaggle; the local concurrency test used two CPU processes.

Passed checks:

- Both presets across all eight cached public dataset snapshots, all configured seeds: split disjointness, every-class coverage, fixed test IDs, pilot-test exclusion and few-shot examples restricted to training.
- All eleven models plus majority baseline, four candidates per tuned family, on Iris, Breast Cancer, synthetic multiclass text and synthetic mixed categorical/numeric binary data. Five-tree integration runs exercise both parameter settings and weighting options, early stopping/refit and native CatBoost categories. No candidate failures.
- Two independent joblib worker processes, bounded CPU threads, resumable model files.
- Mocked Jev zero/few-shot calls, tqdm, exact-request cache, key exclusion, policy thresholds, metrics, six matrices and paired bootstrap reporting.
- Failed API rows count as wrong; policy failure retains raw choices with a diagnostic status; missing probability values serialize as JSON null rather than NaN.
- Notebook schema and Python-cell compilation, embedded source equality, ZIP integrity and absence of secrets/outputs in the delivered bundle.

## Real-data sanity check on the old pilot holdout

This diagnostic reused the **old seed-42 pilot test rows**, not the new v2 test set. Training/model-selection/policy partitions were created from the pilot development rows, with the v2 four-candidate search and 400-tree ceiling. It is not the final benchmark or an independent performance estimate. No model settings were changed in response to these scores.

| Dataset | Model | Default balanced accuracy | Policy-adjusted balanced accuracy |
|---|---|---:|---:|
| Bank Marketing | Logistic regression | 57.82% | 67.17% |
| Bank Marketing | XGBoost | 58.01% | 68.81% |
| Bank Marketing | CatBoost | 56.58% | 71.13% |
| Online Shoppers | Logistic regression | 68.06% | 63.53% |
| Online Shoppers | XGBoost | 74.66% | 69.51% |
| Online Shoppers | CatBoost | 51.34% | 69.06% |

Threshold tuning is not guaranteed to improve a particular test set. Both protocols are reported; selecting between them after inspecting the test scores would bias the comparison.

Diagnostic script: `validate_tabular_v2.py`. Detailed local artifacts: `results/v2_old_holdout_diagnostic_340ee342/`. Final integration artifacts: `results/v2_validation_f2366bb3/`. These result directories are intentionally excluded from the distribution bundle. The final v2 comparison remains to be run in Kaggle.
