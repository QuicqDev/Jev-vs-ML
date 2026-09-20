# Jev classification benchmark v2

Use `jev_benchmark_v2.ipynb` in a **fresh Kaggle session**, Internet on, T4 × 2, with the `TYPESAFE_API_KEY` secret enabled. The notebook includes the Python package and extracts it automatically. You do not need to upload individual modules. `jev_benchmark_v2_bundle.zip` contains the editable package, notebook, requirements and this protocol.

The default is the benchmark preset: 8 datasets, 3 training seeds, 11 ML models, an untuned majority baseline, and Jev zero/few-shot. Run ML first; inspect its saved diagnostics before starting the API cell. The API cell makes paid requests. Model alias `jev-1.13.0` is pinned and returned model identifiers are recorded. If your account does not expose it, explicitly choose a supported model ID and a new run directory.

## What changed

| Component | v2 protocol |
|---|---|
| Binary imbalance | Compare ordinary and balanced training weights; select thresholds for balanced accuracy on a separate policy set |
| Model search | Four fixed candidates per family; validation balanced accuracy selects the winner |
| Boosting | Up to 400 trees; XGBoost/CatBoost early stop on model-selection validation, then refit with the selected tree count |
| Linear text models | Word/bigram + character TF-IDF for logistic regression and SVM |
| Text trees | Up to 2,000 word TF-IDF features selected by chi-square using training labels; replaces the pilot's 64-dimensional SVD representation |
| Native categories | CatBoost receives categorical columns directly; other tabular models use imputation, scaling and one-hot encoding |
| Ensemble | Fixed equal-probability average of logistic regression, random forest and XGBoost; no test-based member selection |
| GPU use | Two isolated processes, each assigned one GPU, process independent dataset/seed jobs; XGBoost and CatBoost use the assigned device |
| API display | tqdm bars for test and binary policy calls, exact-request disk caching, bounded retries |

SVM is linear for text and RBF for tabular data. Naive Bayes is multinomial for text and Gaussian for tabular data. k-NN uses scaled SVD text features or scaled/one-hot tabular features. CPU estimators remain CPU estimators; each worker limits CPU threads. GPU availability is checked by actual XGBoost and CatBoost fits. A device that fails either check is excluded; no usable devices means CPU fallback. GPU training can be nondeterministic.

## Exact model-selection budget

The four candidates cross the two settings below with ordinary/balanced sample weights, in that order. k-NN uses four neighbor/weight combinations instead. This is a small, predeclared search, **not exhaustive optimization**. The quick preset tries only the first setting with both weighting choices, or the first two k-NN candidates; it is a smoke test.

| Model | Setting 1 | Setting 2 |
|---|---|---|
| Logistic regression | C=0.3 | C=3 |
| SVM | C=0.3 | C=3 |
| Decision tree | depth=10, minimum leaf=3 | unrestricted depth, minimum leaf=10 |
| Random forest / Extra trees | depth=16, minimum leaf=2, sqrt features | unrestricted depth, minimum leaf=1, 50% features |
| Multinomial NB | alpha=0.1 | alpha=1 |
| Gaussian NB | variance smoothing=1e-9 | variance smoothing=1e-6 |
| Histogram gradient boost | 15 leaves, learning rate=.05, L2=1 | 31 leaves, learning rate=.1, L2=5 |
| XGBoost | depth=3, learning rate=.05, minimum child weight=3, L2=3 | depth=6, learning rate=.1, minimum child weight=5, L2=5 |
| CatBoost | depth=4, learning rate=.05, L2=3 | depth=7, learning rate=.1, L2=5 |

XGBoost row/column subsampling is .9. Both forests use 400 trees. Histogram boosting uses 400 iterations; XGBoost/CatBoost have a 400-tree ceiling and 30-round early stopping. k-NN candidates are (5, distance), (15, distance), (31, distance), (15, uniform). All unspecified estimator parameters retain the recorded library-version defaults. Trial parameters, failures, warnings, validation scores and elapsed times are saved. If all candidates fail, the run stops. Inspect any partial trial failures before publishing.

## Splits and evaluation

1. Snapshot, hash, truncate text to 4,000 characters and deduplicate model inputs. Conflicting-label duplicate inputs are removed. Official test copies take precedence over identical training copies. These are modified dataset variants, not directly comparable with unmodified leaderboard scores.
2. Preserve official train/test boundaries for AG News, Banking77 and IMDb. Otherwise take a stratified holdout. All models and all training seeds share exactly the same test rows.
3. Exclude the union of pilot test rows reconstructed with seeds 42/43/44 and test cap 300. This assumes the same snapshots and pilot settings. If your pilot differed, update the exclusion logic with its actual saved IDs before running; do not claim an untouched holdout from a seed change alone. Previously tested rows may be used for development. For nonofficial splits, v2 test rows may have been training/validation inputs in the pilot; v2 models are trained afresh and are disjoint from their own test set. This is a new holdout within the same public datasets, not independent external validation.
4. Split the development pool 60/20/20 into training, model-selection validation and decision-policy data. Caps are respectively 12,000 / 1,500 / 500. Test cap is 1,000, or 1,500 for Banking77. Smaller datasets use available rows. Every partition must contain every class. Exact IDs/counts are saved.
5. Fit candidate preprocessing only on training. Select each candidate by validation balanced accuracy; for binary models this includes a provisional validation threshold. This provisional threshold is discarded.
6. Refit the winner and its preprocessing on training + model-selection validation. Boosters use the validation-selected tree count. Tune the final binary threshold on the separate policy set, using up to 101 score quantiles, the default threshold, and a threshold above the maximum. Choose ties nearest the default. **Do not refit after this step.** The test labels do not enter selection or fitting.
7. Jev few-shot uses one labeled example per class from training (2 examples for binary tasks; 77 for Banking77). Raw Jev uses its returned choice. For the adjusted panel, binary Jev also receives the same policy inputs and uses their labels to select a threshold externally. These labels are not sent as extra prompt examples. Thus adjusted Jev zero-shot means **zero-shot inference with a supervised decision threshold**, not a zero-shot system end to end. ML and Jev have different training budgets; record both.

Binary class 1 is the positive class in the saved label mapping. The majority baseline is deliberately untuned. If a Jev test request fails after retries, its prediction is -1 and counts as incorrect; rows are never silently discarded. Any failed policy request disables threshold tuning for that Jev run, retaining raw choices with an explicit status in the result. Probability metrics are omitted if any required probability vector is missing. API probability normalization within a 0.01 sum tolerance is recorded in code.

## Reading results

`raw_balanced_accuracy.csv` and `adjusted_balanced_accuracy.csv` both have **datasets as rows, models as columns**. Raw means default decision rules for validation-selected models; the search objective was adjusted validation balanced accuracy, so these are not separately optimized raw-decision systems. Publish both panels to make threshold effects visible. Accuracy and macro-F1 matrices are also saved. `scores_by_seed.csv` includes ROC-AUC and average precision for binary models, plus Brier, clipped log loss and 10-bin ECE when probabilities exist. SVM margins support ranking/thresholding but are not presented as probabilities.

Each result JSON includes per-class recall, confusion matrix, raw/adjusted predictions, scores/probabilities, selected parameters and final threshold. `run_diagnostics.csv` exposes trial/API failures, disabled threshold tuning and resolved Jev model IDs; check for failures or mixed model versions before publishing. Class imbalance makes raw accuracy alone misleading. Balanced accuracy averages class recalls, so always predicting one class yields 50% on a two-class test.

Mean ± SD is across three training seeds on one shared test set. It is **not** three independent test sets or a confidence interval. The paired stratified bootstrap resamples test cases within classes and keeps model/seed pairing; 95% intervals describe test-sampling uncertainty conditional on the fitted models. They exclude retraining uncertainty, do not correct multiple comparisons, and do not justify a universal winner. Iris and Breast Cancer remain small illustrative tasks; Banking77 per-class recall can still be noisy.

All source-module hashes, configuration, package versions, dataset hashes and split IDs are saved. Changing code/configuration/environment requires a new output directory. Finished model files and successful exact API requests resume safely within the same run. The notebook zips the run output for download. Resume by restoring that directory in a matching environment; Kaggle working storage must be saved/downloaded to survive session loss. The API attempt log includes timestamps and request hashes, never the secret or authorization header. The package intentionally does not load your local `.env`.

## Cost and runtime

The full preset currently plans up to roughly 52,410 logical API requests before cache reuse and retries on these snapshots. Identical zero-shot test requests are reused across seeds. The attempt ceiling is 65,000; retries can exhaust it. The $20 estimate guard uses a configurable input-token price and UTF-8 byte proxy. It excludes output-token pricing and is **not a guaranteed invoice cap**. Verify current pricing and limits; adjust them before freezing a run. No automatic credit purchases occur.

Two GPUs help independent boosting fits; they do not accelerate every estimator or remote Jev calls. Larger text forests and four candidates across all datasets can take hours. Model search/refit wall time is diagnostic and includes prediction/evaluation overhead, excludes feature preparation, and is measured under concurrent load. This release makes no controlled latency/cost superiority claim. API request latency includes client pacing/retries and cached results retain earlier request timing.

## A defensible publication claim

“We compare Jev zero/few-shot inference with eleven conventional pipelines under a declared modest validation-search budget, using identical test cases and reporting both default and validation-tuned decision rules.”

Do not claim the best possible ML performance, equal pretraining resources, absence of public-dataset contamination, or superiority across all domains. Bank Marketing excludes call duration; Online Shoppers excludes PageValues. Those choices define pre-call / reduced-feature tasks and must be disclosed. Random stratified splits do not establish future-time generalization. Public datasets may have appeared in Jev pretraining. Freeze this protocol before reading new test scores; if you revise it afterward, label that run exploratory and obtain another holdout.

Threshold separation follows the [scikit-learn guidance](https://scikit-learn.org/stable/modules/classification_threshold.html). Native categorical handling follows [CatBoost documentation](https://catboost.ai/docs/en/features/categorical-features). Device/tree parameters follow [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/parameter.html).

## Module layout

`config.py`: budget and presets; `datasets.py`: snapshots/splits; `features.py`: fitted preprocessing; `models.py`: search spaces; `decisions.py`: thresholds/metrics; `training.py`: ML jobs; `api.py`: API client/payloads; `runner.py`: orchestration; `reporting.py`: matrices and paired intervals.

Local verification: `python validate_v2.py` exercises all models on numeric, categorical and text data, both-worker CPU orchestration, split exclusions on eight real snapshots, mock API responses, caching, thresholds and reporting. It makes no real API requests. GPU paths require an actual Kaggle GPU session to verify.
