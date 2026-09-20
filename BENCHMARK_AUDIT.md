# Audit of the first Kaggle benchmark run

The current results are an exploratory, lightly tuned pipeline comparison. They do not establish that one model family is generally superior. The existing run should finish unchanged; protocol revisions belong in a separately labeled experiment.

## Observed issue: default decision thresholds and imbalance

Local reproduction on the frozen dataset snapshots, seed 42, CPU XGBoost, same default training caps and two candidate settings:

| Dataset | Model | Default balanced accuracy | Positive recall | ROC-AUC | Diagnostic validation-selected threshold | Diagnostic balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| Bank Marketing | Logistic regression | 56.39% | 5/35 | 0.7467 | 0.10 | 66.63% |
| Bank Marketing | XGBoost | 57.82% | 6/35 | 0.7472 | 0.10 | 70.49% |
| Online Shoppers | Logistic regression | 52.80% | 3/47 | 0.7645 | 0.15 | 65.49% |
| Online Shoppers | XGBoost | 54.80% | 6/47 | 0.7984 | 0.10 | 73.53% |

These are local diagnostics, not the user's complete Kaggle results or replacement leaderboard scores. Thresholds were selected from 19 candidates (0.05–0.95) on validation predictions from the selected training-only model, then applied after the final train+validation refit. No test labels selected the thresholds. The investigation itself followed inspection of test outcomes, so a strengthened protocol should be frozen and evaluated on new untouched holdouts or datasets. Refit can shift probability distributions; a future protocol should explicitly fix the calibration/threshold-selection procedure.

Default models are unweighted and binary predictions generally use 0.5 probability or zero decision margin. Balanced accuracy averages per-class recall; majority-heavy predictions can have high ordinary accuracy and near-50% balanced accuracy. Report class prevalence, per-class recall, confusion matrices and threshold-free ranking metrics in addition to the main table. A majority-class dummy is an essential reference. Include raw/default and validation-tuned decision-policy panels if appropriate; do not tune Jev on its test predictions.

## Other limitations relevant to publishing

1. Text trees/k-NN see only 64 SVD components; logistic regression, linear SVM and multinomial NB see up to 12,000 TF-IDF features. This compares whole pipelines, not classifier families in isolation. Include representation in row/column notes. Test sparse TF-IDF or supervised feature-selection alternatives for tree models in a separately validated protocol before claiming Jev outperforms strong text tree baselines.
2. Only two candidate settings and up to 8,000 training rows are used. Say "lightly tuned baselines", not "best ML models". CatBoost is not using its native categorical handling. SVM is linear for text and RBF for tabular; NB also changes by input type.
3. At 300 rows and 77 classes, Banking77 has roughly 3–4 test examples per class per seed. Small gaps cannot support a reliable ranking. Use a substantially larger untouched test sample/full official test split for the final run. Iris has 30 held-out rows per seed. Report actual sample counts.
4. Three seeds reuse overlapping test populations. Their standard deviation is not a 95% confidence interval, nor are three seeds three independent datasets. For final paired comparisons, use identical fixed held-out cases and paired uncertainty estimates at the example (or natural group) level. Account for repeated rows when aggregating.
5. Bank Marketing excludes duration. Online Shoppers excludes PageValues. These are modified feature sets and cannot be compared directly to standard full-feature literature scores. PageValues being predictive or outcome-related alone does not prove leakage; construction and availability at prediction time determine leakage. Explain the exclusion as a conservative experimental choice, not a proven defect in the original dataset. Online Shoppers uses completed-session features, so it is retrospective classification.
6. Jev has external pretraining, semantic instructions and named fields; ML receives task-labeled data. Few-shot has one example per class (77 for Banking77, 2 for binary tasks). This is a useful operational comparison, not equal information or training budgets. Public-data contamination cannot be excluded.
7. Latencies include different systems and workloads: parallel CPU/GPU training versus concurrent remote API requests with queueing/retries. Avoid inference-speed superiority claims without a separate controlled latency experiment.
8. Progress counts indicate completed futures, not successful predictions. Verify saved `failures`, model versions, probability validity and cache counts. Freeze dataset revisions, parameters, prompts and code; release predictions and split IDs, subject to dataset licensing. The main table must explicitly say "Balanced accuracy (%) — mean ± seed SD; n=3 seeds".

## What is already sound

Shared test cases within each seed; disjoint training/validation/test indices; preprocessing fit on training only during selection; validation-only hyperparameter selection; training-only few-shot examples; request/result caching; saved resolved model identity; failed API predictions counted as incorrect. These properties must still be verified against the actual exported run artifacts before sign-off.

## Next run UI

The local notebook source now uses tqdm for each Jev dataset/seed/mode. This is a display-only change. Do not replace definitions or the notebook while a Kaggle run is in progress. The source hash changes between notebook versions; retain the current run's original notebook and output together.
