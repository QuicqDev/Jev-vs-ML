# Post draft

I benchmarked Jev 1.13.0 against 11 classical classification pipelines across eight datasets.

The strongest result was IMDb sentiment classification: Jev reached 96.3% balanced accuracy with zero-shot prompts, compared with 88.4% for the best classical pipeline in this run.

The rest was mixed:

- AG News was close: 87.5% for raw zero-shot Jev versus 88.4% for SVM.
- SMS Spam showed why decision thresholds matter. Jev led with raw decisions, but classical models caught up after thresholds were tuned on separate labeled data.
- Banking77 favored SVM, and classical pipelines led on all four tabular datasets.
- One example per class helped some tasks and hurt others.

My takeaway: Jev looks promising for some language classification tasks with little task-specific labeling. These results do not support treating it as a general replacement for classical ML.

This was a bounded-budget experiment: four candidates per classical family, three training seeds, and shared held-out test cases. Threshold-adjusted Jev uses labeled data and is not zero-shot end to end. Cached zero-shot predictions are not independent replications, and these means alone do not establish statistical significance.

The executed notebook, both result panels, source, protocol, and limitations are here:
https://github.com/QuicqDev/Jev-vs-ML

## Before posting

The full Kaggle result archive is not in this release. Banking77 warnings need the saved failure diagnostics for a complete model-quality interpretation. Do not claim V3 latency, statistically significant wins, or best-possible classical optimization from the current public artifacts.
