# Jev classification benchmark v3 (3.0.1 patch)

3.0.1 fixes device isolation. Native-library/cuML probes and training now run in fresh subprocesses with one `CUDA_VISIBLE_DEVICES` entry set before imports. The two physical GPUs each become local device 0 in their own process. Both probe subprocesses must succeed before any training subprocess starts; logs are saved under `gpu_workers/`. Interrupting the parent terminates its active children. The probe stage has a 180-second timeout. Hyperparameters, split selection and eight Jev workers are unchanged.

Upload `jev_benchmark_v3.ipynb`. It contains its Python modules. Use Internet on, T4 x2, **Kaggle Settings > Environment Preferences > Always use latest environment**, and your existing Kaggle secret. This uses Kaggle's preinstalled RAPIDS rather than blindly installing a CUDA wheel. The setup checks cuML imports; real adapter fits on both GPUs run before expensive training. Keep the old notebook/results as a separate experiment.

The original v2 increased the number of candidates from two to four and the tree ceiling from 200 to 400, while increasing training size and text tree features. CPU histogram boosting on Banking77 builds one tree per class per iteration: 400 iterations can mean 30,800 trees per candidate, before refitting. GPU idleness during these CPU models is expected.

The **v3 preset** retains all 11 models, three seeds, four candidates per family, class-weight comparisons, independent policy thresholds and the same test selection. Its declared computational budget is smaller:

| Setting | Original v2 | v3 |
|---|---:|---:|
| Training cap | 12,000 | 8,000 |
| Selection-validation cap | 1,500 | 1,000 |
| Forest/boosting tree ceiling | 400 | 150 |
| Histogram-boosting iteration ceiling | 400 | 60 |
| Histogram bins | 255 | 63 |
| Text histogram-boosting features | 2,000 selected TF-IDF | 32 scaled SVD components |
| Other text-tree features | 2,000 | 512 selected TF-IDF |
| Word/character vocabulary caps | 30,000 / 20,000 | 20,000 / 10,000 |
| SVD dimensions for k-NN | 64 | 32 |
| Forest second candidate depth/features | unlimited / 50% | 24 / 15% |
| Jev request workers | 4 | 8 |

These are runtime/representation tradeoffs, not mathematically equivalent accelerations; accuracy can change. The four-candidate search remains two parameter settings crossed with ordinary/balanced weights (k-NN uses four neighbor settings). See `BENCHMARK_V2.md` for the common protocol; **this document overrides its budget/representation/backend values for v3**.

## Explicit GPU backends

| Family | v3 backend |
|---|---|
| Logistic regression | cuML GPU, including sparse text; QN solver, same C candidates |
| Random forest | cuML GPU for unweighted candidates (128 bins, one stream); sklearn CPU for sample-weighted candidates |
| k-NN | cuML GPU brute-force neighbors, same distance/uniform weighting candidates |
| SVM | cuML GPU RBF on binary tabular tasks; sklearn for sparse text LinearSVC and multiclass tabular SVC |
| XGBoost / CatBoost | Their native CUDA implementations |
| Decision tree / Extra trees / Naive Bayes / Histogram gradient boost | sklearn CPU |
| Voting ensemble | CPU probability averaging of the three fitted members |

Backend routing is explicit rather than `cuml.accel` monkey-patching. No weighted random forest candidate silently loses its sample weights. GPU RF uses the dense form of the SAME selected TF-IDF values. cuML RF uses quantile-based splits and differs algorithmically from sklearn RF: this column selects among four **random-forest pipeline candidates with declared mixed implementations**. Do not describe it as pure sklearn or pure cuML. GPU logistic regression also uses a different solver. Each trial/result records its backend and estimator class; `run_diagnostics.csv` records the selected backend. Package/cuML/CuPy versions and GPU probe results are saved.

Each fresh worker sees only its assigned GPU; a CuPy context alone is no longer used to isolate devices. Failed cuML imports or preflight fits stop before the long benchmark. Sparse text LinearSVC, weighted forests and the other listed CPU cases remain CPU by explicit policy. Small datasets may be slower on GPU due to transfer/initialization overhead.

References: [RAPIDS on Kaggle](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/), [cuML compatibility restrictions](https://docs.nvidia.com/cuml/latest/cuml-accel/compatibility/), [GPU logistic regression](https://docs.nvidia.com/cuml/latest/api/generated/cuml.linear_model.LogisticRegression/), [GPU neighbors](https://docs.nvidia.com/cuml/latest/api/generated/cuml.neighbors.KNeighborsClassifier/).

Histogram boosting now enables sklearn's internal early stopping using 15% of the training partition. That subset remains within training, separate from model-selection validation, policy and test data. Model-selection validation still selects the candidate. The selected number of iterations is then fixed for the train+validation refit, with early stopping disabled. XGBoost/CatBoost use 15-round patience; their early stopping uses model-selection validation as before.

Both GPU libraries are probed before the run; the fast preset stops if no GPU passes instead of silently running boosting on CPU. XGBoost's actual fitted device is also checked. GPU estimators run first within each dataset/seed job. Each GPU has one worker; CPU models remain CPU models. This does not promise continuous GPU utilization or GPU acceleration of every algorithm. Per-candidate start/end times and CPU/GPU labels show where time is being spent. Failed trials remain visible.

## Switch from the interrupted run

1. Preserve/download the existing `/kaggle/working/jev_benchmark_v2` output folder before resetting a Kaggle session. Completed model results remain useful as old-protocol results.
2. Upload the v3 notebook. If the old folder remains accessible, set `PILOT_ROOT` in its configuration cell to that directory to reuse dataset snapshots. Otherwise leave it `None` to fetch datasets again.
3. Restart the kernel and run the updated notebook from the first cell. Its patched default `ROOT` is `/kaggle/working/jev_benchmark_v3_1`; do not reuse the previous manifest. If `/kaggle/working/jev_benchmark_v3` is still present, its data snapshots are reused automatically. Old model scores are not copied. Preserve/download files before any Kaggle session reset that clears working storage.
4. Run ML, then the separate paid Jev cell. API calls are not made while testing ML. The test/policy caps and cost controls are unchanged. Eight threads do not override the request-start spacing.

The new models use the same test selection as original v2. If you already inspected those test scores, disclose this as a computational revision on that holdout, not a new untouched test. This speed revision was driven by runtime, not selecting models on test scores.

There is no verified 30-minute guarantee on Kaggle. The structural workload is reduced substantially, but full runtime depends on CPUs, T4 availability, class counts and convergence. Actual two-T4 testing must happen in Kaggle. Pin and report the fast protocol with results; do not describe its search as best-possible ML optimization.
