# V3 verification

## Published completed run

The supplied completed Kaggle notebook is now `jev_benchmark_v3.ipynb`, preserved byte for byte. Its SHA-256 is recorded in `published_results/manifest.json`. All embedded source modules and protocol files match the editable repository files exactly. Saved outputs include successful GPU probe messages, completion statuses for all 24 dataset/seed ML jobs, and both final result panels with all 14 columns populated for three seeds (11 classical families, majority baseline, and two Jev modes).

The notebook records 38,922 API attempts. Its final results are extracted into two CSVs without rerunning any cells. The source bundle includes the executed notebook. `validate_v3_artifact.py` now validates executed outputs, the embedded package hash, source equality, exported tables, and bundle consistency without depending on a private local results directory.

This is evidence from the supplied saved run, not an independent GPU rerun. Banking77 API warnings remain visible. Full per-example results, diagnostics, snapshots and paired bootstrap intervals are not included in the supplied artifact. See README limitations.

The sections below are the historical pre-run validation record. Their statements about Kaggle verification still being needed describe that earlier stage.


## 3.0.1 device-isolation patch

The reported RAFT/cuML k-NN invalid-resource-handle error occurred during the original same-process GPU probes. Device-context reuse is a suspected cause, not a GPU-reproduced diagnosis. The patch removes that exposure by assigning one visible GPU per fresh interpreter before CUDA imports, for both probes and training. Native probes and cuML probes now use local device 0 only. The parent no longer imports cuML to obtain version metadata.

`validate_gpu_process.py` passed with real child interpreters (without CUDA): distinct process IDs, per-child device masks visible before CUDA imports, no parent environment mutation, inherited UUID mapping, probe-before-training barrier, error propagation and child cleanup. `validate_v3_backends.py` also passed. Source compilation and notebook package validation passed. Actual cuML execution still needs verification on Kaggle; this patch does not claim the reported GPU error was reproduced locally.

Local offline integration: `python validate_v2.py --v3` passed. It exercises the revised CPU pipelines with four candidates each, all 11 families and dummy, two workers, mixed categorical/numeric/text inputs, refit/threshold logic, mocked Jev, caches and report generation. Small synthetic integration fits use reduced tree counts; the actual Banking77 timing below uses the v3 histogram ceiling. Split and pilot-exclusion checks cover all eight real snapshots and every preset/seed.

`python validate_v3_backends.py` passed: cuML constructor routing is tested with doubles, including explicit preservation of weighted CPU forests, sparse-text SVM CPU routing, GPU-required failure, the histogram budget and eight Jev workers. This does not verify CUDA computation.

`python profile_v3_histogram.py` measured the revised Banking77 bottleneck on training data only: 6,001 training rows, 77 classes, 32 SVD features, 60 histogram iterations, two CPU threads. Feature preparation took about 0.7 seconds and one candidate fit about 4.0 seconds on this local machine. This is not a Kaggle full-suite runtime estimate; it excludes the other candidates, refit and other models.

No new paid Jev calls were made. The local Windows environment cannot execute cuML/CUDA. The delivered notebook therefore runs real dense/sparse, binary/multiclass, weighted/unweighted cuML adapter probes on each T4 before the full training suite. Failed probes stop the run; actual GPU performance and correctness must be verified there. No 30-minute total-runtime claim is made.

cuML APIs were checked against NVIDIA's documentation. The GPU random-forest implementation and logistic-regression solver differ from sklearn; backend selection is saved rather than described as numerically identical acceleration. See `BENCHMARK_V3.md` for routing and budget details.
