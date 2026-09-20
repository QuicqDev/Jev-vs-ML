"""Generate the portable Kaggle notebook without requiring nbformat locally."""
import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', default='generated', help='Destination for a fresh V1 notebook.')
args = parser.parse_args()
path = (ROOT / args.output_dir).resolve() / 'jev_classification_benchmark.ipynb'
if path.exists():
    existing = json.loads(path.read_text(encoding='utf-8'))
    if any(cell.get('outputs') for cell in existing['cells']):
        raise SystemExit('Refusing to overwrite an executed notebook. Choose another --output-dir.')
path.parent.mkdir(parents=True, exist_ok=True)
cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.strip().splitlines(keepends=True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                      source=text.strip().splitlines(keepends=True)))


md('''# Jev vs 11 conventional classifiers · eight domains/datasets

**Upload this notebook to Kaggle and Run All. No companion files required.**

Before running:
1. In **Settings**, enable **Internet** and select an **NVIDIA GPU** accelerator.
2. In **Add-ons → Secrets**, add `TYPESAFE_API_KEY` and enable notebook access to it.
3. Choose `PRESET = "quick"` for a pipeline check or `"benchmark"` for the default experiment below.

**Run All makes paid requests to TypeSafe.** The key is read from Kaggle Secrets (or an environment variable), never embedded in the notebook or outputs. On **T4×2**, two independent dataset/seed workers run in parallel, one assigned to each GPU. GPU speeds up XGBoost and CatBoost; Jev runs remotely, and other baselines use CPU. Each GPU is tested with real fits; unusable devices are excluded, with CPU fallback when none pass.

Outputs land in `/kaggle/working/jev_benchmark_parallel`: tables, a heatmap, an HTML report, per-row predictions, split manifests, source hashes, selected hyperparameters, and resumable API responses. Download the final ZIP from the Output panel. Across Kaggle sessions, attach prior outputs from this notebook version as an input dataset and set `RESUME_FROM` below to its extracted output folder. The earlier sequential notebook has a different run signature and must not be mixed into this experiment. Downloaded artifacts contain dataset examples and model predictions, but no API key.

This is a **bounded practical comparison**, not an exhaustive model ranking or a contamination-free evaluation.''')

md('''## Experiment contract

| Setting | Quick | Benchmark (default) |
|---|---:|---:|
| Seeds | 42 | 42, 43, 44 |
| Maximum training rows per dataset/seed | 2,000 | 8,000 |
| Maximum validation rows | 400 | 1,000 |
| Maximum test rows | 100 | 300 |
| Candidate settings per base model | 1 | 2 |
| Trees/boosting iterations | 80 | 200 |
| Jev | Zero-shot + few-shot | Zero-shot + few-shot |

Each dataset/seed uses the **same test rows for every model**. Smaller datasets retain their actual smaller splits. Official train/test boundaries are retained for AG News, Banking77 and IMDb. Other datasets use stratified 80/20 train-pool/test splits; 20% of the training pool is validation. Training and validation are capped separately. Model selection uses validation balanced accuracy; final models and preprocessing are refitted on training + validation. No test-label-based tuning.

The 11 baselines are logistic regression, SVM, decision tree, random forest, extra trees, k-NN, naive Bayes, histogram gradient boosting, XGBoost, CatBoost, and a fixed hard-voting ensemble of logistic regression + random forest + XGBoost. The ensemble reuses tuned members; ties choose the smallest class ID. These are representative model families, not a claim they are the universally best eleven.

**Text pipelines:** linear logistic regression, linear SVM and multinomial NB use sparse word/bigram TF-IDF. Other models use training-fitted TF-IDF → 64-dimensional SVD → scaling. Tabular SVM uses RBF; tabular NB uses Gaussian NB. Tabular data gets median numeric imputation/scaling and categorical imputation/one-hot encoding. CatBoost deliberately uses the shared representation rather than a separately optimized native text/categorical pipeline. These choices limit runtime and must accompany any published results.

**Jev settings:** semantic class definitions and named input fields; no task demonstrations for zero-shot, one randomly selected training example per class for few-shot. Banking77 consequently has up to 77 examples, not one total. Both settings get exactly the same input fields/text truncation as ML. A maximum 4,000 characters of text per item is shared by all models. Few-shot examples are training-only, selected without test outcomes. Results are a comparison of training regimes, not equal training-data exposure.

Duplicate inputs are removed after truncation; conflicting-label duplicates are excluded. Official test copies take priority over duplicate training copies. This modifies the original benchmark population, so do not compare these scores directly to published full-dataset scores. Snapshots and exact row IDs are saved.

**Interpretation:** report balanced accuracy, macro-F1 and accuracy per dataset. Mean ± standard deviation across seeds is **not** a confidence interval; test sets can overlap. Iris and Breast Cancer have small test sets. Public datasets may have been seen in Jev pretraining. No universal average accuracy is reported; mean rank uses only fully completed datasets. API failures count as wrong and are reported. Jev probability diagnostics are supplementary, on successful responses only, not a calibration comparison against uncalibrated SVM/ensemble outputs.''')

md('''## 1 · Dependencies

XGBoost is pinned to a CUDA-12-compatible release rather than requiring the newest CUDA runtime. Existing compatible core packages are retained. Versions and hardware are recorded. CatBoost GPU training can have small nondeterministic variation even with fixed seeds.''')
code('''import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "xgboost==2.1.4", "catboost==1.2.8", "scikit-learn>=1.5,<1.9",
    "huggingface_hub>=0.26,<2", "pyarrow>=14", "pandas>=2,<3",
    "numpy>=1.26,<3", "requests>=2.31", "matplotlib>=3.7", "threadpoolctl>=3", "tqdm>=4.66"])
''')

source = (ROOT / 'legacy' / 'kaggle_benchmark.py').read_text(encoding='utf-8')
sections = source.split('# %%\n')
headings = ['## 2 · Data loading and reproducible splits', '## 3 · ML pipelines and modest validation search',
            '## 4 · Jev API, bounded concurrency and checkpointing', '## 5 · Orchestration and reporting']
for index, (title, section) in enumerate(zip(headings, [s for s in sections if s.strip()])):
    md(title)
    if index == 0:
        section += '\nNOTEBOOK_IMPLEMENTATION_SHA = ' + repr(hashlib.sha256(source.encode()).hexdigest()) + '\n'
    code(section)

md('''## 6 · Configuration

Change `PRESET` before your first run. Use a different `OUTPUT` folder when changing any configuration or dependency versions. Resume only with identical configuration. `max_api_attempts` includes retries and counts persisted attempts across resumes; the dollar guard uses a conservative UTF-8-byte token proxy and an **editable assumed tariff**, not a guaranteed invoice cap. Reported token usage is saved separately. Retries may themselves be billed.

Default nominal volume is about 11,664 requests before deduplication/cache reuse/retries. Quick mode is about 1,460. Actual counts are printed after preparation. Run time depends on Kaggle hardware, API limits and service availability; no fixed duration is promised. Increase test/training caps for a larger follow-up, using a new output directory.''')
code('''PRESET = "benchmark"  # change to "quick" for a first end-to-end check
CFG = config(PRESET)
OUTPUT = "/kaggle/working/jev_benchmark_parallel"
RESUME_FROM = None  # e.g. "/kaggle/input/my-previous-run/jev_benchmark"

# Optional overrides (change before the first run):
# CFG["datasets"] = ["AG News", "Bank Marketing", "Iris"]
# CFG["jev_modes"] = ["zero-shot"]
# CFG["test_cap"] = 1000
# CFG["train_cap"] = 20000
# CFG["max_parallel_jobs"] = 1  # disable parallel jobs for isolated timings
# CFG["jev_model"] = "jev-latest"  # resolved versions are saved; pin for reproducibility

ROOT = init_run(CFG, OUTPUT, RESUME_FROM)
GPU_IDS, HARDWARE = probe_gpus()
write_json(ROOT / "hardware.json", {"gpu_ids": GPU_IDS, "description": HARDWARE})
print("Hardware:", HARDWARE)
print("Usable GPU IDs:", GPU_IDS)  # T4x2 should show [0, 1]
print(json.dumps(CFG, indent=2))
# Fail early on a missing key, before doing ML work. No key is printed.
_key_check = get_key()
del _key_check
''')

md('''## 7 · Fetch and freeze datasets

Sources: [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets), [SMS Spam](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb), [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), [Online Shoppers](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), [Breast Cancer and Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html).

Bank Marketing excludes `duration` because it is unavailable before a call. Online Shoppers excludes outcome-related `PageValues`; remaining features describe the completed session, so this is **retrospective classification**, not a real-time early-purchase forecast. Medical data is used solely as an offline benchmark. This notebook is not a clinical tool.

Hugging Face revisions are resolved once and saved; other source downloads are hashed. Check upstream licenses before redistributing source snapshots.''')
code('''PREPARED, OVERVIEW = prepare_suite(ROOT, CFG)
display(OVERVIEW)
''')
md('''## 8 · Train and evaluate the 11 baselines

Two small, explicit candidate settings per model under the default preset. No broad grid search. Up to two isolated worker processes train independent dataset/seed jobs; each worker owns one GPU for its entire queue. Within a worker, models fit sequentially, avoiding overlapping fits on the same GPU. With T4×2, both devices can train independent models at the same time. GPU activity is intermittent while a worker preprocesses data or trains CPU-only baselines; this does not guarantee a 2× speedup.

CPU libraries use at most two threads per worker, reduced when needed to stay within available CPU cores. Shared preprocessing costs and model costs are recorded separately. Each worker loads one dataset at a time from the frozen snapshot. Completed dataset/seed jobs are reused on rerun; only the parent writes aggregate reports. Prediction latency is collected under parallel workload and may include CPU/memory contention. Use `max_parallel_jobs=1` in a **new** output folder for isolated timing measurements.''')
code('''run_baselines(PREPARED, ROOT, CFG, GPU_IDS)
display(render_results(ROOT, CFG))
''')
md('''## 9 · Evaluate Jev on the same test rows

Requests run with four workers and a global start-rate limit. Successful responses are cached by the complete request hash; no additional charge for cache reuse. HTTP 429 and transient errors have bounded retries. Permanent errors halt API work with an actionable message. Completed failed predictions are kept as failures; an interrupted job can reuse its successful requests on rerun.

API latency includes concurrency/rate-limit wait, HTTP, model time and retries; cache hits are excluded from new latency statistics. ML latency includes preprocessing and prediction on individual rows, with warm-up, while other ML workers may be active. These are deployment measurements on different hardware, not normalized or isolated inference-speed claims. Cached predictions can recur across seeds, especially zero-shot.''')
code('''run_api(PREPARED, ROOT, CFG)
''')
md('''## 10 · Results and downloadable bundle

The primary table is balanced accuracy (%). Accuracy and macro-F1 matrices, raw per-seed scores, predictions and hyperparameter selections are also saved. Inspect per-dataset outcomes rather than declaring a universal winner from a few benchmarks.''')
code('''from IPython.display import display, Image, FileLink
TABLE = render_results(ROOT, CFG)
display(TABLE)
display(Image(filename=str(ROOT / "balanced_accuracy_heatmap.png")))

scores = pd.read_csv(ROOT / "scores_by_seed.csv")
display(scores.groupby("model")[["latency_p50_ms", "latency_p95_ms"]].median().rename(
    columns=lambda c: "median_across_jobs_" + c))
jev = scores[scores.model.str.startswith("Jev")]
if not jev.empty:
    print("Jev evaluated-row failures:", int(jev.failures.fillna(0).sum()))
    # This includes evaluated cached responses across jobs; not unique billable tokens.
    print("For billing analysis, deduplicate files in jev_cache; API attempts are in api_attempts.jsonl.")
    cached = [json.loads(p.read_text()) for p in (ROOT / "jev_cache").glob("*.json")]
    tokens = sum(r.get("usage", {}).get("input_tokens", 0) for r in cached)
    models = sorted({str(r.get("resolved_model")) for r in cached})
    print("Resolved Jev model versions:", models)
    if len(models) > 1:
        print("WARNING: multiple resolved model versions; do not treat them as one fixed model.")
    print("Input tokens across unique successful cached responses:", tokens)
    print("Estimated successful-response input cost (excludes failed/retried billed requests): $",
          round(tokens * CFG["estimated_usd_per_million_input_tokens"] / 1e6, 4))

bundle = shutil.make_archive(str(ROOT.parent / (ROOT.name + "_results")), "zip", ROOT)
display(FileLink(bundle))
display(FileLink(str(ROOT / "report.html")))
''')
md('''## References and limitations

- [Jev API and typed questions](https://docs.typesafe.ai/introduction/quickstart)
- [Jev confidence semantics](https://docs.typesafe.ai/confidence): use option probabilities for calibration; the API confidence field is not simply the winning-class probability.
- [XGBoost GPU](https://xgboost.readthedocs.io/en/release_2.1.0/gpu/)
- [CatBoost GPU](https://catboost.ai/docs/en/features/training-on-gpu)
- [Kaggle notebook settings](https://www.kaggle.com/docs/notebooks)

The benchmark caps training data and lightly tunes models. It does not measure fully optimized model ceilings, provide equal pretraining exposure, or establish causal/live prediction performance. Text tree results depend on SVD features. Few-shot counts vary with the number of classes. Class definitions and named columns supply semantic information to Jev that the conventional models learn from labeled rows. GPU results may vary slightly between executions. Persist configuration and outputs with every shared table.

For stronger conclusions: increase held-out samples, retain untouched final datasets, add recent/private data, and separately evaluate the sensitivity to prompts, demonstration counts and training-set size.''')

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
               language_info=dict(name='python', version='3.11'),
               benchmark_source_sha256=hashlib.sha256(source.encode()).hexdigest()), nbformat=4, nbformat_minor=5)
for i, cell in enumerate(cells):
    cell['id'] = f'benchmark-{i:03d}'
path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(path)
