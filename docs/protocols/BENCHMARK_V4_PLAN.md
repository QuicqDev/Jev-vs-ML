# Jev follow-up: alternatives, generalization, and iterative decisions

Status: proposed study, not a frozen protocol or an executed benchmark. Prepared 2026-09-21 from the Reddit feedback and the existing V3 experiment. User constraints: Kaggle-friendly; Jev tokens are currently unlimited. Runtime, GPU memory, API throughput, and quality of evidence are the practical limits.

## Questions and interpretation

1. How does Jev compare with local decision models and a stronger automated ML baseline?
2. Which results survive changes in wording, composition, and time?
3. Does feedback improve Jev's decisions across a sequence of actions, and more than it improves other models?

A good IID holdout result is not itself evidence of overfitting. It can be valid for the sampled distribution while saying little about deployment under shift. V3 preserves official test boundaries where available and deduplicates exact inputs. Its text SVM is linear TF-IDF; its tabular SVM uses an RBF kernel. Tests should target specified generalization failures rather than select datasets after discovering where SVM loses.

TypeSafe describes RLCD as optimizing calibrated decisions. This does not establish online learning, memory across calls, or superior long-horizon planning. A successful loop would demonstrate usefulness as a decision component; attributing the gain to RL would require a comparable model without that training, which we do not have. See the [TypeSafe launch description](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## Model roster

| Model/pipeline | Role | Scope |
|---|---|---|
| Jev, pinned model ID | Hosted decision model | All core tasks |
| Von, pinned source and weights | Local decision model | All supported core tasks |
| Laya, pinned router and checkpoint revisions | Local decision system | All supported core tasks; record the checkpoint per request |
| TF-IDF + linear SVM | Strong inexpensive text baseline | Text and observable-state action classification |
| Frozen sentence embeddings + logistic regression | Semantic representation control | Text; separates lexical representation limits from classifier limits |
| CatBoost and scaled RBF SVM | Conventional tabular baselines | Bank Marketing and temporal task |
| AutoGluon Tabular | Automated model selection and ensembling | Bank Marketing and temporal task |
| Majority / persistence / explicit rules | Sanity and task-specific baselines | As applicable |

Do not rerun all eleven classical families initially. Freeze one sentence encoder by revision after a compatibility smoke test, without selecting it using test performance. Log its task-specific labels and pretrained status.

[Von](https://github.com/wfzyx/von) exposes Choice, Noul, and Score interfaces similar to Jev. Treat repository accuracy, calibration, and latency claims as hypotheses to measure independently.

[Laya](https://github.com/NandhaKishorM/laya) documents multiple checkpoints, a router, AG News in its training mix, and option/context-budget limitations. Its typed-decisions checkpoint has task-specific training. Freeze routing before evaluation; report checkpoint and known training exposure. Do not select a checkpoint per test result or silently truncate Banking77 choices. The default routed system is the main entry; a fixed checkpoint is a separately named diagnostic.

## Core experiment matrix

| Experiment | Proposed data and size | Main question / primary measure |
|---|---|---|
| Continuity: language | IMDb: 1,000 cases; Banking77: 1,500 | Alternatives on Jev's strong result and a challenging intent task; balanced accuracy |
| Continuity: tabular | Bank Marketing, duration excluded: 1,000 | Does a stronger automated baseline change the comparison? Balanced accuracy |
| Fresh policy decisions | 1,200 cases in 600 related pairs, divided equally between familiar and held-out composition families | Negation, exceptions, chronology, paraphrase; accuracy and both-members-correct pair rate |
| Future prediction | Hourly Bike Sharing, three forward windows, target 500 consecutive test cases/window | Does performance survive a real temporal boundary? Balanced accuracy versus persistence/seasonality |
| Iterative decisions | 500 held-out support episodes, at most four model decisions/episode | Success and net utility after tool feedback; cost of mistakes and unnecessary steps |

The continuity results are extensions on known public datasets, not new confirmation of the original conclusions. Reconstruct and hash exact V3 cases if possible; otherwise use fresh manifests and rerun every comparator under V4. Existing aggregate CSVs are insufficient for paired inference against new predictions. Do not join an old mean to a new prediction file and call it a paired comparison.

Use up to 8,000 training and 1,000 selection rows where available. Keep final policy data separate, with up to 500 rows on static tasks. Freeze actual counts and IDs after data-only feasibility checks. New synthetic data needs separate development, selection, policy, and test families; the 1,200 count above refers only to test cases.

### Fresh policy decisions

Build a small refund/support policy world with an executable rule-based labeler and varied natural-language observations. Include four predeclared families:

- Negation or word-order changes that flip the correct action despite similar vocabulary.
- A general policy with an exception that changes the outcome.
- Event ordering, such as cancellation before versus after dispatch.
- Meaning-preserving paraphrases and irrelevant details that should preserve the action.

Include straightforward cases too. Reserve unseen combinations and wording families before evaluating any model. Keep every pair and all derivatives of a base scenario within one partition. An independent human review should check a fixed sample for ambiguity and agreement with the executable labels before the test is frozen. Do not use Jev or a competitor as the sole test-label judge.

All systems receive the same available observations and policy; classical pipelines receive labeled development examples. Raw-state and structured-feature diagnostic results must be separate if the latter supplies an extraction advantage. The rule oracle's privileged access, if any, must be explicit.

This probes specific lexical/compositional limitations, not a universal weakness of SVMs. If the embedding baseline fixes a failure, attribute the difference to the full representation/pipeline. Synthetic success does not establish real-world robustness. Optional later extension: [Amazon-WILDS](https://wilds.stanford.edu/datasets/) for natural user-distribution shift, retaining its official split and group metadata.

### Temporal experiment

Use [UCI Bike Sharing hourly data](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). Define a classification task compatible with all three decision APIs: at the end of hour t, will demand in hour t+1 exceed a fixed high-demand threshold? Derive that threshold once from the earliest training block, freeze it, and supply it to all systems.

- Construct exact hourly timestamps. Require the next-hour target and requested lag timestamps to exist; do not assume adjacent CSV rows are adjacent hours.
- Inputs: known calendar fields, demand observed through t, and past/current observations. Exclude future realized weather and the next-hour count or its casual/registered components.
- Use expanding training windows followed by selection, policy, and contiguous test blocks. Freeze three nonoverlapping test windows from timestamp coverage before viewing model outcomes. Use one training seed per window initially; these windows are not interchangeable with seed replicates.
- Purge examples whose target availability crosses a partition boundary. Add a predeclared 24-hour gap between development partitions and test for the conservative primary analysis; report it. Legal historical lags at prediction time are allowed, including demand observed earlier in a test window. This is rolling one-hour prediction, not forecasting an entire block without new observations.
- Fit preprocessing, feature selection, and tuning on past data only. No random internal cross-validation for the primary temporal run. Models stay fixed within each test window.
- Include last-observed demand and same-hour-last-week baselines, applying the same high-demand threshold. Record missing seasonal history and the fixed fallback rule.
- Add a random-split version as an explicitly retrospective diagnostic. Its test population differs; a score gap alone does not isolate leakage or prove why performance changed. Pair models within each protocol, not cases across different test sets.

This is a historical deployment simulation on public data, not a live prospective test or evidence that pretrained models never saw the dataset. Publish class counts per window. If a metric is undefined in a window, report that rather than reshuffling timestamps to obtain a desirable test distribution.

### AutoML settings

Use a version-pinned AutoGluon Tabular run with a proposed **30-minute fit budget per fit** and balanced accuracy as the selection objective. Allow normal ensembling on the IID tabular task. For temporal data, use explicit past-to-future selection data and disable bagging/stacking initially unless the pinned version's forward-only behavior is verified. Keep policy and test data outside AutoML; disable its automatic decision-threshold adjustment and apply the shared external policy procedure. Record the actual preset, model inventory, hardware, and elapsed time. [AutoGluon fit documentation](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html)

Use three Bank Marketing seeds and three temporal fits: six fits give a nominal three-hour fit allocation. Downloads, preprocessing, overruns, and inference add time; this is not a session-runtime guarantee. A second 60-minute budget belongs in a predeclared extension across all selected tasks, not only a task whose result disappointed us. Do not describe a bounded AutoML run as the best achievable ML.

### Iterative decision experiment

Build a deterministic support simulator: inspect a payment or delivery record, then refund, deny, or escalate. Tool actions reveal previously unavailable evidence and consume steps/cost. Ground truth comes from the simulator's latent state and fixed policy. The model gets only information that its actions have revealed. Use a state envelope that records observations, actions, remaining steps, and the policy. No model-specific hidden memory.

Train the SVM and embedding policy on development-state optimal-action labels under the same observation constraints. Hold out complete scenario families and episodes. Freeze error costs, inspection costs, escalation cost, and success criteria on development episodes. Include an executable policy baseline using the same observations, plus an explicitly privileged fully informed oracle as an upper bound.

Run four conditions on the same episode seeds:

1. **One decision:** choose a terminal action from the initial evidence.
2. **Feedback loop:** choose inspections and actions, observe outcomes, maximum four decisions.
3. **Repeated calls without new evidence:** same maximum call budget, final terminal decision; tests whether extra calls alone explain an improvement.
4. **Fixed evidence schedule:** a predefined inspection schedule provides evidence within the same cap; tests whether adaptive information gathering beats a simple workflow.

Distinguish additional-information gains from planning gains. The one-decision condition inherently sees less evidence. Compare adaptive and fixed schedules at matched allowed step budgets, and report actual steps and utility. A valid advantage must survive inspection costs, rather than come only from making more calls. Repeated calls need not improve a deterministic model; that is a useful control result.

Primary outputs: episode success, mean net utility, critical-error rate, escalation rate, steps, and end-to-end episode latency. Reset state between episodes. Cache keys include the full trajectory and run identity; repeated-call controls must actually execute fresh calls. No test-episode feedback enters another episode's training. Online updating across episodes is a separate future study.

## Fair comparisons and uncertainty

- Main pretrained comparison: fixed prompts without task examples. Secondary few-shot comparison: one example/class on compact tasks that fit all models. Freeze identical examples and label descriptions. Banking77 starts with zero-shot prompts because full few-shot context may not be mutually supported.
- Separate raw/default outputs from supervised threshold adjustment. Report training, selection, examples, and policy label counts separately; threshold-adjusted zero-shot prompts do not make a zero-label system.
- Audit serialized inputs and actual tokenization before freezing the run. Preserve every candidate choice and decision-relevant fact. If a model cannot support a task within the declared representation, report unsupported; any shorter-context or hierarchical variant is a separately named system with its full cost.
- Use returned class probabilities for probability metrics, not an arbitrary field named confidence. SVM margins remain margins unless a separate calibration procedure is fitted. Primary probability analysis uses native probabilities; a later calibrated panel needs disjoint or properly cross-fitted calibration/policy data.
- Report balanced accuracy, macro-F1, per-class recall, Brier score and log loss where valid. For automation, report error at matched coverage and coverage at a target error selected on policy data; show attained test error without promising the policy target will hold.
- Count failed/invalid responses in operational accuracy and success. Show probability-metric coverage explicitly; never silently compare only a model's successful subset. Keep failures, retries, truncation, and unsupported tasks visible.
- Use paired intervals: example resampling for independent cases; base-scenario/pair clusters for generated data; time blocks for temporal data; whole episodes for loops. Preserve model pairing. The existing row-stratified bootstrap is not sufficient for every new task.
- Freeze a small list of primary comparisons and report all of them. For formal multiple superiority claims, predeclare a correction such as Holm; otherwise present intervals as exploratory. Training-seed SD is not a test confidence interval. Treat 500 episodes and proposed test sizes as initial study sizes, not a guarantee of resolving small differences; freeze sizes before test inspection.
- Measure uncached end-to-end latency with controlled concurrency, reporting p50/p95, failures and retries. Also show local model-only inference, warm/cold startup, hardware, peak memory, and throughput separately. API network time and local kernel time are different measurements. Unlimited Jev tokens does not mean unlimited request rate; self-hosting still uses compute.

## Kaggle execution and deliverables

1. **Compatibility pilot:** 50 development cases/model, token/choice preservation checks, library compatibility, observed VRAM, request throughput, and simulator validation. No test-based model or prompt selection. Install heavy AutoML and local-model dependencies in separate sessions if necessary.
2. **Freeze V4:** package versions, weight revisions, Jev model identifier, split IDs, feature availability, prompts, metrics, budgets, and artifact hashes. Freeze test-family generation independently from pilot examples.
3. **Classification sessions:** run local models sequentially on a GPU; only parallelize across the two T4s when each workload fits independently. They are not a single pooled-memory device. Run CPU/AutoML jobs with explicit resource limits. Run Jev with measured service-safe pacing and bounded retries.
4. **Loop session:** validate transition rules and action availability with deterministic policies, then run every model and control on the frozen episodes.
5. **Export before session shutdown:** predictions, probabilities, trajectories, per-call timing, diagnostics, split manifests, provenance, package versions, and generated tables. Verify the exported archive can reproduce summary tables.

Unlimited tokens allows larger paired evaluations and real repeatability measurements where useful. Keep request/elapsed-time ceilings to stop runaway loops; make any monetary estimate guard configurable for this account rather than letting V3's old $20 proxy budget determine the study. No API calls are part of this planning step.

Implementation map for a subsequent build:

| Existing area | Proposed change |
|---|---|
| `jevbench/api.py`, `runner.py` | Common decision-provider interface and provider-aware cache/provenance; retain Jev adapter |
| New provider modules | Von/Laya adapters and capability/truncation diagnostics |
| `datasets.py`, new split module | Group/temporal split strategies, timestamp availability assertions, scenario manifests |
| `models.py`, `features.py`, `training.py` | AutoML and embedding baselines; explicit resource and label budgets |
| New simulator module | Seeded episodes, legal actions, transitions, costs, control conditions |
| `decisions.py`, `reporting.py` | Coverage/risk and episode utility; cluster/block paired intervals |
| Notebook builder/config | Separate resumable classification/loop notebooks and immutable run manifests |

First deliverable: comparator adapters and the three continuity datasets. Second: fresh policy and temporal experiments. Third: the loop study. All belong to a new V4 report; preserve the published V3 artifacts. The intended conclusion is a task-specific map of accuracy, robustness, automation risk, and runtime—not one aggregate winner across unrelated tasks.
