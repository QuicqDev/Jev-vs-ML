# V4 additions: alternatives, generalization, and iterative decisions

Status: proposed study, not a frozen protocol or an executed benchmark. Updated 2026-09-21 against the Reddit comments supplied by the user. V4 adds new experiments; **do not rerun V3 datasets, Jev results, or the eleven-family ML benchmark**. The published V3 release remains historical context. Run **two independent notebooks in parallel**, one for Jev and one for local models, over the same frozen new cases. Jev tokens are currently unlimited; runtime, GPU memory, API throughput, and quality of evidence remain practical limits.

## Reddit feedback to experiment mapping

The user supplied these comments directly; no Reddit thread URL was provided. The hypotheses below must be tested, not assumed true.

| Comment / question | New V4 experiment | Evidence needed |
|---|---|---|
| Compare Jev with open-source Von and Laya | Include pinned Von and routed Laya alongside Jev in all three new tracks where inputs are supported | Matched cases, full input/choice preservation, recorded routing and failures |
| Compare with an AutoML framework | AutoGluon Tabular on the new hourly demand task | Explicit fit budget, past-only selection, same features and future test windows |
| Include weaknesses of SVM; the holdout may permit interpolation | New policy pairs with negation, exceptions, chronology, paraphrases, and held-out combinations | SVM plus frozen-embedding logistic regression; both-members-correct rate by family and composition |
| Random time-series holdouts say little about the future | Train in the past and test on three later, nonoverlapping windows | Future balanced accuracy versus persistence, seasonal, SVM, CatBoost, and AutoML baselines; random-split diagnostic kept separate |
| Would RL training help iterative problems? | New support simulator with bounded inspections and terminal actions | Success and utility across one-shot, feedback, repeated-no-evidence, and fixed-evidence conditions |

The first implementation now has provider adapters, a **draft** policy-pair generator, pair metrics, and two notebook workers. AutoML, embeddings, temporal splits, simulator conditions, and paired uncertainty are still to be implemented. A checked box for infrastructure is not an answered research question.

## Questions and interpretation

1. How does Jev compare with local decision models and a stronger automated ML baseline on new tasks?
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
| CatBoost and scaled RBF SVM | Conventional tabular baselines | New temporal task |
| AutoGluon Tabular | Automated model selection and ensembling | New temporal task |
| Majority / persistence / explicit rules | Sanity and task-specific baselines | As applicable |

Fit these targeted comparators only on the new tasks that need them. Do not refit the completed V3 benchmark. Freeze one sentence encoder by revision after a compatibility smoke test, without selecting it using test performance. Log its task-specific labels and pretrained status.

[Von](https://github.com/wfzyx/von) exposes Choice, Noul, and Score interfaces similar to Jev. Treat repository accuracy, calibration, and latency claims as hypotheses to measure independently.

[Laya](https://github.com/NandhaKishorM/laya) documents multiple checkpoints, a router, AG News in its training mix, and option/context-budget limitations. Its typed-decisions checkpoint has task-specific training. Freeze routing before evaluation; report checkpoint and known training exposure. Do not select a checkpoint per test result or silently truncate choices. The default routed system is the main entry; a fixed checkpoint is a separately named diagnostic.

## Core experiment matrix

| Experiment | Proposed data and size | Main question / primary measure |
|---|---|---|
| Fresh policy decisions | 1,200 cases in 600 related pairs, divided equally between familiar and held-out composition families | Negation, exceptions, chronology, paraphrase; accuracy and both-members-correct pair rate |
| Future prediction | Hourly Bike Sharing, three forward windows, target 500 consecutive test cases/window | Does performance survive a real temporal boundary? Balanced accuracy versus persistence/seasonality |
| Iterative decisions | 500 held-out support episodes, at most four model decisions/episode | Success and net utility after tool feedback; cost of mistakes and unnecessary steps |

V3's IMDb, Banking77, Bank Marketing, and other original datasets are outside this matrix. Use existing published scores only in a clearly separate background section. The original per-case predictions and snapshots are unavailable in the release; aggregate CSVs cannot support paired inference against new predictions. Their absence does not trigger a rerun of the old benchmark.

Use up to 8,000 training and 1,000 selection rows for the new temporal task where timestamp coverage permits, plus a separate policy block of up to 500 rows. Freeze actual counts and IDs after data-only feasibility checks. For policy pairs, the draft study preset has 1,200 training, 320 selection, 320 policy, and 1,200 test observations, grouped into pairs. Half the test pairs use familiar combinations and half held-out combinations and wording. Pilot counts are 80/32/32/64 observations. These are generation targets pending label and distribution review.

### Fresh policy decisions

Build a small refund/support policy world with an executable rule-based labeler and varied natural-language observations. Include four predeclared families:

- Negation or word-order changes that flip the correct action despite similar vocabulary.
- A general policy with an exception that changes the outcome.
- Event ordering, such as cancellation before versus after dispatch.
- Meaning-preserving paraphrases and irrelevant details that should preserve the action.

Include straightforward cases too. Reserve unseen combinations and wording families before evaluating any model. Keep every pair and all derivatives of a base scenario within one partition. An independent human review should check a fixed sample for ambiguity and agreement with the executable labels before the test is frozen. Do not use Jev or a competitor as the sole test-label judge.

The implemented draft uses a refund policy with cancellation-before-dispatch precedence, an inclusive 30-day window, a defect exception, and unopened/final-sale rules. Paired edits can flip the action or preserve it when another rule overrides the edited fact. Each model sees the policy and rendered observations; structured oracle facts and pair metadata are withheld. This small template generator needs expanded wording and independent review before it can support a publication claim. Synthetic success does not settle natural distribution shift.

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

Use a version-pinned AutoGluon Tabular run with a proposed **30-minute fit budget per fit** and balanced accuracy as the selection objective. For temporal data, use explicit past-to-future selection data and disable bagging/stacking initially unless the pinned version's forward-only behavior is verified. Keep policy and test data outside AutoML; disable its automatic decision-threshold adjustment and apply the shared external policy procedure. Record the actual preset, model inventory, hardware, and elapsed time. [AutoGluon fit documentation](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html)

Three future windows require three primary AutoML fits: a nominal 90-minute fit allocation. A separately labeled random-split diagnostic would add its own fits and budget on this new dataset. Downloads, preprocessing, overruns, and inference add time; this is not a session-runtime guarantee. A second 60-minute budget belongs in a predeclared extension across all selected windows, not only a window whose result disappointed us. Do not describe a bounded AutoML run as the best achievable ML.

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

- Main pretrained comparison: fixed prompts without task examples. Secondary few-shot comparison: one example/class on compact tasks that fit all models. Freeze identical examples and label descriptions. Few-shot tests are optional additions after the main three tracks, not a reason to repeat V3.
- Separate raw/default outputs from supervised threshold adjustment. Report training, selection, examples, and policy label counts separately; threshold-adjusted zero-shot prompts do not make a zero-label system.
- Audit serialized inputs and actual tokenization before freezing the run. Preserve every candidate choice and decision-relevant fact. If a model cannot support a task within the declared representation, report unsupported; any shorter-context or hierarchical variant is a separately named system with its full cost.
- Use returned class probabilities for probability metrics, not an arbitrary field named confidence. SVM margins remain margins unless a separate calibration procedure is fitted. Primary probability analysis uses native probabilities; a later calibrated panel needs disjoint or properly cross-fitted calibration/policy data.
- Report balanced accuracy, macro-F1, per-class recall, Brier score and log loss where valid. For automation, report error at matched coverage and coverage at a target error selected on policy data; show attained test error without promising the policy target will hold.
- Count failed/invalid responses in operational accuracy and success. Show probability-metric coverage explicitly; never silently compare only a model's successful subset. Keep failures, retries, truncation, and unsupported tasks visible.
- Use paired intervals: example resampling for independent cases; base-scenario/pair clusters for generated data; time blocks for temporal data; whole episodes for loops. Preserve model pairing. The existing row-stratified bootstrap is not sufficient for every new task.
- Freeze a small list of primary comparisons and report all of them. For formal multiple superiority claims, predeclare a correction such as Holm; otherwise present intervals as exploratory. Training-seed SD is not a test confidence interval. Treat 500 episodes and proposed test sizes as initial study sizes, not a guarantee of resolving small differences; freeze sizes before test inspection.
- Measure uncached end-to-end latency with controlled concurrency, reporting p50/p95, failures and retries. Also show local model-only inference, warm/cold startup, hardware, peak memory, and throughput separately. API network time and local kernel time are different measurements. Unlimited Jev tokens does not mean unlimited request rate; self-hosting still uses compute.

## Kaggle execution and deliverables

1. **Shared preparation:** generate/download only new task data once; validate splits, timestamps, pairs, and simulator transitions. Freeze a common study artifact containing exact inputs, case IDs, episodes, protocol, source, and hashes. Build both notebooks from that artifact. They must not independently resample cases.
2. **Notebook A — Jev:** CPU session with Internet and the API secret. Run compatibility and new-task Jev inference with bounded retries and service-safe pacing. It does no classical training and installs no Von/Laya SDKs.
3. **Notebook B — local models:** two-T4 GPU session. Run Von on GPU 0 and Laya on GPU 1 concurrently in processes that each see only their assigned card. Require two GPUs in the notebook, exercise FP16 CUDA allocation/math before loading, and reject SDK CPU fallback. Save assignment, actual device, per-worker logs, and peak-memory records. The two T4s are not pooled memory; each model must fit independently. Frozen embeddings and CPU-limited SVM/CatBoost/AutoML jobs follow as their implementations become available. If AutoML requires incompatible packages, use an isolated environment within this notebook before starting work.
4. **Parallel evaluation:** both notebooks may run at the same time and have separate checkpoints, caches, logs, and archives. Each performs all conditions assigned to its models, including loop controls when implemented. A development pilot uses only development cases; final package/model revisions and test-family generation must be fixed before the study run.
5. **Combine after completion:** download both archives and merge extracted directories. Require matching study ID, source and dataset hashes, case order, and control definitions. Reject conflicting results instead of overwriting them. Recompute metrics from predictions/trajectories. A local merge command is sufficient; no third model-running notebook is needed.

Unlimited tokens allows larger paired evaluations and real repeatability measurements where useful. Keep request/elapsed-time ceilings to stop runaway loops; make any monetary estimate guard configurable for this account rather than letting V3's old $20 proxy budget determine the study. No API calls are part of this planning step.

Implementation map for a subsequent build:

| Existing area | Proposed change |
|---|---|
| `jevbench_v4/providers.py`, `client.py` | Pinned decision adapters, complete-input audits, provider-aware cache/provenance; implemented |
| `jevbench_v4/policy.py`, `data.py` | Draft policy pairs and group splits implemented; temporal timestamps/windows still needed |
| `jevbench_v4/baselines.py` | New-task SVM/majority implemented; embeddings and AutoML still needed |
| New `jevbench_v4` simulator module | Seeded episodes, legal actions, transitions, costs, control conditions; planned |
| `jevbench_v4/metrics.py`, `export.py`, `workers.py` | Raw/pair metrics and verified worker merge implemented; coverage, utility, cluster/block intervals planned |
| `scripts/build_v4_notebook.py` | Exactly two worker notebooks sharing immutable new-task inputs; implemented for the policy draft |
| `jevbench_v4/parallel.py` | Separate-GPU Von/Laya scheduling, child CUDA preflight, cancellation and process isolation; implemented, real dual-T4 model pilot still pending |

Build order: (1) finish and review policy cases, add the embedding comparator, and validate Von/Laya compatibility; (2) implement temporal windows and AutoML; (3) implement the support simulator and four controls; (4) add paired uncertainty, freeze the complete study, and execute both notebooks. A full V4 report needs all three tracks. Preserve the published V3 artifacts and report task-specific accuracy, robustness, automation risk, and runtime.

## Retained continuity tooling

The earlier draft proposed IMDb 1,000, Banking77 1,500, and Bank Marketing 1,000 test cases with three training seeds. That is **outside the requested V4 scope**. Its loaders, reference baselines, and regression checks remain available as optional diagnostics behind explicit `--suite continuity` commands. Neither new notebook schedules them. Retaining the code does not authorize or require running those old tasks again.
