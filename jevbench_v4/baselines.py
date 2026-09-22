"""Small optional comparators fitted only for the selected task."""
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits

from .contracts import canonical, digest
from .data import case_context, load_job
from .metrics import summarize
from .storage import environment, freeze, write_json

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class FrozenSentenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name=EMBEDDING_MODEL, revision=EMBEDDING_REVISION):
        self.model_name, self.revision = model_name, revision

    def fit(self, x, y=None):
        from sentence_transformers import SentenceTransformer
        self.model_ = SentenceTransformer(self.model_name, revision=self.revision, device="cpu")
        return self

    def transform(self, x):
        return self.model_.encode(list(x), batch_size=128, normalize_embeddings=True,
                                  show_progress_bar=False)


def candidates(kind, name, frame, seed, threads):
    if name == "majority":
        return [("majority", DummyClassifier(strategy="most_frequent"))]
    if name == "svm" and kind == "text":
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=10000, sublinear_tf=True))])
        return [(f"C={c}, class_weight={weight}", make_pipeline(clone(features),
                 LinearSVC(C=c, class_weight=weight, random_state=seed, max_iter=10000)))
                for c in (.3, 3.0) for weight in (None, "balanced")]
    if name == "embedding" and kind == "text":
        return [(f"frozen={EMBEDDING_MODEL}@{EMBEDDING_REVISION}, C={c}",
                 make_pipeline(FrozenSentenceEncoder(), LogisticRegression(
                     C=c, class_weight="balanced", random_state=seed, max_iter=3000)))
                for c in (.3, 3.0)]
    if name == "svm" and kind == "tabular":
        numeric = list(frame.select_dtypes(include=[np.number]).columns)
        categorical = [col for col in frame if col not in numeric]
        prep = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                          OneHotEncoder(handle_unknown="ignore")), categorical)])
        return [(f"C={c}, class_weight={weight}", make_pipeline(clone(prep),
                 SVC(C=c, kernel="rbf", class_weight=weight, cache_size=512, random_state=seed)))
                for c in (.3, 3.0) for weight in (None, "balanced")]
    if name == "catboost" and kind == "tabular":
        from catboost import CatBoostClassifier
        cats = [col for col in frame if not np.issubdtype(frame[col].dtype, np.number)]
        return [(f"depth={depth}, weights={weight}", CatBoostClassifier(iterations=150,
                 depth=depth, auto_class_weights=weight, cat_features=cats, thread_count=threads,
                 random_seed=seed, verbose=False, allow_writing_files=False))
                for depth in (4, 7) for weight in ("None", "Balanced")]
    raise ValueError(f"{name} is not a {kind} baseline")


def run_baseline(root, dataset, seed, name="svm", threads=2):
    frame, metadata, split = load_job(root, dataset, seed)
    x = frame["text"] if metadata["kind"] == "text" else frame[metadata["features"]].copy()
    if metadata.get("suite") == "policy":
        x = metadata["task"] + "\n" + x
    if name == "catboost":
        for col in x.select_dtypes(exclude=[np.number]):
            x[col] = x[col].fillna("__MISSING__").astype(str)
    y = frame.label.to_numpy()
    parts = {key: np.array(value["indices"]) for key, value in split["partitions"].items()}
    train, selection, test = (parts[key] for key in ("train", "validation", "test"))
    job = Path(root) / "baselines" / name / dataset.replace(" ", "_") / str(seed)
    from .storage import read_json
    run = read_json(Path(root) / "run.json")
    freeze(job / "job.json", {"run_id": run["run_id"], "snapshot_sha256": metadata["sha256"],
          "baseline": name, "threads": threads, "environment": environment(), "split": digest(split)})
    if (job / "summary.json").exists():
        return read_json(job / "summary.json")
    started = time.perf_counter()
    scores, fitted = [], []
    if name in ("persistence", "seasonal"):
        if metadata.get("suite") != "future":
            raise ValueError(f"{name} is only defined for Bike Demand")
        column = "count_now" if name == "persistence" else "count_lag_168"
        prediction = (x.iloc[test][column].to_numpy() > metadata["threshold"]).astype(int)
        scores = [{"candidate": column + ">threshold", "selection_balanced_accuracy": None}]
        probabilities = None
        fit_seconds = 0.0
        best = 0
    else:
        with threadpool_limits(limits=threads):
            for description, model in candidates(metadata["kind"], name, x, seed, threads):
                model.fit(x.iloc[train], y[train])
                score = float(balanced_accuracy_score(y[selection], model.predict(x.iloc[selection])))
                scores.append({"candidate": description, "selection_balanced_accuracy": score})
                fitted.append(model)
            best = max(range(len(scores)), key=lambda i: scores[i]["selection_balanced_accuracy"])
            model = clone(fitted[best])
            refit = np.r_[train, selection]
            model.fit(x.iloc[refit], y[refit])
            fit_seconds = time.perf_counter() - started
            prediction = np.asarray(model.predict(x.iloc[test])).ravel().astype(int)
            probabilities = model.predict_proba(x.iloc[test]) if hasattr(model, "predict_proba") else None
    records = [{"case_id": case, "label": int(y[row]), "prediction": int(pred), "status": "ok",
                "probabilities": probabilities[i].tolist() if probabilities is not None else None,
                "cache_hit": False, "latency_ms": None, **case_context(frame, row)}
               for i, (case, row, pred) in enumerate(zip(split["partitions"]["test"]["case_ids"], test, prediction))]
    summary = {"dataset": dataset, "seed": seed, "partition": "test", "baseline": name, "panel": "raw",
               "suite": run["config"].get("suite", "continuity"), "status": "draft-evaluation",
               "label_status": metadata.get("label_status", "public-dataset"),
               "fit_seconds": fit_seconds, "selected": scores[best], "training_labels": len(train),
               "selection_labels": len(selection), "policy_labels_used": 0,
               "probability_kind": "native" if probabilities is not None else "unavailable-svm-margins",
               **summarize(records, len(metadata["labels"]))}
    write_json(job / "trials.json", scores)
    (job / "predictions.jsonl").write_text("".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    write_json(job / "summary.json", summary)
    print(f"{name} / {dataset}: balanced accuracy {summary['balanced_accuracy']}", flush=True)
    return summary


def run_automl(root, dataset, seed, threads=4, time_limit=1800):
    """AutoGluon gets only the frozen train and forward validation partitions."""
    if dataset != "Bike Demand":
        raise ValueError("AutoML is confined to the new future-prediction task")
    from autogluon.tabular import TabularPredictor
    frame, metadata, split = load_job(root, dataset, seed)
    parts = {key: np.array(value["indices"]) for key, value in split["partitions"].items()}
    features = metadata["features"]
    train = frame.iloc[parts["train"]][features + ["label"]].copy()
    tuning = frame.iloc[parts["validation"]][features + ["label"]].copy()
    test = frame.iloc[parts["test"]][features].copy()
    job = Path(root) / "baselines" / "autogluon" / dataset.replace(" ", "_") / str(seed)
    from .storage import read_json
    run = read_json(Path(root) / "run.json")
    freeze(job / "job.json", {"run_id": run["run_id"], "snapshot_sha256": metadata["sha256"],
          "baseline": "autogluon", "threads": threads, "time_limit": time_limit,
          "environment": environment(), "split": digest(split)})
    if (job / "summary.json").exists():
        return read_json(job / "summary.json")
    started = time.perf_counter()
    predictor = TabularPredictor(label="label", problem_type="binary", eval_metric="balanced_accuracy",
                                 path=str(job / "model"),
                                 verbosity=2).fit(
        train_data=train, tuning_data=tuning, time_limit=time_limit, presets="medium_quality",
        num_cpus=threads, num_gpus=0, num_bag_folds=0, num_stack_levels=0,
        dynamic_stacking=False, calibrate_decision_threshold=False)
    prediction = predictor.predict(test).to_numpy().astype(int)
    proba = predictor.predict_proba(test).to_numpy()
    fit_seconds = time.perf_counter() - started
    rows = parts["test"]
    records = [{"case_id": case, "label": int(frame.iloc[row].label), "prediction": int(pred),
                "status": "ok", "probabilities": proba[i].tolist(), "cache_hit": False,
                "latency_ms": None, **case_context(frame, row)}
               for i, (case, row, pred) in enumerate(zip(split["partitions"]["test"]["case_ids"], rows, prediction))]
    summary = {"dataset": dataset, "seed": seed, "partition": "test", "baseline": "autogluon",
               "panel": "raw", "suite": "full", "status": "draft-evaluation",
               "label_status": metadata["label_status"], "fit_seconds": fit_seconds,
               "selected": {"model": predictor.model_best, "time_limit": time_limit},
               "training_labels": len(train), "selection_labels": len(tuning), "policy_labels_used": 0,
               "probability_kind": "native", **summarize(records, 2)}
    (job / "predictions.jsonl").write_text("".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    write_json(job / "trials.json", predictor.leaderboard(silent=True).to_dict(orient="records"))
    write_json(job / "summary.json", summary)
    return summary
