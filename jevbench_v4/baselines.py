"""Small predeclared continuity baselines, selected only on development data."""
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits

from .contracts import canonical, digest
from .data import load_job
from .metrics import summarize
from .storage import environment, freeze, write_json


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
                "cache_hit": False, "latency_ms": None}
               for i, (case, row, pred) in enumerate(zip(split["partitions"]["test"]["case_ids"], test, prediction))]
    summary = {"dataset": dataset, "seed": seed, "partition": "test", "baseline": name, "panel": "raw",
               "fit_seconds": fit_seconds, "selected": scores[best], "training_labels": len(train),
               "selection_labels": len(selection), "policy_labels_used": 0,
               "probability_kind": "native" if probabilities is not None else "unavailable-svm-margins",
               **summarize(records, len(metadata["labels"]))}
    write_json(job / "trials.json", scores)
    (job / "predictions.jsonl").write_text("".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    write_json(job / "summary.json", summary)
    print(f"{name} / {dataset}: balanced accuracy {summary['balanced_accuracy']}", flush=True)
    return summary
