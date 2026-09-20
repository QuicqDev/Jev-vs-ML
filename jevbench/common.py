"""Shared utilities and explicit device probing."""
import concurrent.futures as futures
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import zipfile
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
MODELS = ['Logistic regression', 'SVM', 'Decision tree', 'Random forest',
          'Extra trees', 'k-NN', 'Naive Bayes', 'Hist gradient boost',
          'XGBoost', 'CatBoost', 'Voting ensemble']
ALL_DATASETS = ['AG News', 'Banking77', 'SMS Spam', 'IMDb', 'Bank Marketing',
                'Online Shoppers', 'Breast Cancer', 'Iris']

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)

def environment():
    versions = {}
    for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost', 'scipy', 'huggingface-hub']:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = 'missing'
    # Metadata only: the parent must not initialize CUDA to inspect versions.
    for p in ['cuml', 'cuml-cu12', 'cuml-cu13', 'cupy', 'cupy-cuda12x', 'cupy-cuda13x']:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)

def probe_gpus():
    """Check each visible CUDA ordinal independently with actual library fits."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15)
        names = result.stdout.strip()
        if result.returncode or not names:
            return [], 'CPU (no NVIDIA GPU detected)'
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
        x = np.arange(80, dtype=np.float32).reshape(40, 2)
        y = np.arange(40) % 2
        # CUDA_VISIBLE_DEVICES remaps ordinals. Kaggle normally exposes both T4s.
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        count = len([v for v in visible.split(',') if v.strip() and v.strip() != '-1']) if visible is not None else len(names.splitlines())
        usable, messages = [], []
        for gpu_id in range(count):
            try:
                m = XGBClassifier(n_estimators=2, device=f'cuda:{gpu_id}', tree_method='hist').fit(x, y)
                actual = json.loads(m.get_booster().save_config())['learner']['generic_param']['device']
                if actual != f'cuda:{gpu_id}':
                    raise RuntimeError(f'XGBoost selected {actual}, expected cuda:{gpu_id}')
                CatBoostClassifier(iterations=2, task_type='GPU', devices=str(gpu_id), verbose=False,
                                   allow_writing_files=False).fit(x, y)
                usable.append(gpu_id)
                messages.append(f'GPU {gpu_id}: XGBoost and CatBoost probe passed')
            except Exception as exc:
                messages.append(f'GPU {gpu_id}: excluded ({type(exc).__name__}: {str(exc)[:180]})')
        return usable, names + '\n' + '\n'.join(messages)
    except Exception as exc:
        return [], f'CPU fallback; GPU probe failed: {type(exc).__name__}: {str(exc)[:250]}'

def metrics(y, pred, labels):
    # Failed predictions use -1 and count as incorrect. Never drop failed rows.
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=list(range(len(labels))), average='macro', zero_division=0)))

def hard_vote(predictions, classes):
    stack = np.asarray(predictions)
    counts = np.stack([(stack == c).sum(axis=0) for c in range(classes)], axis=1)
    return counts.argmax(axis=1)

def probability_metrics(y, probabilities):
    p = np.asarray(probabilities)
    onehot = np.eye(p.shape[1])[y]
    confidence, predicted = p.max(axis=1), p.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for b in range(10):
        mask = (confidence >= edges[b]) & ((confidence < edges[b + 1]) if b < 9 else (confidence <= 1))
        if mask.any():
            ece += mask.mean() * abs((predicted[mask] == y[mask]).mean() - confidence[mask].mean())
    return dict(brier=float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
                log_loss_clipped=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
                ece_10_bin=float(ece))
