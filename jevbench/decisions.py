"""Decision-policy selection never receives test labels."""
from .common import *
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score


def outputs(model, x, n_classes):
    pred = np.asarray(model.predict(x)).ravel().astype(int)
    if hasattr(model, 'predict_proba') and getattr(model, 'probability', True):
        p = np.asarray(model.predict_proba(x))
        return dict(raw=pred, score=p[:, 1] if n_classes == 2 else p,
                    probability=p, default_threshold=.5)
    score = np.asarray(model.decision_function(x))
    if n_classes == 2:
        score = score.ravel()
    return dict(raw=pred, score=score, probability=None, default_threshold=0.0)


def choose_threshold(y_policy, scores, default, quantiles=101):
    y_policy, scores = np.asarray(y_policy), np.asarray(scores)
    if not np.isfinite(scores).all() or set(y_policy) != {0, 1}:
        raise ValueError('Threshold selection requires finite binary scores with both classes')
    # A fixed quantile grid accommodates both margins and probabilities.
    thresholds = np.unique(np.r_[default, np.quantile(scores, np.linspace(0, 1, quantiles)),
                                  np.nextafter(scores.max(), np.inf)])
    values = [float(balanced_accuracy_score(y_policy, scores >= t)) for t in thresholds]
    best = max(range(len(thresholds)), key=lambda i: (values[i], -abs(thresholds[i] - default)))
    return dict(threshold=float(thresholds[best]), selection_balanced_accuracy=values[best],
                grid_size=len(thresholds), selection_partition='policy', positive_class=1)


def score_result(y, pred, n_classes, scores=None, probabilities=None):
    result = metrics(y, pred, list(range(n_classes)))
    result['per_class_recall'] = recall_score(y, pred, labels=list(range(n_classes)), average=None, zero_division=0).tolist()
    result['confusion_matrix'] = confusion_matrix(y, pred, labels=list(range(n_classes)) + ([-1] if -1 in pred else [])).tolist()
    result['predicted_class_counts'] = {str(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))}
    if n_classes == 2 and scores is not None and np.isfinite(scores).all():
        result.update(roc_auc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)))
    if probabilities is not None and np.isfinite(probabilities).all():
        result.update(probability_metrics(np.asarray(y), probabilities))
    return result
