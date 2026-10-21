"""
Metrics for evaluating the predictions of a trained model on a fixed set of windows.

They are computed from the predictions rather than by keras, so that a single deterministic pass yields
the same numbers in every evaluation - on the test groups of an experiment as on another dataset - and
so that they cover what keras' metrics do not, e.g. a confusion matrix or the error per forecast step.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["classification_metrics", "forecast_metrics"]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[Any]) -> dict[str, Any]:
    """
    Metrics of a classification, with each window counted once.

    :param y_true: True class index per window
    :param y_pred: Predicted class index per window
    :param classes: The classes, in the order of their index
    :return: ``accuracy``, ``balanced_accuracy`` (mean recall over the classes present), ``macro_f1``
        (mean F1 over all classes), ``per_class`` precision/recall/F1/support, and the
        ``confusion_matrix`` with a row per true and a column per predicted class
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(classes)
    confusion = np.zeros((n, n), dtype=int)
    np.add.at(confusion, (y_true, y_pred), 1)

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    # a class never predicted has no precision, one never present no recall - both count as 0 in F1
    precision = np.divide(correct, predicted, out=np.zeros(n), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)

    total = int(confusion.sum())
    return {
        "classes": [str(c) for c in classes],
        "windows": total,
        "accuracy": float(correct.sum() / total) if total else None,
        "balanced_accuracy": float(recall[support > 0].mean()) if (support > 0).any() else None,
        "macro_f1": float(f1.mean()),
        "per_class": {str(c): {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion.tolist(),
    }


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> dict[str, Any]:
    """
    Metrics of a forecast, per forecast step.

    The Euclidean error combines all targets of a step into one distance - meaningful when the targets
    are components of one vector in one unit, e.g. a displacement east and north in km.

    :param y_true: True values, of shape :code:`(windows, forecast_length, targets)`
    :param y_pred: Predicted values, of the same shape
    :param targets: Names of the targets
    :return: Per step: ``mae`` and ``rmse`` per target, and the ``euclidean`` error's mean, median, p90
        and max
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise ValueError(f"forecast_metrics: expected two arrays of shape (windows, steps, targets),"
                         f" got {y_true.shape} and {y_pred.shape}")

    if y_true.shape[0] == 0:
        return {"windows": 0, "targets": list(targets), "steps": []}

    error = y_pred - y_true
    euclidean = np.sqrt((error ** 2).sum(axis=2))
    steps = []
    for k in range(y_true.shape[1]):
        steps.append({
            "step": k + 1,
            "mae": {t: float(np.abs(error[:, k, j]).mean()) for j, t in enumerate(targets)},
            "rmse": {t: float(np.sqrt((error[:, k, j] ** 2).mean())) for j, t in enumerate(targets)},
            "euclidean": {"mean": float(euclidean[:, k].mean()), "median": float(np.median(euclidean[:, k])),
                          "p90": float(np.quantile(euclidean[:, k], 0.9)), "max": float(euclidean[:, k].max())},
        })
    return {"windows": int(y_true.shape[0]), "targets": list(targets), "steps": steps}
