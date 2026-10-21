import numpy as np
import pytest

from damast.ml.evaluation import classification_metrics, forecast_metrics


def test_classification_metrics_from_a_hand_counted_confusion_matrix():
    classes = ["cargo", "fishing", "tanker"]
    # true:      cargo x3,      fishing x2, tanker x1
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 0, 1]

    metrics = classification_metrics(y_true, y_pred, classes)

    assert metrics["confusion_matrix"] == [[2, 1, 0], [1, 1, 0], [0, 1, 0]]
    assert metrics["windows"] == 6
    assert metrics["accuracy"] == pytest.approx(3 / 6)
    cargo, fishing, tanker = (metrics["per_class"][c] for c in classes)
    assert (cargo["precision"], cargo["recall"], cargo["support"]) == (pytest.approx(2 / 3), pytest.approx(2 / 3), 3)
    assert (fishing["precision"], fishing["recall"]) == (pytest.approx(1 / 3), pytest.approx(1 / 2))
    # never predicted: no precision, and so no F1 - it still counts in the macro average
    assert tanker == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2 + 0) / 3)
    fishing_f1 = 2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + fishing_f1 + 0) / 3)


def test_balanced_accuracy_leaves_out_classes_not_present():
    metrics = classification_metrics([0, 0, 1], [0, 0, 1], ["a", "b", "c"])

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["per_class"]["c"]["support"] == 0


def test_forecast_metrics_per_step():
    # two windows, two steps, east/north in km: errors of (3, 4) -> 5 km and (0, 0) -> 0 km at step 1
    y_true = np.zeros((2, 2, 2))
    y_pred = np.array([[[3.0, 4.0], [1.0, 0.0]],
                       [[0.0, 0.0], [0.0, 1.0]]])

    metrics = forecast_metrics(y_true, y_pred, targets=["east", "north"])

    first, second = metrics["steps"]
    assert first["euclidean"] == {"mean": 2.5, "median": 2.5, "p90": pytest.approx(4.5), "max": 5.0}
    assert first["mae"] == {"east": 1.5, "north": 2.0}
    assert first["rmse"]["east"] == pytest.approx(np.sqrt(4.5))
    assert second["euclidean"]["mean"] == 1.0
    assert metrics["windows"] == 2


def test_forecast_metrics_reject_mismatching_shapes():
    with pytest.raises(ValueError, match="expected two arrays"):
        forecast_metrics(np.zeros((2, 1, 2)), np.zeros((2, 2)), targets=["a", "b"])


def test_forecast_metrics_of_no_window():
    assert forecast_metrics(np.zeros((0, 2, 2)), np.zeros((0, 2, 2)), ["a", "b"])["steps"] == []
