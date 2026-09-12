from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from mlcore.evaluation.classification import evaluate_multiclass_classification


def test_canonical_multiclass_precision_recall_f1_are_weighted() -> None:
    y_true = np.asarray([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.asarray([0, 0, 1, 1, 2, 2, 2, 0])
    result = evaluate_multiclass_classification(
        y_true,
        y_pred,
        metrics=("precision", "recall", "f1"),
    )
    assert result["precision"] == precision_score(y_true, y_pred, average="weighted", zero_division=0)
    assert result["recall"] == recall_score(y_true, y_pred, average="weighted", zero_division=0)
    assert result["f1"] == f1_score(y_true, y_pred, average="weighted", zero_division=0)
