"""High-level structured prediction API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saber._api._common import load_active_model, predict_model
from saber.datasets import DatasetBundle

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from saber.core.prediction import PredictionResult
    from saber.core.results import TrainResult
    from saber.persistence import LoadedModelArtifact


def predict(
    model: str | Path | TrainResult | LoadedModelArtifact,
    X: Any,
    *,
    sample_ids: Sequence[Any] | None = None,
    positive_class: Any | None = None,
) -> PredictionResult:
    """Generate a structured PredictionResult from a fitted or persisted model.

    ``X`` is a :class:`DatasetBundle`, a NumPy array, or a Polars/pandas
    DataFrame.  DataFrames and datasets are checked against the training
    feature names/order; a NumPy array only against the feature count.  A path
    is loaded with :func:`saber.load_model` defaults; call ``load_model``
    yourself for ``strict_environment`` or ``verify`` control.
    """
    active = load_active_model(model)
    feature_names = None
    if isinstance(X, DatasetBundle):
        feature_names = X.feature_names
        if sample_ids is None:
            sample_ids = X.sample_ids
        X = X.X
    return predict_model(
        active,
        X,
        feature_names=feature_names,
        sample_ids=sample_ids,
        positive_class=positive_class,
    )


__all__ = ["predict"]
