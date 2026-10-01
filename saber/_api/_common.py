"""Shared helpers for the public high-level API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from saber.core.prediction import PredictionResult, collect_model_outputs
from saber.core.results import TrainResult
from saber.exceptions import ValidationContractError
from saber.persistence import LoadedModelArtifact, load_model
from saber.preprocessing.pipeline import pipeline_input

if TYPE_CHECKING:
    from collections.abc import Sequence


def load_active_model(model: Any) -> TrainResult | LoadedModelArtifact:
    """Resolve an artifact path, TrainResult, or LoadedModelArtifact to a fitted model."""
    if isinstance(model, (str, Path)):
        return load_model(model)
    if isinstance(model, (TrainResult, LoadedModelArtifact)):
        return model
    raise ValidationContractError("model must be an artifact path, LoadedModelArtifact, or TrainResult.")


def predict_model(
    model: TrainResult | LoadedModelArtifact,
    X: Any,
    *,
    feature_names: Sequence[str] | None = None,
    sample_ids: Sequence[Any] | None = None,
    positive_class: Any | None = None,
) -> PredictionResult:
    """Generate a structured prediction from a resolved fitted model."""
    if X is None:
        raise ValidationContractError("Prediction requires a feature matrix or DatasetBundle.")
    if isinstance(model, LoadedModelArtifact):
        return model._predict_result(  # noqa: SLF001  # the artifact's own inference path
            X,
            feature_names=feature_names,
            sample_ids=sample_ids,
            positive_class=positive_class,
        )

    fitted = model.model
    if fitted is None:
        raise ValidationContractError("TrainResult does not contain a fitted model.")

    if model.feature_schema is not None:
        # Validate against the training schema before the NumPy conversion.
        model.feature_schema.validate_compatible(X, feature_names=feature_names)
    X = pipeline_input(fitted, X)
    capabilities = model.spec.resolved_capabilities
    outputs = collect_model_outputs(
        fitted,
        X,
        task=model.spec.task,
        use_proba=capabilities.predict_proba,
        use_decision=capabilities.decision_function,
    )

    return PredictionResult(
        task=model.spec.task,
        **outputs,
        positive_class=model.positive_class if positive_class is None else positive_class,
        sample_ids=None if sample_ids is None else np.asarray(sample_ids),
        metadata={"algorithm": model.spec.name},
    )
