"""Leakage-safe numerical preprocessing and estimator pipeline construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from saber.preprocessing.imputation import build_imputer
from saber.preprocessing.scaling import build_scaler, resolve_scaler_name
from saber.preprocessing.validation import validate_estimator_dataset_requirements
from saber.utils.tabular import as_frame, to_numpy

if TYPE_CHECKING:
    from saber.core.specs import AlgorithmSpec
    from saber.datasets.schemas import DatasetBundle


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Numerical preprocessing policy fitted independently inside each split."""

    imputation: str | None = "auto"
    scaler: str | None = "auto"
    fill_value: float | int | str | None = 0.0
    transformer: Any | None = None


def build_model_pipeline(
    *,
    spec: AlgorithmSpec,
    estimator: Any,
    training_data: DatasetBundle,
    preprocessing: PreprocessingConfig | None = None,
) -> Pipeline:
    """Build a fresh sklearn Pipeline for one training split.

    All data-dependent preprocessing is part of the returned pipeline, so fit
    statistics are learned only from the split supplied to ``Pipeline.fit``.
    """
    config = PreprocessingConfig() if preprocessing is None else preprocessing

    if config.transformer is not None:
        transformer = clone(config.transformer)
        validate_estimator_dataset_requirements(
            training_data,
            spec=spec,
            imputation_enabled=False,
            non_negative_transform=False,
            custom_transformer=True,
        )
        return Pipeline(
            [
                ("preprocess", transformer),
                # A custom transformer may emit a DataFrame; the estimator must not.
                ("to_numpy", FunctionTransformer(_frame_to_numpy)),
                ("estimator", estimator),
            ]
        )

    imputation = config.imputation
    if imputation == "auto":
        imputation = None if spec.resolved_capabilities.native_missing_values else "median"

    scaler_name = resolve_scaler_name(
        config.scaler,
        requirements=spec.requirements,
    )

    validate_estimator_dataset_requirements(
        training_data,
        spec=spec,
        imputation_enabled=imputation not in {None, "none"},
        non_negative_transform=scaler_name == "minmax",
        custom_transformer=False,
    )

    imputer = build_imputer(
        imputation,
        fill_value=config.fill_value,
    )
    scaler = build_scaler(
        scaler_name,
        requirements=spec.requirements,
    )

    return Pipeline(
        [
            ("imputer", imputer),
            ("scaler", scaler),
            ("estimator", estimator),
        ]
    )


def pipeline_input(model: Any, X: Any) -> Any:
    """Select the estimator-facing form of ``X`` for a given pipeline.

    A pipeline with a custom "preprocess" step (an arbitrary transformer, e.g.
    a ``ColumnTransformer`` selecting columns by name) needs a DataFrame so
    that column selection by name keeps working; every other pipeline keeps
    the NumPy boundary towards sklearn/XGBoost/LightGBM.
    """
    named_steps = getattr(model, "named_steps", None)
    if named_steps is not None and "preprocess" in named_steps:
        return as_frame(X)
    return to_numpy(X)


def _frame_to_numpy(X: Any) -> Any:
    """Convert a DataFrame to the estimator-facing matrix; pass arrays and sparse output through."""
    X = as_frame(X)
    return to_numpy(X) if isinstance(X, pl.DataFrame) else X
