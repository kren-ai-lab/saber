"""Preprocessing and estimator-input contract validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from saber.exceptions import PreprocessingContractError
from saber.utils.tabular import missing_mask, to_numpy

if TYPE_CHECKING:
    from saber.core.specs import AlgorithmSpec
    from saber.datasets.schemas import DatasetBundle


def has_missing_features(X: Any) -> bool:
    """Return whether a numerical feature matrix contains missing values."""
    return bool(missing_mask(to_numpy(X)).any())


def has_negative_features(X: Any) -> bool:
    """Return whether a numerical feature matrix contains a negative finite value."""
    values = to_numpy(X).astype(float)
    finite = values[np.isfinite(values)]
    return bool(finite.size and (finite < 0).any())


def validate_estimator_dataset_requirements(
    dataset: DatasetBundle,
    *,
    spec: AlgorithmSpec,
    imputation_enabled: bool,
    non_negative_transform: bool,
    custom_transformer: bool = False,
) -> None:
    """Validate dataset constraints that preprocessing must make safe."""
    if spec.requirements.positive_y:
        y = np.asarray(dataset.y, dtype=float)
        if not np.all(y > 0):
            raise PreprocessingContractError(
                f"Algorithm '{spec.name}' requires strictly positive target values."
            )

    if (
        has_missing_features(dataset.X)
        and not imputation_enabled
        and not spec.resolved_capabilities.native_missing_values
        and not custom_transformer
    ):
        raise PreprocessingContractError(
            f"Algorithm '{spec.name}' does not support missing feature values and "
            "no imputation/custom preprocessing was configured."
        )

    if (
        spec.requirements.non_negative_X
        and has_negative_features(dataset.X)
        and not non_negative_transform
        and not custom_transformer
    ):
        raise PreprocessingContractError(
            f"Algorithm '{spec.name}' requires non-negative features. "
            "Use MinMax scaling/auto preprocessing or provide non-negative inputs."
        )
