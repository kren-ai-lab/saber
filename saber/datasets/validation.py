"""Validation helpers for supervised dataset contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.utils.multiclass import type_of_target

from saber.exceptions import DatasetValidationError


def validate_feature_matrix(
    X: Any,
    *,
    require_finite: bool = False,
) -> tuple[int, int]:
    """Validate a numerical two-dimensional feature matrix.

    Missing values are allowed by default so leakage-safe preprocessing can impute
    them inside each training fold. Infinite values are always rejected.
    """

    if isinstance(X, pd.DataFrame):
        if X.ndim != 2:
            raise DatasetValidationError("X must be a two-dimensional feature matrix.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise DatasetValidationError("X must contain at least one sample and one feature.")
        if X.columns.duplicated().any():
            raise DatasetValidationError("DataFrame feature names must be unique.")

        non_numeric = [str(column) for column in X.columns if not is_numeric_dtype(X[column].dtype)]
        if non_numeric:
            raise DatasetValidationError(
                "X must contain only numerical features. Non-numerical columns: "
                + ", ".join(non_numeric)
            )

        values = X.to_numpy(dtype=float, copy=False)
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise DatasetValidationError("X must be a two-dimensional feature matrix.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise DatasetValidationError("X must contain at least one sample and one feature.")
        if array.dtype.kind not in {"b", "i", "u", "f"}:
            raise DatasetValidationError("X must contain only numerical feature values.")
        values = array.astype(float, copy=False)

    if np.isinf(values).any():
        raise DatasetValidationError("X contains positive or negative infinite values.")
    if require_finite and not np.isfinite(values).all():
        raise DatasetValidationError("X must contain only finite values for this workflow.")

    return int(values.shape[0]), int(values.shape[1])


def validate_target(y: Any, *, n_samples: int) -> np.ndarray:
    """Validate and normalize a single-target vector."""

    values = np.asarray(y)
    if values.ndim != 1:
        raise DatasetValidationError(
            "y must be one-dimensional; multi-output targets are outside saber scope."
        )
    if len(values) != n_samples:
        raise DatasetValidationError(
            f"X and y are misaligned: X has {n_samples} samples but y has {len(values)}."
        )
    if len(values) == 0:
        raise DatasetValidationError("y must contain at least one target value.")

    if pd.isna(values).any():
        raise DatasetValidationError("y cannot contain missing target values.")

    if values.dtype.kind == "c":
        raise DatasetValidationError("Complex-valued targets are outside saber scope.")
    if values.dtype.kind == "f" and not np.isfinite(values.astype(float)).all():
        raise DatasetValidationError("y cannot contain infinite target values.")

    for value in values:
        scalar = _to_python_scalar(value)
        if not isinstance(scalar, (str, int, float, bool)):
            raise DatasetValidationError(
                "y must contain scalar str/int/float/bool target values."
            )

    return values


def validate_aligned_vector(
    values: Any,
    *,
    n_samples: int,
    name: str,
    allow_missing: bool = False,
) -> np.ndarray:
    """Validate a one-dimensional vector aligned to dataset samples."""

    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise DatasetValidationError(f"{name} must be one-dimensional.")
    if len(array) != n_samples:
        raise DatasetValidationError(
            f"{name} must contain {n_samples} entries; received {len(array)}."
        )
    if not allow_missing and pd.isna(array).any():
        raise DatasetValidationError(f"{name} cannot contain missing values.")
    return array


def validate_sample_ids(sample_ids: Sequence[Any], *, n_samples: int) -> tuple[Any, ...]:
    """Validate stable sample identifiers."""

    values = validate_aligned_vector(
        sample_ids,
        n_samples=n_samples,
        name="sample_ids",
    )
    ids = tuple(_to_python_scalar(value) for value in values)

    try:
        unique_count = len(set(ids))
    except TypeError as exc:
        raise DatasetValidationError("sample_ids must contain hashable scalar values.") from exc

    if unique_count != len(ids):
        raise DatasetValidationError("sample_ids must be unique.")

    for sample_id in ids:
        if not isinstance(sample_id, (str, int, float, bool)):
            raise DatasetValidationError(
                "sample_ids must use scalar str/int/float/bool identifiers."
            )

    return ids


def validate_groups(groups: Any, *, n_samples: int) -> tuple[Any, ...]:
    """Validate optional group labels."""

    values = validate_aligned_vector(
        groups,
        n_samples=n_samples,
        name="groups",
    )
    groups_tuple = tuple(_to_python_scalar(value) for value in values)

    for group in groups_tuple:
        if not isinstance(group, (str, int, float, bool)):
            raise DatasetValidationError(
                "groups must use scalar str/int/float/bool identifiers."
            )

    return groups_tuple


def validate_sample_weight(sample_weight: Any, *, n_samples: int) -> np.ndarray:
    """Validate optional non-negative finite sample weights."""

    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim != 1:
        raise DatasetValidationError("sample_weight must be one-dimensional.")
    if len(weights) != n_samples:
        raise DatasetValidationError(
            f"sample_weight must contain {n_samples} entries; received {len(weights)}."
        )
    if not np.isfinite(weights).all():
        raise DatasetValidationError("sample_weight must contain only finite values.")
    if (weights < 0).any():
        raise DatasetValidationError("sample_weight cannot contain negative values.")
    if not (weights > 0).any():
        raise DatasetValidationError("sample_weight must contain at least one positive value.")
    return weights


def validate_target_for_task(y: Any, task: str) -> str:
    """Validate target semantics for classification or regression.

    Returns ``binary``, ``multiclass``, or ``regression``.
    """

    values = np.asarray(y)

    if task == "classification":
        try:
            target_type = type_of_target(values)
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(
                "Classification targets must use a supported one-dimensional "
                "binary or multiclass label representation."
            ) from exc

        if target_type == "binary":
            if len(pd.unique(values)) < 2:
                raise DatasetValidationError(
                    "Classification datasets must contain at least two target classes."
                )
            return "binary"
        if target_type == "multiclass":
            return "multiclass"

        raise DatasetValidationError(
            "Classification targets must be binary or multiclass labels; "
            f"received target type '{target_type}'."
        )

    if task == "regression":
        try:
            numeric = values.astype(float)
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError("Regression targets must be numeric.") from exc
        if not np.isfinite(numeric).all():
            raise DatasetValidationError("Regression targets must be finite.")
        return "regression"

    raise DatasetValidationError(
        "task must be either 'classification' or 'regression'."
    )


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
