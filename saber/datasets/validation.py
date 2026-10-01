"""Validation helpers for supervised dataset contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.utils.multiclass import type_of_target

from saber.exceptions import DatasetValidationError
from saber.utils.tabular import as_frame, missing_mask, python_scalar

if TYPE_CHECKING:
    from collections.abc import Sequence


def validate_feature_matrix(
    X: Any,
    *,
    require_finite: bool = False,
) -> tuple[int, int]:
    """Validate a numerical two-dimensional feature matrix.

    Missing values are allowed by default so leakage-safe preprocessing can impute
    them inside each training fold. Infinite values are always rejected.
    """
    X = as_frame(X)
    if isinstance(X, pl.DataFrame):
        if X.height == 0 or X.width == 0:
            raise DatasetValidationError("X must contain at least one sample and one feature.")

        non_numeric = [
            column
            for column, dtype in zip(X.columns, X.dtypes, strict=True)
            if not (dtype.is_numeric() or dtype == pl.Boolean)
        ]
        if non_numeric:
            raise DatasetValidationError(
                "X must contain only numerical features. Non-numerical columns: " + ", ".join(non_numeric)
            )

        values = X.cast(pl.Float64).to_numpy()
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
    if missing_mask(values).any():
        raise DatasetValidationError("y cannot contain missing target values.")

    if values.dtype.kind == "c":
        raise DatasetValidationError("Complex-valued targets are outside saber scope.")
    if values.dtype.kind == "f" and not np.isfinite(values.astype(float)).all():
        raise DatasetValidationError("y cannot contain infinite target values.")

    for value in values:
        scalar = python_scalar(value)
        if not isinstance(scalar, (str, int, float, bool)):
            raise DatasetValidationError("y must contain scalar str/int/float/bool target values.")

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
        raise DatasetValidationError(f"{name} must contain {n_samples} entries; received {len(array)}.")
    if not allow_missing and missing_mask(array).any():
        raise DatasetValidationError(f"{name} cannot contain missing values.")
    return array


def validate_sample_ids(sample_ids: Sequence[Any], *, n_samples: int) -> tuple[Any, ...]:
    """Validate stable sample identifiers."""
    values = validate_aligned_vector(
        sample_ids,
        n_samples=n_samples,
        name="sample_ids",
    )
    ids = tuple(python_scalar(value) for value in values)

    for sample_id in ids:
        if not isinstance(sample_id, (str, int, float, bool)):
            raise DatasetValidationError("sample_ids must use scalar str/int/float/bool identifiers.")

    if len(set(ids)) != len(ids):
        raise DatasetValidationError("sample_ids must be unique.")

    classes = {type(sample_id) for sample_id in ids}
    if len(classes) > 1:
        found = ", ".join(sorted(cls.__name__ for cls in classes))
        raise DatasetValidationError(
            f"sample_ids must all share one type (str, int, float or bool); found: {found}."
        )

    return ids


def validate_groups(groups: Any, *, n_samples: int) -> tuple[Any, ...]:
    """Validate optional group labels."""
    values = validate_aligned_vector(
        groups,
        n_samples=n_samples,
        name="groups",
    )
    groups_tuple = tuple(python_scalar(value) for value in values)

    for group in groups_tuple:
        if not isinstance(group, (str, int, float, bool)):
            raise DatasetValidationError("groups must use scalar str/int/float/bool identifiers.")

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
            if len(set(values.tolist())) < 2:
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

    raise DatasetValidationError("task must be either 'classification' or 'regression'.")
