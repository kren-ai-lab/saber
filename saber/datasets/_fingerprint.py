"""Deterministic hashing helpers for dataset and partition contracts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from saber.utils.tabular import canonical_dtype

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_FINGERPRINT_VERSION = "saber-fingerprint-v1"


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar_token(value: Any) -> list[Any]:
    """Return a JSON-safe, type-aware token for a scalar value."""
    value = _python_scalar(value)

    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "inf" if value > 0 else "-inf"]
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]

    raise TypeError(
        f"Fingerprint values must be scalar None/bool/int/float/str values; received {type(value).__name__}."
    )


def _update_json(hasher: Any, payload: Any) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher.update(encoded)


def _update_sequence(hasher: Any, label: str, values: Iterable[Any]) -> None:
    hasher.update(label.encode("utf-8"))
    for value in values:
        _update_json(hasher, _scalar_token(value))


def _normalized_numeric_bytes(values: np.ndarray) -> bytes:
    """Return stable bytes for a numeric array independent of byte order."""
    array = np.asarray(values)

    if array.dtype.kind == "f":
        normalized = np.asarray(array, dtype="<f8").copy()
        # Normalize all NaN payload representations before hashing.
        normalized[np.isnan(normalized)] = np.nan
        return np.ascontiguousarray(normalized).tobytes(order="C")

    if array.dtype.kind in {"i", "u"}:
        normalized = np.asarray(array, dtype="<i8" if array.dtype.kind == "i" else "<u8")
        return np.ascontiguousarray(normalized).tobytes(order="C")

    if array.dtype.kind == "b":
        normalized = np.asarray(array, dtype=np.uint8)
        return np.ascontiguousarray(normalized).tobytes(order="C")

    # Nullable integer columns materialize as float with NaN.
    try:
        normalized = np.asarray(array, dtype="<f8").copy()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported numeric dtype for fingerprinting: {array.dtype}.") from exc
    normalized[np.isnan(normalized)] = np.nan
    return np.ascontiguousarray(normalized).tobytes(order="C")


def update_feature_matrix(hasher: Any, X: Any) -> None:
    """Update a hasher from a validated numerical feature matrix."""
    hasher.update(b"X")

    if isinstance(X, pl.DataFrame):
        _update_json(hasher, ["dataframe", X.shape])
        for column in X.columns:
            series = X[column]
            _update_json(hasher, [column, canonical_dtype(series.dtype)])
            hasher.update(_normalized_numeric_bytes(series.to_numpy()))
        return

    array = np.asarray(X)
    _update_json(hasher, ["ndarray", array.shape, str(array.dtype)])
    hasher.update(_normalized_numeric_bytes(array))


def dataset_fingerprint(
    *,
    X: Any,
    y: Sequence[Any],
    sample_ids: Sequence[Any],
    feature_names: Sequence[str],
    groups: Sequence[Any] | None = None,
    sample_weight: Sequence[float] | None = None,
) -> str:
    """Create a deterministic content fingerprint for a supervised dataset."""
    hasher = hashlib.sha256()
    hasher.update(_FINGERPRINT_VERSION.encode("utf-8"))
    update_feature_matrix(hasher, X)
    _update_sequence(hasher, "y", y)
    _update_sequence(hasher, "sample_ids", sample_ids)
    _update_sequence(hasher, "feature_names", feature_names)

    if groups is None:
        hasher.update(b"groups:none")
    else:
        _update_sequence(hasher, "groups", groups)

    if sample_weight is None:
        hasher.update(b"sample_weight:none")
    else:
        _update_sequence(hasher, "sample_weight", sample_weight)

    return hasher.hexdigest()


def feature_schema_fingerprint(
    *,
    names: Sequence[str],
    dtypes: Sequence[str],
) -> str:
    """Create a deterministic fingerprint for a feature schema."""
    hasher = hashlib.sha256()
    hasher.update(_FINGERPRINT_VERSION.encode("utf-8"))
    _update_json(
        hasher,
        {
            "kind": "feature_schema",
            "names": list(names),
            "dtypes": list(dtypes),
        },
    )
    return hasher.hexdigest()


def partition_fingerprint(
    *,
    kind: str,
    splits: Sequence[Mapping[str, Any]],
    dataset_fingerprint_value: str | None,
) -> str:
    """Create a membership-based deterministic partition fingerprint.

    Membership order inside each role is intentionally ignored. Split names are
    semantic identifiers and therefore participate in the fingerprint.
    """
    canonical_splits: list[dict[str, Any]] = []

    for split in sorted(splits, key=lambda item: str(item["name"])):
        canonical: dict[str, Any] = {"name": str(split["name"])}
        for role in ("train_ids", "validation_ids", "test_ids"):
            values = split.get(role, ())
            tokens = [_scalar_token(value) for value in values]
            tokens.sort(key=lambda token: json.dumps(token, separators=(",", ":")))
            canonical[role] = tokens
        canonical_splits.append(canonical)

    hasher = hashlib.sha256()
    hasher.update(_FINGERPRINT_VERSION.encode("utf-8"))
    _update_json(
        hasher,
        {
            "kind": kind,
            "dataset_fingerprint": dataset_fingerprint_value,
            "splits": canonical_splits,
        },
    )
    return hasher.hexdigest()
