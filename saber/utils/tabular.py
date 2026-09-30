"""Tabular boundary: Polars inside saber, NumPy towards estimators.

pandas is not a dependency. A pandas DataFrame is accepted at public entry
points and converted once, here; ``pl.from_pandas`` only imports pandas when
it receives a pandas object.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from saber.exceptions import DatasetValidationError
from saber.utils.serialization import stable_json_dumps

if TYPE_CHECKING:
    from pathlib import Path

# Cell values read as missing in CSV/TSV tables: the common spellings, not pandas' full list.
MISSING_VALUE_TOKENS = ["", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "None", "#N/A"]


def _is_pandas_frame(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "DataFrame" and cls.__module__.split(".")[0] == "pandas"


def as_frame(X: Any) -> Any:
    """Return DataFrames as Polars with missing floats as null; leave other inputs unchanged."""
    if isinstance(X, pl.DataFrame):
        return _nan_to_null(X)
    if not _is_pandas_frame(X):
        return X
    if len({str(column) for column in X.columns}) != len(X.columns):
        raise DatasetValidationError("DataFrame feature names must be unique.")
    try:
        frame = pl.from_pandas(X)
    except ImportError:
        # No pyarrow: convert column by column without it and without importing
        # pandas. Non-numeric columns (e.g. plain strings) are then rejected by
        # the normal validate_feature_matrix path, with its usual message.
        frame = pl.DataFrame(_pandas_columns_without_pyarrow(X))
    return _nan_to_null(frame)


def _nan_to_null(frame: pl.DataFrame) -> pl.DataFrame:
    """Store float NaN as Polars null, the tabular layer's single missing marker."""
    return frame.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))


def _pandas_columns_without_pyarrow(X: Any) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for column in X.columns:
        series = X[column]
        if isinstance(series.dtype, np.dtype):
            columns[str(column)] = series.to_numpy()
        else:
            # A pandas extension dtype (e.g. "Int64", "boolean", "string") is not a
            # plain numpy dtype; duck-typing this way needs no pandas import here.
            columns[str(column)] = series.astype(object).where(series.notna(), None).tolist()
    return columns


def to_numpy(X: Any) -> np.ndarray:
    """Return the estimator-facing matrix; Polars null becomes NaN."""
    X = as_frame(X)
    if isinstance(X, pl.DataFrame):
        return X.to_numpy()
    return np.asarray(X)


def canonical_dtype(dtype: pl.DataType | type[pl.DataType]) -> str:
    """Return the NumPy-style dtype name used by FeatureSchema and fingerprints."""
    name = str(dtype).lower()
    return "bool" if name == "boolean" else name


def missing_mask(values: np.ndarray) -> np.ndarray:
    """Flag None and NaN entries, as ``pandas.isna`` did for these inputs."""
    if values.dtype.kind == "f":
        return np.isnan(values)
    if values.dtype.kind == "O":
        return np.array(
            [v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)) for v in values.ravel()],
            dtype=bool,
        ).reshape(values.shape)
    return np.zeros(values.shape, dtype=bool)


def cell_value(value: Any) -> Any:
    """Return a result-table cell: NumPy scalars as Python, nested values as stable JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return stable_json_dumps(value)
    return value


def records_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a result table from row dicts; nested cells become stable JSON strings."""
    if not rows:
        return pl.DataFrame()
    return pl.from_dicts(
        [{k: cell_value(v) for k, v in row.items()} for row in rows], infer_schema_length=None
    )


def read_table(path: Path, *, separator: str) -> pl.DataFrame:
    """Read a CSV/TSV table; an empty file yields an empty frame."""
    try:
        frame = pl.read_csv(
            path, separator=separator, infer_schema_length=None, null_values=MISSING_VALUE_TOKENS
        )
    except pl.exceptions.NoDataError:
        return pl.DataFrame()
    height = frame.height
    all_null_columns = [
        name
        for name, dtype in frame.schema.items()
        if height > 0 and dtype in (pl.String, pl.Null) and frame[name].null_count() == height
    ]
    if not all_null_columns:
        return frame
    return frame.with_columns(pl.col(name).cast(pl.Float64) for name in all_null_columns)
