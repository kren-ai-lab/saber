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

# pandas.read_csv's default missing-value tokens, so config CSVs keep their meaning.
PANDAS_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
]


def _is_pandas_frame(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "DataFrame" and cls.__module__.split(".")[0] == "pandas"


def as_frame(X: Any) -> Any:
    """Return pandas DataFrames as Polars; leave every other input unchanged."""
    if not _is_pandas_frame(X):
        return X
    try:
        return pl.from_pandas(X)
    except ImportError as exc:
        raise DatasetValidationError(
            "pandas extension dtypes (e.g. 'Int64') need pyarrow to convert; "
            "pass a Polars DataFrame or a NumPy array instead."
        ) from exc


def to_numpy(X: Any) -> np.ndarray:
    """Return the estimator-facing matrix; Polars null becomes NaN."""
    X = as_frame(X)
    if isinstance(X, pl.DataFrame):
        return X.to_numpy()
    return np.asarray(X)


def canonical_dtype(dtype: pl.DataType) -> str:
    """Return the NumPy-style dtype name used by FeatureSchema and fingerprints."""
    name = str(dtype).lower()
    return "bool" if name == "boolean" else name


def missing_mask(values: np.ndarray) -> np.ndarray:
    """Flag None and NaN entries, as ``pandas.isna`` did for these inputs."""
    if values.dtype.kind == "f":
        return np.isnan(values)
    if values.dtype.kind == "O":
        return np.array(
            [v is None or (isinstance(v, float) and math.isnan(v)) for v in values.ravel()], dtype=bool
        ).reshape(values.shape)
    return np.zeros(values.shape, dtype=bool)


def _cell(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return stable_json_dumps(value)
    return value


def records_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a result table from row dicts; nested cells become stable JSON strings."""
    if not rows:
        return pl.DataFrame()
    return pl.from_dicts([{k: _cell(v) for k, v in row.items()} for row in rows], infer_schema_length=None)


def read_table(path: Path, *, separator: str) -> pl.DataFrame:
    """Read a CSV/TSV table; an empty file yields an empty frame."""
    try:
        return pl.read_csv(path, separator=separator, infer_schema_length=None, null_values=PANDAS_NA_VALUES)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()
