import numpy as np
import pandas as pd
import polars as pl
import pytest

from saber.datasets.validation import validate_feature_matrix
from saber.exceptions import DatasetValidationError
from saber.utils.tabular import as_frame, canonical_dtype, missing_mask, read_table, records_frame, to_numpy


def _force_no_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``pl.from_pandas`` fail as it would with no pyarrow installed."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ImportError("forced: pyarrow not installed")

    monkeypatch.setattr(pl, "from_pandas", _raise)


def test_as_frame_converts_pandas_and_leaves_others():
    frame = as_frame(pd.DataFrame({"a": [1.0, np.nan]}))
    assert isinstance(frame, pl.DataFrame)
    assert frame["a"].null_count() == 1
    array = np.ones((2, 2))
    assert as_frame(array) is array


def test_as_frame_rejects_duplicate_pandas_column_names():
    frame = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])
    with pytest.raises(DatasetValidationError, match="unique"):
        as_frame(frame)


def test_as_frame_reports_non_numerical_columns_for_object_dtype_without_pyarrow(monkeypatch):
    _force_no_pyarrow(monkeypatch)
    frame = pd.DataFrame({"a": [1.0, 2.0], "s": [{"x": 1}, {"y": 2}]})
    with pytest.raises(DatasetValidationError, match="Non-numerical columns: s"):
        validate_feature_matrix(frame)


def test_as_frame_rejects_pandas_string_columns_without_pyarrow(monkeypatch):
    _force_no_pyarrow(monkeypatch)
    frame = pd.DataFrame({"a": [1.0, 2.0], "s": pd.array(["p", "q"], dtype="string")})
    with pytest.raises(DatasetValidationError, match="Non-numerical columns: s"):
        validate_feature_matrix(frame)


def test_as_frame_converts_pandas_string_and_nullable_columns_with_pyarrow():
    """pl.from_pandas (pyarrow present) handles dtypes the no-pyarrow fallback needs a workaround for."""
    frame = pd.DataFrame(
        {
            "a": pd.array([1, None], dtype="Int64"),
            "s": pd.array(["p", "q"], dtype="string"),
        }
    )
    result = as_frame(frame)
    assert isinstance(result, pl.DataFrame)
    assert result["a"].to_list() == [1, None]
    assert result["s"].to_list() == ["p", "q"]


def test_as_frame_converts_nullable_int64_without_pyarrow(monkeypatch):
    _force_no_pyarrow(monkeypatch)
    frame = pd.DataFrame({"a": pd.array([1, None], dtype="Int64")})
    result = as_frame(frame)
    assert isinstance(result, pl.DataFrame)
    assert result["a"].to_list() == [1, None]
    assert result["a"].null_count() == 1


def test_to_numpy_maps_null_and_nan_to_nan():
    frame = pl.DataFrame({"a": [1.0, float("nan"), None], "b": [1, 2, 3]})
    values = to_numpy(frame)
    assert values.dtype == np.float64
    assert np.isnan(values[1, 0])
    assert np.isnan(values[2, 0])


def test_canonical_dtype_matches_numpy_names():
    assert [canonical_dtype(d) for d in (pl.Float64, pl.Float32, pl.Int64, pl.UInt8, pl.Boolean)] == [
        "float64",
        "float32",
        "int64",
        "uint8",
        "bool",
    ]


def test_missing_mask_handles_object_and_float():
    assert missing_mask(np.array([1.0, np.nan])).tolist() == [False, True]
    assert missing_mask(np.array(["a", None, float("nan")], dtype=object)).tolist() == [False, True, True]
    assert missing_mask(np.array([1, 2])).tolist() == [False, False]
    assert missing_mask(np.array(["a", np.float32("nan")], dtype=object)).tolist() == [False, True]


def test_records_frame_serializes_nested_cells_and_handles_empty():
    assert records_frame([]).is_empty()
    frame = records_frame([{"a": 1, "p": {"x": 1}}, {"b": "z", "p": (1, 2)}])
    assert frame.columns == ["a", "p", "b"]
    assert frame["p"].to_list() == ['{"x":1}', "[1,2]"]


def test_read_table_uses_pandas_missing_tokens(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1.0,NA\n2.0,3\n", encoding="utf-8")
    frame = read_table(path, separator=",")
    assert frame["b"].null_count() == 1
    empty = tmp_path / "e.csv"
    empty.write_text("", encoding="utf-8")
    assert read_table(empty, separator=",").is_empty()


def test_read_table_casts_all_empty_column_to_float64(tmp_path):
    path = tmp_path / "empty_col.csv"
    path.write_text("a,b\n1.0,\n2.0,\n", encoding="utf-8")
    frame = read_table(path, separator=",")
    assert frame["b"].dtype == pl.Float64
    assert frame["b"].null_count() == frame.height
    assert frame["a"].dtype == pl.Float64
