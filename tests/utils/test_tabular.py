import importlib.util

import numpy as np
import pandas as pd
import polars as pl
import pytest

from saber.exceptions import DatasetValidationError
from saber.utils.tabular import as_frame, canonical_dtype, missing_mask, read_table, records_frame, to_numpy


def test_as_frame_converts_pandas_and_leaves_others():
    frame = as_frame(pd.DataFrame({"a": [1.0, np.nan]}))
    assert isinstance(frame, pl.DataFrame)
    assert frame["a"].null_count() == 1
    array = np.ones((2, 2))
    assert as_frame(array) is array


def test_as_frame_rejects_extension_dtypes_without_pyarrow():
    if importlib.util.find_spec("pyarrow") is not None:
        pytest.skip("pyarrow is installed, so extension dtypes convert")
    with pytest.raises(DatasetValidationError, match="pyarrow"):
        as_frame(pd.DataFrame({"a": pd.array([1, None], dtype="Int64")}))


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
