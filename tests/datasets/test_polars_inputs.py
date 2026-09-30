import numpy as np
import pandas as pd
import polars as pl
import pytest

from saber.datasets import DatasetBundle, FeatureSchema
from saber.exceptions import DatasetValidationError, FeatureSchemaMismatchError


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "f0": [0.5, 1.5, 2.5, 3.5],
            "f1": [1.0, None, 4.0, 5.0],
            "n": [1, 2, 3, 4],
            "flag": [True, False, True, False],
        }
    )


def test_polars_and_pandas_inputs_share_fingerprint():
    pandas_frame = pd.DataFrame(
        {
            "f0": [0.5, 1.5, 2.5, 3.5],
            "f1": [1.0, np.nan, 4.0, 5.0],
            "n": np.array([1, 2, 3, 4], dtype=np.int64),
            "flag": [True, False, True, False],
        }
    )
    y, ids = np.array([0, 1, 0, 1]), [10, 11, 12, 13]
    from_polars = DatasetBundle(X=_frame(), y=y, sample_ids=ids)
    from_pandas = DatasetBundle(X=pandas_frame, y=y, sample_ids=ids)
    assert isinstance(from_pandas.X, pl.DataFrame)
    assert from_polars.fingerprint == from_pandas.fingerprint
    assert from_polars.feature_schema.dtypes == ("float64", "float64", "int64", "bool")
    assert from_polars.feature_names == ("f0", "f1", "n", "flag")


def test_nan_and_null_are_both_missing_features():
    frame = pl.DataFrame({"a": [1.0, float("nan"), None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    bundle = DatasetBundle(X=frame, y=np.array([0, 1, 0, 1]))
    with pytest.raises(DatasetValidationError):
        bundle.validate(require_finite_features=True)


def test_infinite_features_are_rejected():
    with pytest.raises(DatasetValidationError, match="infinite"):
        DatasetBundle(X=pl.DataFrame({"a": [1.0, float("inf")]}), y=np.array([0, 1]))


def test_non_numeric_columns_are_rejected():
    with pytest.raises(DatasetValidationError, match="Non-numerical columns: s"):
        DatasetBundle(X=pl.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]}), y=np.array([0, 1]))


def test_null_target_is_rejected():
    with pytest.raises(DatasetValidationError):
        DatasetBundle(X=np.ones((2, 1)), y=np.array(["a", None], dtype=object))


def test_feature_schema_rejects_reordered_polars_columns():
    schema = FeatureSchema.from_data(_frame())
    reordered = _frame().select(["f1", "f0", "n", "flag"])
    with pytest.raises(FeatureSchemaMismatchError):
        schema.validate_compatible(reordered)


def test_feature_schema_checks_only_width_for_unnamed_numpy_input():
    schema = FeatureSchema.from_data(_frame())
    schema.validate_compatible(np.ones((2, 4)))
    with pytest.raises(FeatureSchemaMismatchError, match="4 from the training schema"):
        schema.validate_compatible(np.ones((2, 3)))
