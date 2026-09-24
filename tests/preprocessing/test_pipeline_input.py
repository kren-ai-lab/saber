from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from saber import MODEL_REGISTRY
from saber.datasets import DatasetBundle
from saber.preprocessing import PreprocessingConfig, build_model_pipeline, pipeline_input


def _dataset(X):
    return DatasetBundle(
        X=np.asarray(X, dtype=float),
        y=np.array([0, 1, 0, 1, 0, 1], dtype=int),
        sample_ids=[f"s{i}" for i in range(6)],
    )


def _named_frame():
    return pl.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.fixture
def default_pipeline():
    dataset = _dataset([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    spec = MODEL_REGISTRY.get("logistic_regression")
    return build_model_pipeline(
        spec=spec,
        estimator=spec.build_estimator(random_state=7),
        training_data=dataset,
        preprocessing=PreprocessingConfig(),
    )


@pytest.fixture
def custom_transformer_pipeline():
    frame = _named_frame()
    dataset = DatasetBundle(
        X=frame, y=np.array([0, 1, 0, 1, 0, 1], dtype=int), sample_ids=[f"s{i}" for i in range(6)]
    )
    spec = MODEL_REGISTRY.get("logistic_regression")
    transformer = ColumnTransformer([("scale", StandardScaler(), ["a"])], remainder="passthrough")
    return build_model_pipeline(
        spec=spec,
        estimator=spec.build_estimator(random_state=7),
        training_data=dataset,
        preprocessing=PreprocessingConfig(transformer=transformer),
    )


def test_default_pipeline_keeps_numpy_boundary(default_pipeline):
    frame = _named_frame()
    result = pipeline_input(default_pipeline, frame)
    assert isinstance(result, np.ndarray)


def test_custom_preprocess_pipeline_receives_a_frame_for_named_column_selection(custom_transformer_pipeline):
    frame = _named_frame()
    result = pipeline_input(custom_transformer_pipeline, frame)
    assert isinstance(result, pl.DataFrame)


def test_custom_preprocess_pipeline_keeps_numpy_x_as_numpy(custom_transformer_pipeline):
    array = np.asarray(_named_frame())
    result = pipeline_input(custom_transformer_pipeline, array)
    assert isinstance(result, np.ndarray)
