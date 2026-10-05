from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from saber.core import get_algorithm
from saber.datasets import DatasetBundle
from saber.preprocessing.pipeline import PreprocessingConfig, build_model_pipeline, pipeline_input


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
    spec = get_algorithm("logistic_regression")
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
    spec = get_algorithm("logistic_regression")
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


class _InputTypeSpy(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.fit_input_type_ = type(X)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        self.predict_input_type_ = type(X)
        return np.zeros(len(X), dtype=int)


def test_estimator_receives_numpy_when_custom_transformer_outputs_a_frame():
    frame = _named_frame()
    dataset = DatasetBundle(X=frame, y=np.array([0, 1, 0, 1, 0, 1]), sample_ids=[f"s{i}" for i in range(6)])
    spec = get_algorithm("logistic_regression")
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=_InputTypeSpy(),
        training_data=dataset,
        preprocessing=PreprocessingConfig(transformer=StandardScaler().set_output(transform="polars")),
    )
    X = pipeline_input(pipeline, frame)
    pipeline.fit(X, dataset.y)
    pipeline.predict(X)
    estimator = pipeline.named_steps["estimator"]
    assert estimator.fit_input_type_ is np.ndarray
    assert estimator.predict_input_type_ is np.ndarray
