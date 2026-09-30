from __future__ import annotations

import numpy as np
import pytest

from saber import MODEL_REGISTRY
from saber.datasets import DatasetBundle
from saber.exceptions import PreprocessingContractError
from saber.preprocessing import PreprocessingConfig, build_model_pipeline


def _dataset(X):
    return DatasetBundle(
        X=np.asarray(X, dtype=float),
        y=np.array([0, 1, 0, 1, 0, 1], dtype=int),
        sample_ids=[f"s{i}" for i in range(6)],
    )


def test_auto_preprocessing_uses_standard_scaling_when_recommended():
    dataset = _dataset([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    spec = MODEL_REGISTRY.get("logistic_regression")
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=spec.build_estimator(random_state=7),
        training_data=dataset,
        preprocessing=PreprocessingConfig(),
    )
    assert pipeline.named_steps["scaler"].__class__.__name__ == "StandardScaler"
    assert pipeline.named_steps["imputer"].__class__.__name__ == "SimpleImputer"


def test_auto_preprocessing_uses_minmax_for_non_negative_estimators():
    dataset = _dataset([[-5, 1], [-4, 2], [-3, 3], [-2, 4], [-1, 5], [0, 6]])
    spec = MODEL_REGISTRY.get("multinomial_nb")
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=spec.build_estimator(),
        training_data=dataset,
        preprocessing=PreprocessingConfig(),
    )
    assert pipeline.named_steps["scaler"].__class__.__name__ == "MinMaxScaler"


def test_missing_values_require_imputation_for_non_native_estimators():
    dataset = _dataset([[np.nan, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    spec = MODEL_REGISTRY.get("logistic_regression")
    with pytest.raises(PreprocessingContractError, match="missing"):
        build_model_pipeline(
            spec=spec,
            estimator=spec.build_estimator(),
            training_data=dataset,
            preprocessing=PreprocessingConfig(imputation=None, scaler=None),
        )


def test_minmax_scaling_clips_held_out_values_for_non_negative_estimators():
    train = _dataset([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    spec = MODEL_REGISTRY.get("multinomial_nb")
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=spec.build_estimator(),
        training_data=train,
        preprocessing=PreprocessingConfig(imputation=None, scaler="auto"),
    )
    pipeline.fit(train.X, train.y)

    transformed = pipeline.named_steps["scaler"].transform(np.array([[-100.0, -100.0], [100.0, 100.0]]))
    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 1.0)
