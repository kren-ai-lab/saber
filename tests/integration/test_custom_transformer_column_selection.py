from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

import saber
from saber.core.search_space import SearchSpace
from saber.datasets import DatasetBundle, PartitionPlan
from saber.persistence import load_model, save_model
from saber.preprocessing import PreprocessingConfig
from saber.tuning import TuningConfig

FEATURE_NAMES = ["a", "b", "c"]


def _raw_frame():
    X, y = make_classification(
        n_samples=60,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    return pd.DataFrame(X, columns=FEATURE_NAMES), y


def _dataset() -> DatasetBundle:
    frame, y = _raw_frame()
    return DatasetBundle(X=pl.from_pandas(frame), y=y, sample_ids=[f"s{i}" for i in range(len(y))])


def _preprocessing() -> PreprocessingConfig:
    # Selects a single column by name; only works if the pipeline receives a
    # DataFrame, not a bare NumPy array (sklearn's ColumnTransformer contract).
    transformer = ColumnTransformer([("scale", StandardScaler(), ["a"])], remainder="passthrough")
    return PreprocessingConfig(transformer=transformer)


def _fold_plan(dataset: DatasetBundle) -> PartitionPlan:
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=[i % 3 for i in range(dataset.n_samples)],
        dataset_fingerprint=dataset.fingerprint,
    )


def test_train_validate_tune_predict_and_artifact_round_trip_with_named_column_selection(tmp_path):
    # pandas input is converted to Polars inside DatasetBundle, so one frame type covers both.
    dataset = _dataset()
    preprocessing = _preprocessing()

    train_result = saber.train(
        dataset=dataset,
        algorithm="logistic_regression",
        preprocessing=preprocessing,
        random_state=7,
    )
    prediction = saber.predict(train_result, dataset=dataset)
    assert prediction.predictions.shape[0] == dataset.n_samples

    validation_result = saber.validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=_fold_plan(dataset),
        preprocessing=preprocessing,
        metrics=("accuracy",),
        random_state=7,
    )
    assert "accuracy" in validation_result.aggregate_metrics

    tune_result = saber.tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(
            optimizer="grid",
            metrics=("accuracy",),
            refit_metric="accuracy",
            random_state=7,
            n_jobs=1,
        ),
        partition_plan=_fold_plan(dataset),
        preprocessing=preprocessing,
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
    )
    assert tune_result.best_model is not None

    artifact_path = tmp_path / "model"
    save_model(
        artifact_path,
        model=train_result.model,
        algorithm="logistic_regression",
        task="classification",
        dataset=dataset,
    )
    loaded = load_model(artifact_path)
    loaded_prediction = loaded.predict_result(dataset.X, sample_ids=dataset.sample_ids)
    np.testing.assert_array_equal(loaded_prediction.predictions, prediction.predictions)
    assert loaded_prediction.probabilities is not None
    assert prediction.probabilities is not None
    np.testing.assert_allclose(loaded_prediction.probabilities, prediction.probabilities)
