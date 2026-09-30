from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

import saber
from saber.datasets import DatasetBundle
from saber.preprocessing import PreprocessingConfig

# These need purpose-built data and have their own tests below.
SPECIAL_DATA = {"categorical_nb", "gamma_regression"}
SPECS = [spec for spec in saber.ALGORITHMS.values() if spec.name not in SPECIAL_DATA]
# Keep the matrix fast; everything else runs with registry defaults.
FAST_PARAMS = {
    "gaussian_process": {"max_iter_predict": 30},
    "qda": {"reg_param": 0.1},
    "xgb_classifier": {"verbosity": 0},
    "xgb_regressor": {"verbosity": 0},
    "xgb_rf_classifier": {"verbosity": 0},
    "xgbrf_regressor": {"verbosity": 0},
    "lgbm_classifier": {"verbose": -1},
    "lgbm_regressor": {"verbose": -1},
}


def _classification_dataset():
    X, y = make_classification(
        n_samples=48,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        class_sep=1.3,
        random_state=19,
    )
    return DatasetBundle(X=X, y=y, sample_ids=[f"c{i}" for i in range(48)])


def _regression_dataset():
    X, y = make_regression(  # pyrefly: ignore[bad-unpacking]
        n_samples=48,
        n_features=7,
        n_informative=5,
        noise=0.5,
        random_state=21,
    )
    return DatasetBundle(X=X, y=y, sample_ids=[f"r{i}" for i in range(48)])


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
def test_every_registered_algorithm_trains_predicts_and_evaluates(spec):
    if spec.task == "classification":
        dataset, metric = _classification_dataset(), "accuracy"
    else:
        dataset, metric = _regression_dataset(), "rmse"
    params = dict(FAST_PARAMS.get(spec.name, {}))
    if "n_estimators" in spec.build_estimator().get_params():
        params["n_estimators"] = 8

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = saber.train(dataset=dataset, algorithm=spec.name, random_state=42, model_params=params)
        prediction = saber.predict(result, dataset=dataset)
        evaluation = saber.evaluate(dataset=dataset, model=result, metrics=(metric,))

    assert prediction.n_samples == dataset.n_samples
    assert np.isfinite(evaluation.metrics[metric])


def test_categorical_nb_runs_on_prepared_nonnegative_integer_features_without_scaling():
    rng = np.random.default_rng(11)
    X = rng.integers(0, 4, size=(60, 6))
    y = np.array([0, 1] * 30)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(60)])
    result = saber.train(
        dataset=dataset,
        algorithm="categorical_nb",
        preprocessing=PreprocessingConfig(imputation=None, scaler=None),
    )
    evaluation = saber.evaluate(dataset=dataset, model=result, metrics=("accuracy",))
    assert 0.0 <= evaluation.metrics["accuracy"] <= 1.0


def test_gamma_regression_runs_on_strictly_positive_target():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 5))
    y = np.exp(0.2 * X[:, 0] - 0.1 * X[:, 1]) + 0.1
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"g{i}" for i in range(60)])
    result = saber.train(dataset=dataset, algorithm="gamma_regression")
    evaluation = saber.evaluate(dataset=dataset, model=result, metrics=("rmse",))
    assert np.isfinite(evaluation.metrics["rmse"])
