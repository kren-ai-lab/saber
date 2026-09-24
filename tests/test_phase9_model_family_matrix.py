from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

import saber
from saber.datasets import DatasetBundle
from saber.preprocessing import PreprocessingConfig

CLASSIFIERS = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "svc",
    "linear_svc",
    "knn",
    "nearest_centroid",
    "decision_tree",
    "extra_tree",
    "adaboost",
    "bagging",
    "hist_gradient_boosting",
    "ridge_classifier",
    "sgd_classifier",
    "lda",
    "qda",
    "gaussian_process",
    "gaussian_nb",
    "bernoulli_nb",
    "multinomial_nb",
    "complement_nb",
)

REGRESSORS = (
    "linear_regression",
    "ridge_regressor",
    "lasso_regressor",
    "elastic_net",
    "bayesian_ridge",
    "random_forest_regressor",
    "extra_trees_regressor",
    "gradient_boosting_regressor",
    "hist_gradient_boosting_regressor",
    "adaboost_regressor",
    "knn_regressor",
    "svr",
    "linear_svr",
    "nu_svr",
    "decision_tree_regressor",
    "extra_tree_regressor",
    "gaussian_process_regressor",
    "bagging_regressor",
    "ard_regression",
    "huber_regression",
    "lars_regressor",
    "lasso_lars_regressor",
    "orthogonal_matching_pursuit",
)


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
    X, y = make_regression(
        n_samples=48,
        n_features=7,
        n_informative=5,
        noise=0.5,
        random_state=21,
    )
    return DatasetBundle(X=X, y=y, sample_ids=[f"r{i}" for i in range(48)])


@pytest.mark.parametrize("algorithm", CLASSIFIERS)
def test_public_api_classification_family_smoke_matrix(algorithm):
    dataset = _classification_dataset()
    params = {}
    if algorithm in {"random_forest", "extra_trees", "gradient_boosting", "adaboost"}:
        params["n_estimators"] = 12
    if algorithm == "bagging":
        params["n_estimators"] = 6
    if algorithm == "gaussian_process":
        params["max_iter_predict"] = 30
    if algorithm == "qda":
        params["reg_param"] = 0.1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = saber.train(
            dataset=dataset,
            algorithm=algorithm,
            random_state=42,
            model_params=params,
        )
        prediction = saber.predict(result, dataset=dataset)
        evaluation = saber.evaluate(dataset=dataset, model=result, metrics=("accuracy",))

    assert prediction.n_samples == dataset.n_samples
    assert 0.0 <= evaluation.metrics["accuracy"] <= 1.0


@pytest.mark.parametrize("algorithm", REGRESSORS)
def test_public_api_regression_family_smoke_matrix(algorithm):
    dataset = _regression_dataset()
    params = {}
    if algorithm in {
        "random_forest_regressor",
        "extra_trees_regressor",
        "gradient_boosting_regressor",
        "adaboost_regressor",
    }:
        params["n_estimators"] = 12
    if algorithm == "bagging_regressor":
        params["n_estimators"] = 6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = saber.train(
            dataset=dataset,
            algorithm=algorithm,
            random_state=42,
            model_params=params,
        )
        prediction = saber.predict(result, dataset=dataset)
        evaluation = saber.evaluate(dataset=dataset, model=result, metrics=("rmse", "mae"))

    assert prediction.n_samples == dataset.n_samples
    assert evaluation.metrics["rmse"] >= 0.0
    assert evaluation.metrics["mae"] >= 0.0


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


@pytest.mark.parametrize(
    "module_name, classifier, regressor, class_params, reg_params",
    [
        (
            "xgboost",
            "xgb_classifier",
            "xgb_regressor",
            {"n_estimators": 8, "max_depth": 3, "verbosity": 0},
            {"n_estimators": 8, "max_depth": 3, "verbosity": 0},
        ),
        (
            "lightgbm",
            "lgbm_classifier",
            "lgbm_regressor",
            {"n_estimators": 8, "verbose": -1},
            {"n_estimators": 8, "verbose": -1},
        ),
    ],
)
def test_optional_provider_public_api_end_to_end(
    module_name, classifier, regressor, class_params, reg_params
):
    pytest.importorskip(module_name)

    classification = _classification_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trained_classifier = saber.train(
            dataset=classification,
            algorithm=classifier,
            random_state=42,
            model_params=class_params,
        )
        classification_eval = saber.evaluate(
            dataset=classification,
            model=trained_classifier,
            metrics=("accuracy",),
        )
    assert np.isfinite(classification_eval.metrics["accuracy"])

    regression = _regression_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trained_regressor = saber.train(
            dataset=regression,
            algorithm=regressor,
            random_state=42,
            model_params=reg_params,
        )
        regression_eval = saber.evaluate(
            dataset=regression,
            model=trained_regressor,
            metrics=("rmse",),
        )
    assert np.isfinite(regression_eval.metrics["rmse"])
