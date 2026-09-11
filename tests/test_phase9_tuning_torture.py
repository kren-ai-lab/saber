from __future__ import annotations

import numpy as np
import pytest
import warnings
from sklearn.datasets import make_classification, make_regression

from mlcore import MODEL_REGISTRY
from mlcore.core.search_space import Categorical, Float, Integer, LogFloat, SearchSpace
from mlcore.datasets import DatasetBundle, PartitionPlan
from mlcore.exceptions import MetricProblemTypeError, MetricTaskMismatchError, OptimizationError, ValidationContractError
from mlcore.preprocessing import PreprocessingConfig
from mlcore.tuning import TuningConfig, TuningEngine


def _cv(dataset: DatasetBundle, n=3):
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=np.arange(dataset.n_samples) % n,
        dataset_fingerprint=dataset.fingerprint,
    )


def _classification(n=60, seed=7):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5, random_state=seed)
    return DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(n)])


@pytest.mark.parametrize("optimizer", ["grid", "random", "halving_grid", "halving_random"])
def test_all_sklearn_tuning_backends_respect_explicit_cv(optimizer):
    dataset = _classification(54)
    space = SearchSpace("lr", {"C": Categorical([0.2, 1.0])})
    config = TuningConfig(
        optimizer=optimizer,
        metrics=("accuracy",),
        refit_metric="accuracy",
        random_state=4,
        n_jobs=1,
        n_iter=2,
        factor=2,
        min_resources="smallest" if optimizer == "halving_random" else "exhaust",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TuningEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            config=config,
            partition_plan=_cv(dataset),
            search_space=space,
        )
    assert result.best_model is not None
    assert np.isfinite(result.best_score)
    assert result.metadata["n_splits"] == 3


def test_optuna_tuning_runs_typed_log_space_and_is_seed_reproducible():
    pytest.importorskip("optuna")
    dataset = _classification(54)
    space = SearchSpace("lr", {"C": LogFloat(1e-2, 10.0)})
    config = TuningConfig(
        optimizer="optuna",
        metrics=("accuracy",),
        refit_metric="accuracy",
        random_state=17,
        n_trials=4,
        n_jobs=1,
    )
    engine = TuningEngine(MODEL_REGISTRY)
    a = engine.run(dataset=dataset, algorithm="logistic_regression", config=config, partition_plan=_cv(dataset), search_space=space)
    b = engine.run(dataset=dataset, algorithm="logistic_regression", config=config, partition_plan=_cv(dataset), search_space=space)
    assert a.best_params == b.best_params
    assert a.best_score == pytest.approx(b.best_score)


def test_tuning_preprocessing_is_fitted_only_on_search_development_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [1000.0], [2000.0]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    ids = [f"s{i}" for i in range(8)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.holdout(
        train_ids=ids[:4], validation_ids=ids[4:6], test_ids=ids[6:], dataset_fingerprint=dataset.fingerprint
    )
    result = TuningEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy", n_jobs=1),
        partition_plan=plan,
        search_space=SearchSpace("lr", {"C": [1.0]}),
        preprocessing=PreprocessingConfig(imputation=None, scaler="standard"),
    )
    scaler = result.best_model.named_steps["scaler"]
    assert scaler.mean_[0] == pytest.approx(np.mean(X[:6]))
    assert scaler.mean_[0] != pytest.approx(np.mean(X))
    assert result.metadata["protected_samples"] == 2


def test_tuning_rejects_regression_metric_for_classifier():
    dataset = _classification()
    with pytest.raises(MetricTaskMismatchError):
        TuningEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", metrics=("rmse",), refit_metric="rmse"),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("lr", {"C": [1.0]}),
        )


def test_tuning_rejects_binary_roc_auc_for_multiclass_problem():
    X, y = make_classification(
        n_samples=75, n_features=8, n_informative=6, n_classes=3, n_clusters_per_class=1, random_state=3
    )
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(75)])
    with pytest.raises(MetricProblemTypeError):
        TuningEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", metrics=("roc_auc",), refit_metric="roc_auc"),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("lr", {"C": [1.0]}),
        )


def test_grid_rejects_unbounded_continuous_domain_that_cannot_be_enumerated():
    dataset = _classification()
    with pytest.raises(ValidationContractError):
        TuningEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy"),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("lr", {"C": Float(0.01, 2.0)}),
        )


def test_invalid_candidates_can_fail_without_losing_valid_grid_candidates():
    dataset = _classification()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TuningEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(
                optimizer="grid", metrics=("accuracy",), refit_metric="accuracy", n_jobs=1, error_score=np.nan
            ),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("lr", {"C": Categorical([-1.0, 0.2, 1.0])}),
        )
    assert result.best_params["C"] in {0.2, 1.0}
    assert result.failures


def test_all_invalid_grid_candidates_raise_nonfinite_score_error():
    dataset = _classification()
    with pytest.raises(OptimizationError, match="All candidate fits failed"):
        TuningEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(
                optimizer="grid", metrics=("accuracy",), refit_metric="accuracy", n_jobs=1, error_score=np.nan
            ),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("lr", {"C": Categorical([-1.0, -2.0])}),
        )


def test_refit_false_keeps_best_params_but_no_best_model():
    dataset = _classification()
    result = TuningEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy", refit=False),
        partition_plan=_cv(dataset),
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
    )
    assert result.best_model is None
    assert result.best_params["C"] in {0.1, 1.0}


def test_regression_loss_metric_display_score_is_natural_positive_value():
    X, y = make_regression(n_samples=60, n_features=6, noise=0.5, random_state=6)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"r{i}" for i in range(60)])
    result = TuningEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="ridge_regressor",
        config=TuningConfig(optimizer="grid", metrics=("rmse", "mae"), refit_metric="rmse", n_jobs=1),
        partition_plan=_cv(dataset),
        search_space=SearchSpace("ridge", {"alpha": [0.1, 1.0]}),
    )
    assert result.best_score <= 0.0
    assert result.display_score >= 0.0
    assert result.display_scores["rmse"] >= 0.0
    assert result.display_scores["mae"] >= 0.0


def test_sample_weights_work_through_tuning_pipeline():
    X, y = make_classification(n_samples=60, n_features=6, random_state=8)
    weights = np.linspace(0.25, 2.0, 60)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(60)], sample_weight=weights)
    result = TuningEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy", n_jobs=1),
        partition_plan=_cv(dataset),
        search_space=SearchSpace("lr", {"C": [0.5, 1.0]}),
    )
    assert result.best_model is not None


def test_sample_weights_fail_for_knn_during_tuning():
    X, y = make_classification(n_samples=60, n_features=6, random_state=8)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(60)], sample_weight=np.ones(60))
    with pytest.raises(ValidationContractError, match="sample-weight"):
        TuningEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="knn_classifier",
            config=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy"),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("knn", {"n_neighbors": [3, 5]}),
        )


def test_tuning_rejects_any_explicit_training_fold_with_single_class_before_search():
    X = np.arange(72, dtype=float).reshape(24, 3)
    y = np.array([0] * 12 + [1] * 12)
    ids = [f"s{i}" for i in range(24)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.holdout(
        train_ids=ids[:12],
        validation_ids=ids[12:18],
        test_ids=ids[18:],
        dataset_fingerprint=dataset.fingerprint,
    )
    with pytest.raises(Exception, match="at least two|binary|multiclass"):
        TuningEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy"),
            partition_plan=plan,
            search_space=SearchSpace("lr", {"C": [1.0]}),
        )
