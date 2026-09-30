from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn import model_selection
from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  # exposes the halving classes

import saber.tuning.engine as tuning_engine_module
from saber.core.search_space import Categorical, Float, SearchSpace
from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import (
    MetricProblemTypeError,
    MetricTaskMismatchError,
    OptimizationError,
    ValidationContractError,
)
from saber.tuning import TuningConfig, tune


def _cv(dataset: DatasetBundle, n=3):
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=np.arange(dataset.n_samples) % n,
        dataset_fingerprint=dataset.fingerprint,
    )


def _classification(n=60, seed=7):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5, random_state=seed)
    return DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(n)])


SEARCH_CLASSES = {
    "grid": (tuning_engine_module, "GridSearchCV"),
    "random": (tuning_engine_module, "RandomizedSearchCV"),
    "halving_grid": (model_selection, "HalvingGridSearchCV"),
    "halving_random": (model_selection, "HalvingRandomSearchCV"),
}


@pytest.mark.parametrize("optimizer", ["grid", "random", "halving_grid", "halving_random"])
def test_all_sklearn_tuning_backends_respect_explicit_cv(optimizer, monkeypatch):
    dataset = _classification(54)
    plan = _cv(dataset)
    module, name = SEARCH_CLASSES[optimizer]
    search_class = getattr(module, name)
    received_cv = []

    def recording_search(*args, **kwargs):
        received_cv.append(kwargs["cv"])
        return search_class(*args, **kwargs)

    monkeypatch.setattr(module, name, recording_search)
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
        result = tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=config,
            partition_plan=plan,
            search_space=space,
        )
    assert result.best_model is not None
    assert np.isfinite(result.best_score)
    ids = np.asarray(dataset.sample_ids)
    (cv,) = received_cv
    assert [(set(ids[train]), set(ids[test])) for train, test in cv] == [
        (set(split.train_ids), set(split.validation_ids)) for split in plan.splits
    ]


def test_tuning_rejects_regression_metric_for_classifier():
    dataset = _classification()
    with pytest.raises(MetricTaskMismatchError):
        tune(
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
        tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", metrics=("roc_auc",), refit_metric="roc_auc"),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("lr", {"C": [1.0]}),
        )


def test_grid_rejects_unbounded_continuous_domain_that_cannot_be_enumerated():
    dataset = _classification()
    with pytest.raises(ValidationContractError):
        tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy"),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("lr", {"C": Float(0.01, 2.0)}),
        )


def test_all_invalid_grid_candidates_raise_nonfinite_score_error():
    dataset = _classification()
    with pytest.raises(OptimizationError, match="All candidate fits failed"):
        tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(
                optimizer="grid", metrics=("accuracy",), refit_metric="accuracy", n_jobs=1, error_score=np.nan
            ),
            partition_plan=_cv(dataset),
            search_space=SearchSpace("lr", {"C": Categorical([-1.0, -2.0])}),
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
    with pytest.raises(ValidationContractError, match="at least two target classes"):
        tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy"),
            partition_plan=plan,
            search_space=SearchSpace("lr", {"C": [1.0]}),
        )
