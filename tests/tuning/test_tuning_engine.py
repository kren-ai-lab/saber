from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline

from saber.core.search_space import LogFloat, SearchSpace
from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import ValidationContractError
from saber.tuning import TuningConfig, tune
from saber.validation import validate


def _classification_dataset(n: int = 60, *, sample_weight: bool = False):
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    ids = [f"sample_{i}" for i in range(n)]
    weights = np.linspace(0.05, 5.0, n) if sample_weight else None
    return DatasetBundle(X=X, y=y, sample_ids=ids, sample_weight=weights)


def _three_fold_plan(dataset: DatasetBundle) -> PartitionPlan:
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=[i % 3 for i in range(dataset.n_samples)],
        dataset=dataset,
    )


def test_grid_tuning_uses_pipeline_explicit_folds_and_multiple_metrics() -> None:
    dataset = _classification_dataset()
    result = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(
            optimizer="grid",
            refit_metric="balanced_accuracy",
            n_jobs=1,
        ),
        partition_plan=_three_fold_plan(dataset),
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
        metrics=("accuracy", "balanced_accuracy"),
        random_state=42,
    )

    assert isinstance(result.best_model, Pipeline)
    assert result.refit_metric == "balanced_accuracy"
    assert set(result.best_scores) == {"accuracy", "balanced_accuracy"}
    assert "C" in result.best_params
    assert "estimator__C" not in result.best_params
    assert result.metadata["n_splits"] == 3
    assert result.metadata["partition_fingerprint"] == result.partition_plan.fingerprint


def test_holdout_validation_protects_final_test_from_search_and_refit() -> None:
    X = np.arange(40, dtype=float).reshape(-1, 1)
    X[30:] = 10_000 + np.arange(10, dtype=float).reshape(-1, 1)
    y = np.asarray([0, 1] * 20)
    ids = [f"sample_{i}" for i in range(40)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.holdout(
        train_ids=ids[:20],
        validation_ids=ids[20:30],
        test_ids=ids[30:],
        dataset=dataset,
    )

    result = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(optimizer="grid", n_jobs=1),
        partition_plan=plan,
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
        metrics=("accuracy",),
    )

    assert result.metadata["n_search_samples"] == 30
    assert result.metadata["protected_samples"] == 10
    assert result.metadata["evaluation_roles"] == ("validation",)
    assert result.best_model.named_steps["scaler"].mean_[0] == pytest.approx(14.5)


def test_unpartitioned_tuning_never_falls_back_to_internal_splitter() -> None:
    dataset = _classification_dataset()
    with pytest.raises(ValidationContractError, match="BioSievePartitionConfig"):
        tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid"),
            search_space=SearchSpace("lr", {"C": [1.0]}),
            metrics=("accuracy",),
        )


def test_random_tuning_consumes_continuous_log_space_reproducibly() -> None:
    dataset = _classification_dataset()
    plan = _three_fold_plan(dataset)
    space = SearchSpace("lr", {"C": LogFloat(1e-3, 10.0)})
    config = TuningConfig(
        optimizer="random",
        n_jobs=1,
        n_iter=4,
    )
    first = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=config,
        partition_plan=plan,
        search_space=space,
        metrics=("accuracy",),
        random_state=123,
    )
    second = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=config,
        partition_plan=plan,
        search_space=space,
        metrics=("accuracy",),
        random_state=123,
    )
    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)


def test_halving_grid_reports_requested_secondary_metrics() -> None:
    dataset = _classification_dataset()
    result = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(
            optimizer="halving_grid",
            refit_metric="accuracy",
            n_jobs=1,
        ),
        partition_plan=_three_fold_plan(dataset),
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
        metrics=("accuracy", "balanced_accuracy"),
        random_state=42,
    )
    assert set(result.best_scores) == {"accuracy", "balanced_accuracy"}
    assert result.best_model is not None


def test_refit_false_keeps_best_configuration_without_fitted_model() -> None:
    dataset = _classification_dataset()
    result = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(
            optimizer="grid",
            refit=False,
            n_jobs=1,
        ),
        partition_plan=_three_fold_plan(dataset),
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
        metrics=("accuracy",),
    )
    assert result.best_model is None
    assert result.best_params
    assert np.isfinite(result.best_score)


def test_sample_weights_are_propagated_for_supported_estimator() -> None:
    dataset = _classification_dataset(sample_weight=True)
    result = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(optimizer="grid", n_jobs=1),
        partition_plan=_three_fold_plan(dataset),
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
        metrics=("accuracy",),
    )
    X, y, weights = np.asarray(dataset.X), dataset.y, dataset.sample_weight
    weighted = clone(result.best_model).fit(X, y, estimator__sample_weight=weights)
    unweighted = clone(result.best_model).fit(X, y)
    coef = result.best_model.named_steps["estimator"].coef_
    np.testing.assert_allclose(coef, weighted.named_steps["estimator"].coef_)
    assert not np.allclose(coef, unweighted.named_steps["estimator"].coef_)


def test_sample_weights_are_rejected_when_estimator_does_not_support_them() -> None:
    dataset = _classification_dataset(sample_weight=True)
    with pytest.raises(ValidationContractError, match="sample-weight support"):
        tune(
            dataset=dataset,
            algorithm="knn_classifier",
            config=TuningConfig(optimizer="grid", n_jobs=1),
            partition_plan=_three_fold_plan(dataset),
            search_space=SearchSpace("knn_classifier", {"n_neighbors": [3, 5]}),
            metrics=("accuracy",),
        )


def test_failed_candidate_is_explicit_in_history() -> None:
    dataset = _classification_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", n_jobs=1),
            partition_plan=_three_fold_plan(dataset),
            search_space=SearchSpace("lr", {"C": [-1.0, 1.0]}),
            metrics=("accuracy",),
        )
    assert result.failures
    assert result.failures[0]["status"] == "failed"
    assert np.isfinite(result.best_score)
    assert result.best_params["C"] == 1.0


def test_regression_loss_scores_keep_natural_display_direction() -> None:
    X, y = make_regression(n_samples=60, n_features=5, random_state=42)  # pyrefly: ignore[bad-unpacking]
    ids = [f"sample_{i}" for i in range(60)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    result = tune(
        dataset=dataset,
        algorithm="ridge_regressor",
        config=TuningConfig(optimizer="grid", refit_metric="rmse", n_jobs=1),
        partition_plan=_three_fold_plan(dataset),
        search_space=SearchSpace("ridge", {"alpha": [0.1, 1.0]}),
        metrics=("rmse", "r2"),
    )
    assert result.best_score < 0
    assert result.display_score > 0
    assert result.display_scores["rmse"] > 0


def test_optimization_history_exports_as_flat_dataframe() -> None:
    dataset = _classification_dataset()
    result = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(
            optimizer="grid",
            refit_metric="accuracy",
            n_jobs=1,
        ),
        partition_plan=_three_fold_plan(dataset),
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
        metrics=("accuracy", "balanced_accuracy"),
    )
    frame = result.history_frame()
    assert "param__C" in frame.columns
    assert "metric__accuracy__mean" in frame.columns
    assert len(frame) == len(result.history)


def test_protected_test_rows_do_not_influence_search_or_selection() -> None:
    X, y = make_classification(n_samples=60, n_features=6, n_informative=4, random_state=1)
    ids = [f"sample_{i}" for i in range(60)]

    def run_tune(features):
        dataset = DatasetBundle(X=features, y=y, sample_ids=ids)
        plan = PartitionPlan.holdout(
            train_ids=ids[:30],
            validation_ids=ids[30:45],
            test_ids=ids[45:],
            dataset=dataset,
        )
        return tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", n_jobs=1),
            partition_plan=plan,
            search_space=SearchSpace("lr", {"C": [0.01, 0.1, 1.0, 10.0]}),
            metrics=("accuracy",),
        )

    shifted_test = X.copy()
    shifted_test[45:] = 1e6
    base, shifted = run_tune(X), run_tune(shifted_test)

    assert shifted.best_params == base.best_params
    assert shifted.history_frame().equals(base.history_frame())
    np.testing.assert_allclose(
        shifted.best_model.named_steps["estimator"].coef_,
        base.best_model.named_steps["estimator"].coef_,
    )


@pytest.mark.parametrize("positive_class", [None, "active"], ids=["default-positive", "explicit-positive"])
def test_binary_tuning_scores_match_evaluation_of_the_positive_class(positive_class) -> None:
    # Imbalanced string labels: a class-weighted average would diverge from the
    # positive-class scores that evaluation reports.
    X, y_int = make_classification(
        n_samples=90, n_features=6, n_informative=4, weights=[0.8], flip_y=0.05, random_state=3
    )
    y = np.where(y_int == 1, "active", "inactive")
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"sample_{i}" for i in range(90)])
    plan = _three_fold_plan(dataset)
    metrics = ("precision", "recall", "f1", "roc_auc")

    tuned = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(optimizer="grid", refit_metric="f1", n_jobs=1),
        partition_plan=plan,
        search_space=SearchSpace("lr", {"C": [1.0]}),
        positive_class=positive_class,
        metrics=metrics,
    )
    validated = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=metrics,
        positive_class=positive_class,
    )

    assert tuned.best_scores == pytest.approx(validated.aggregate_metrics)


def test_tuning_rejects_positive_class_outside_the_binary_target() -> None:
    dataset = _classification_dataset()
    with pytest.raises(ValidationContractError, match="positive_class"):
        tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=TuningConfig(optimizer="grid", n_jobs=1),
            partition_plan=_three_fold_plan(dataset),
            search_space=SearchSpace("lr", {"C": [1.0]}),
            positive_class="missing",
            metrics=("f1",),
        )
