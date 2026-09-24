from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from saber import MODEL_REGISTRY
from saber.benchmark import BenchmarkConfig, BenchmarkEngine
from saber.core.search_space import SearchSpace
from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import BenchmarkContractError
from saber.tuning import TuningConfig


def _classification(n=60, seed=5):
    X, y = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=seed)
    ids = [f"s{i}" for i in range(n)]
    return DatasetBundle(X=X, y=y, sample_ids=ids)


def _cv(dataset, n=3):
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=np.arange(dataset.n_samples) % n,
        dataset_fingerprint=dataset.fingerprint,
    )


def test_multi_representation_benchmark_reuses_identical_membership():
    base = _classification()
    second = DatasetBundle(
        X=np.column_stack([np.asarray(base.X), np.asarray(base.X)[:, :2] ** 2]),
        y=base.y.copy(),
        sample_ids=base.sample_ids,
    )
    plan = _cv(base)
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets={"rep_a": base, "rep_b": second},
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(metrics=("accuracy",), seeds=(1, 2), include_baselines=True),
        partitions={"cv": plan},
    )
    assert result.n_runs == 8  # 2 reps × (model + baseline) × 2 seeds
    assert len(result.failures) == 0
    memberships = {
        (run.dataset_label, tuple(tuple(f.evaluation_ids) for f in run.validation.folds))
        for run in result.successes
    }
    by_dataset = {}
    for dataset_label, folds in memberships:
        by_dataset.setdefault(dataset_label, folds)
        assert by_dataset[dataset_label] == folds
    assert by_dataset["rep_a"] == by_dataset["rep_b"]


def test_representation_with_different_target_for_same_id_fails_before_runs():
    base = _classification()
    y2 = base.y.copy()
    y2[0] = 1 - y2[0]
    second = DatasetBundle(X=np.asarray(base.X).copy(), y=y2, sample_ids=base.sample_ids)
    with pytest.raises(BenchmarkContractError, match="target"):
        BenchmarkEngine(MODEL_REGISTRY).run(
            datasets={"a": base, "b": second},
            algorithms=("logistic_regression",),
            config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
            partitions=_cv(base),
        )


def test_representation_with_missing_sample_id_fails_before_runs():
    base = _classification()
    second = DatasetBundle(X=np.asarray(base.X)[:-1], y=base.y[:-1], sample_ids=base.sample_ids[:-1])
    with pytest.raises(BenchmarkContractError, match="same sample IDs"):
        BenchmarkEngine(MODEL_REGISTRY).run(
            datasets={"a": base, "b": second},
            algorithms=("logistic_regression",),
            config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
            partitions=_cv(base),
        )


def test_mixed_classification_and_regression_algorithms_are_rejected_globally():
    dataset = _classification()
    with pytest.raises(BenchmarkContractError, match="mix supervised tasks"):
        BenchmarkEngine(MODEL_REGISTRY).run(
            datasets=dataset,
            algorithms=("logistic_regression", "ridge_regressor"),
            config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
            partitions=_cv(dataset),
        )


def test_unknown_algorithm_is_isolated_as_failed_run():
    dataset = _classification()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression", "does_not_exist"),
        config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False, fail_fast=False),
        partitions=_cv(dataset),
    )
    assert len(result.successes) == 1
    assert len(result.failures) == 1
    assert "AlgorithmNotFoundError" in result.failures[0].error


def test_fail_fast_promotes_run_failure_to_benchmark_contract_error():
    dataset = _classification()
    with pytest.raises(BenchmarkContractError, match="AlgorithmNotFoundError"):
        BenchmarkEngine(MODEL_REGISTRY).run(
            datasets=dataset,
            algorithms=("logistic_regression", "does_not_exist"),
            config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False, fail_fast=True),
            partitions=_cv(dataset),
        )


def test_same_seed_benchmark_is_reproducible_for_random_forest():
    dataset = _classification(72)
    kwargs = dict(
        datasets=dataset,
        algorithms=("random_forest",),
        config=BenchmarkConfig(metrics=("accuracy", "mcc"), seeds=(42,), include_baselines=False),
        partitions=_cv(dataset),
        model_params={"random_forest": {"n_estimators": 20}},
    )
    engine = BenchmarkEngine(MODEL_REGISTRY)
    a = engine.run(**kwargs)
    b = engine.run(**kwargs)
    assert a.aggregate_metrics_frame()["score"].tolist() == pytest.approx(
        b.aggregate_metrics_frame()["score"].tolist()
    )
    assert a.predictions_frame()["y_pred"].tolist() == b.predictions_frame()["y_pred"].tolist()


def test_benchmark_long_form_rows_link_back_to_run_ids():
    dataset = _classification()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets={"rep": dataset},
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(metrics=("accuracy", "mcc"), seeds=(3,), include_baselines=False),
        partitions={"cv": _cv(dataset)},
    )
    run_ids = {run.run_id for run in result.runs}
    assert set(result.metrics_frame()["run_id"]).issubset(run_ids)
    assert set(result.predictions_frame()["run_id"]).issubset(run_ids)


def test_tuned_benchmark_rejects_plain_cv_as_unbiased_final_reporting():
    dataset = _classification()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(
            metrics=("accuracy",),
            modes=("tuned",),
            include_baselines=False,
            tuning=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy"),
        ),
        partitions=_cv(dataset),
        search_spaces={"logistic_regression": SearchSpace("lr", {"C": [0.1, 1.0]})},
    )
    assert len(result.failures) == 1
    assert "protected test" in result.failures[0].error


def test_tuned_holdout_uses_protected_test_only_for_final_metrics():
    dataset = _classification(90)
    ids = dataset.sample_ids
    plan = PartitionPlan.holdout(
        train_ids=ids[:50],
        validation_ids=ids[50:70],
        test_ids=ids[70:],
        dataset_fingerprint=dataset.fingerprint,
    )
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(
            metrics=("accuracy",),
            modes=("tuned",),
            include_baselines=False,
            seeds=(7,),
            tuning=TuningConfig(optimizer="grid", metrics=("accuracy",), refit_metric="accuracy", n_jobs=1),
        ),
        partitions=plan,
        search_spaces={"logistic_regression": SearchSpace("lr", {"C": [0.1, 1.0]})},
    )
    run = result.successes[0]
    assert run.validation.folds[0].evaluation_ids == tuple(ids[70:])
    assert run.optimization.metadata["protected_samples"] == 20


def test_regression_benchmark_with_dummy_baseline_and_multiple_seeds():
    X, y = make_regression(n_samples=60, n_features=5, noise=1.0, random_state=2)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"r{i}" for i in range(60)])
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("ridge_regressor",),
        config=BenchmarkConfig(metrics=("rmse", "mae"), seeds=(1, 2), include_baselines=True),
        partitions=_cv(dataset),
    )
    assert len(result.failures) == 0
    assert {run.algorithm for run in result.runs} == {"dummy_regressor", "ridge_regressor"}
    assert set(result.aggregate_metrics_frame()["metric"]) == {"rmse", "mae"}
