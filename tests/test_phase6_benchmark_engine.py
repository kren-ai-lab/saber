from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from mlcore import MODEL_REGISTRY
from mlcore.benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkEngine,
    BenchmarkPartition,
)
from mlcore.core.search_space import SearchSpace
from mlcore.datasets import DatasetBundle, PartitionPlan
from mlcore.exceptions import BenchmarkContractError
from mlcore.tuning import TuningConfig


def _classification_dataset(seed: int = 42, *, scale: float = 1.0) -> DatasetBundle:
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        random_state=seed,
    )
    return DatasetBundle(
        X=X * scale,
        y=y,
        sample_ids=[f"s{i}" for i in range(60)],
    )


def _cv_plan(dataset: DatasetBundle, offset: int = 0) -> PartitionPlan:
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=[(i + offset) % 3 for i in range(dataset.n_samples)],
        dataset_fingerprint=dataset.fingerprint,
    )


def _safe_holdout(dataset: DatasetBundle) -> PartitionPlan:
    ids = tuple(dataset.sample_ids)
    return PartitionPlan.holdout(
        train_ids=ids[:40],
        validation_ids=ids[40:50],
        test_ids=ids[50:],
        dataset_fingerprint=dataset.fingerprint,
    )


def test_benchmark_runs_algorithm_matrix_with_baseline_and_repeated_seeds() -> None:
    dataset = _classification_dataset()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression", "decision_tree"),
        config=BenchmarkConfig(
            metrics=("accuracy", "balanced_accuracy"),
            seeds=(3, 7),
            include_baselines=True,
        ),
        partitions=_cv_plan(dataset),
    )

    assert result.n_runs == 6
    assert len(result.failures) == 0
    assert {run.mode for run in result.runs if run.algorithm == "dummy_classifier"} == {"baseline"}
    assert {run.algorithm for run in result.successes} == {
        "dummy_classifier",
        "logistic_regression",
        "decision_tree",
    }
    assert all(run.oof_prediction is not None for run in result.successes)
    assert all(run.oof_prediction.n_samples == 60 for run in result.successes)


def test_benchmark_multiple_representations_reuse_identical_partition_membership() -> None:
    roxy = _classification_dataset(scale=1.0)
    sylphy = DatasetBundle(
        X=np.asarray(roxy.X) * 10.0 + 1.5,
        y=np.asarray(roxy.y).copy(),
        sample_ids=np.asarray(roxy.sample_ids).copy(),
        feature_names=[f"emb_{i}" for i in range(roxy.n_features)],
    )
    plan = _cv_plan(roxy)

    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=(
            BenchmarkDataset("roxy", roxy, representation="roxy"),
            BenchmarkDataset("sylphy", sylphy, representation="sylphy"),
        ),
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(
            metrics=("accuracy",),
            seeds=(42,),
            include_baselines=False,
        ),
        partitions=BenchmarkPartition("cluster_disjoint", plan),
    )

    assert len(result.successes) == 2
    roxy_run, sylphy_run = result.successes
    for left, right in zip(roxy_run.validation.folds, sylphy_run.validation.folds, strict=True):
        assert left.train_ids == right.train_ids
        assert left.evaluation_ids == right.evaluation_ids
    assert roxy_run.metadata["partition_fingerprint"] != sylphy_run.metadata["partition_fingerprint"]


def test_representation_benchmark_rejects_target_mismatch() -> None:
    first = _classification_dataset()
    wrong_y = np.asarray(first.y).copy()
    wrong_y[0] = 1 - wrong_y[0]
    second = DatasetBundle(
        X=np.asarray(first.X).copy(),
        y=wrong_y,
        sample_ids=np.asarray(first.sample_ids).copy(),
    )
    with pytest.raises(BenchmarkContractError, match="identical targets"):
        BenchmarkEngine(MODEL_REGISTRY).run(
            datasets={"first": first, "second": second},
            algorithms=("logistic_regression",),
            config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
            partitions=_cv_plan(first),
        )


def test_multiple_partition_scenarios_are_explicit_in_long_form_tables() -> None:
    dataset = _classification_dataset()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets={"roxy": dataset},
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(metrics=("accuracy", "mcc"), include_baselines=False),
        partitions={
            "split_a": _cv_plan(dataset, 0),
            "split_b": _cv_plan(dataset, 1),
        },
    )
    aggregate = result.aggregate_metrics_frame()
    folds = result.fold_metrics_frame()
    assert set(aggregate["partition"]) == {"split_a", "split_b"}
    assert set(folds["partition"]) == {"split_a", "split_b"}
    assert set(folds["split"]) == {"fold_0", "fold_1", "fold_2"}
    assert set(aggregate["metric"]) == {"accuracy", "mcc"}


def test_failed_algorithm_does_not_invalidate_successful_runs() -> None:
    dataset = _classification_dataset()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression", "not_a_model"),
        config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
        partitions=_cv_plan(dataset),
    )
    assert len(result.successes) == 1
    assert len(result.failures) == 1
    assert result.successes[0].algorithm == "logistic_regression"
    assert result.failures[0].algorithm == "not_a_model"
    assert "AlgorithmNotFoundError" in result.failures[0].error


def test_fail_fast_raises_after_recordable_run_failure() -> None:
    dataset = _classification_dataset()
    with pytest.raises(BenchmarkContractError, match="AlgorithmNotFoundError"):
        BenchmarkEngine(MODEL_REGISTRY).run(
            datasets=dataset,
            algorithms=("logistic_regression", "not_a_model"),
            config=BenchmarkConfig(
                metrics=("accuracy",),
                include_baselines=False,
                fail_fast=True,
            ),
            partitions=_cv_plan(dataset),
        )


def test_seeded_benchmark_is_score_reproducible() -> None:
    dataset = _classification_dataset()
    kwargs = dict(
        datasets=dataset,
        algorithms=("random_forest",),
        config=BenchmarkConfig(
            metrics=("accuracy", "mcc"),
            seeds=(123,),
            include_baselines=False,
        ),
        partitions=_cv_plan(dataset),
        model_params={"random_forest": {"n_estimators": 20}},
    )
    engine = BenchmarkEngine(MODEL_REGISTRY)
    first = engine.run(**kwargs)
    second = engine.run(**kwargs)
    pd.testing.assert_frame_equal(
        first.aggregate_metrics_frame().drop(columns="elapsed_seconds"),
        second.aggregate_metrics_frame().drop(columns="elapsed_seconds"),
    )
    np.testing.assert_array_equal(
        first.successes[0].oof_prediction.predictions,
        second.successes[0].oof_prediction.predictions,
    )


def test_predictions_frame_retains_truth_predictions_and_probabilities() -> None:
    dataset = _classification_dataset()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets={"roxy": dataset},
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
        partitions=_cv_plan(dataset),
    )
    frame = result.predictions_frame()
    assert len(frame) == dataset.n_samples
    assert {"sample_id", "y_true", "y_pred", "split", "run_id"} <= set(frame.columns)
    assert any(column.startswith("probability__") for column in frame.columns)
    assert set(frame["sample_id"]) == set(dataset.sample_ids)


def test_tuned_benchmark_reports_only_protected_final_test() -> None:
    dataset = _classification_dataset()
    plan = _safe_holdout(dataset)
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=BenchmarkDataset("roxy", dataset, representation="roxy"),
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(
            metrics=("accuracy", "mcc"),
            seeds=(42,),
            modes=("tuned",),
            include_baselines=False,
            tuning=TuningConfig(
                optimizer="grid",
                metrics=("accuracy",),
                refit_metric="accuracy",
                n_jobs=1,
            ),
        ),
        partitions=plan,
        search_spaces={
            "logistic_regression": SearchSpace("lr", {"C": [0.1, 1.0]}),
        },
    )

    assert len(result.successes) == 1
    run = result.successes[0]
    assert run.mode == "tuned"
    assert run.optimization is not None
    assert run.optimization.metadata["protected_samples"] == 10
    assert run.validation.folds[0].evaluation_role == "test"
    assert set(run.validation.folds[0].evaluation_ids) == set(dataset.sample_ids[50:])
    assert len(run.validation.folds[0].train_ids) == 50
    assert run.oof_prediction.n_samples == 10


def test_tuned_cv_is_rejected_without_destroying_untuned_result() -> None:
    dataset = _classification_dataset()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(
            metrics=("accuracy",),
            modes=("untuned", "tuned"),
            include_baselines=False,
            tuning=TuningConfig(
                optimizer="grid",
                metrics=("accuracy",),
                n_jobs=1,
            ),
        ),
        partitions=_cv_plan(dataset),
        search_spaces={"logistic_regression": SearchSpace("lr", {"C": [1.0]})},
    )
    assert len(result.successes) == 1
    assert result.successes[0].mode == "untuned"
    assert len(result.failures) == 1
    assert result.failures[0].mode == "tuned"
    assert "nested CV" in result.failures[0].error


def test_regression_benchmark_includes_dummy_regressor_baseline() -> None:
    X, y = make_regression(n_samples=60, n_features=5, noise=1.0, random_state=4)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"r{i}" for i in range(60)])
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("ridge_regressor",),
        config=BenchmarkConfig(metrics=("rmse", "r2"), include_baselines=True),
        partitions=_cv_plan(dataset),
    )
    assert {run.algorithm for run in result.successes} == {
        "dummy_regressor",
        "ridge_regressor",
    }
    assert all("rmse" in run.aggregate_metrics for run in result.successes)


def test_run_and_failure_tables_are_exportable() -> None:
    dataset = _classification_dataset()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression", "bad_model"),
        config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
        partitions=_cv_plan(dataset),
    )
    runs = result.runs_frame()
    failures = result.failures_frame()
    assert len(runs) == 2
    assert set(runs["status"]) == {"complete", "failed"}
    assert len(failures) == 1
    assert failures.iloc[0]["algorithm"] == "bad_model"


def test_partition_from_unrelated_dataset_fingerprint_is_rejected() -> None:
    dataset = _classification_dataset()
    unrelated = _classification_dataset(seed=99)
    foreign_plan = _cv_plan(unrelated)
    with pytest.raises(BenchmarkContractError, match="fingerprint"):
        BenchmarkEngine(MODEL_REGISTRY).run(
            datasets=dataset,
            algorithms=("logistic_regression",),
            config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
            partitions=foreign_plan,
        )


def test_metric_rows_link_to_configuration_and_prediction_rows_by_run_id() -> None:
    dataset = _classification_dataset()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(metrics=("accuracy",), include_baselines=False),
        partitions=_cv_plan(dataset),
        model_params={"logistic_regression": {"C": 0.5}},
    )
    metrics = result.fold_metrics_frame()
    predictions = result.predictions_frame()
    assert set(metrics["run_id"]) == set(predictions["run_id"])
    assert set(metrics["configuration_id"]) == set(predictions["configuration_id"])
    assert metrics.iloc[0]["parameters"]["C"] == 0.5


def test_tuned_benchmark_exports_annotated_optimization_history() -> None:
    dataset = _classification_dataset()
    result = BenchmarkEngine(MODEL_REGISTRY).run(
        datasets=dataset,
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(
            metrics=("accuracy",),
            modes=("tuned",),
            include_baselines=False,
            tuning=TuningConfig(optimizer="grid", metrics=("accuracy",), n_jobs=1),
        ),
        partitions=_safe_holdout(dataset),
        search_spaces={"logistic_regression": SearchSpace("lr", {"C": [0.1, 1.0]})},
    )
    history = result.optimization_history_frame()
    assert len(history) == 2
    assert set(history["algorithm"]) == {"logistic_regression"}
    assert set(history["mode"]) == {"tuned"}
    assert "param__C" in history.columns
