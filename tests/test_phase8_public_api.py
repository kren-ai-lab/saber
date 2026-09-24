from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification, make_regression

import saber
from saber.benchmark import BenchmarkConfig
from saber.core import Categorical, SearchSpace
from saber.datasets import DatasetBundle, PartitionPlan
from saber.preprocessing import PreprocessingConfig
from saber.tuning import TuningConfig


def _classification_dataset(n=72):
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        random_state=7,
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return DatasetBundle(frame, y, sample_ids=[f"s{i}" for i in range(n)])


def _cv_plan(dataset):
    folds = [i % 3 for i in range(dataset.n_samples)]
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=folds,
        dataset_fingerprint=dataset.fingerprint,
    )


def test_top_level_public_api_exports():
    for name in (
        "train",
        "validate",
        "evaluate",
        "tune",
        "benchmark",
        "predict",
        "load_config",
        "run_config",
        "save_model",
        "load_model",
    ):
        assert hasattr(saber, name)


def test_train_predict_and_evaluate_share_prediction_contract():
    dataset = _classification_dataset()
    result = saber.train(
        dataset=dataset,
        algorithm="logistic_regression",
        preprocessing=PreprocessingConfig(scaler="standard"),
        random_state=42,
    )
    prediction = saber.predict(result, dataset=dataset)
    evaluation = saber.evaluate(dataset=dataset, model=result, metrics=("accuracy", "roc_auc"))

    assert prediction.n_samples == dataset.n_samples
    assert prediction.sample_ids.tolist() == list(dataset.sample_ids)
    assert set(evaluation.metrics) == {"accuracy", "roc_auc"}
    assert evaluation.prediction is not None


def test_public_validate_matches_engine_contract():
    dataset = _classification_dataset()
    plan = _cv_plan(dataset)
    result = saber.validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=("accuracy", "balanced_accuracy"),
        random_state=42,
    )
    assert result.n_splits == 3
    assert result.oof_prediction is not None
    assert result.oof_prediction.n_samples == dataset.n_samples
    assert set(result.aggregate_metrics) == {"accuracy", "balanced_accuracy"}


def test_public_tune_uses_explicit_partition_plan_and_typed_space():
    dataset = _classification_dataset()
    plan = _cv_plan(dataset)
    result = saber.tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(
            optimizer="grid",
            metrics=("accuracy",),
            refit_metric="accuracy",
            random_state=42,
        ),
        partition_plan=plan,
        search_space=SearchSpace("logreg", {"C": Categorical([0.1, 1.0])}),
    )
    assert result.partition_plan.fingerprint == plan.fingerprint
    assert result.best_params["C"] in {0.1, 1.0}
    assert result.best_model is not None


def test_public_benchmark_runs_matrix_without_reimplementing_engines():
    dataset = _classification_dataset(60)
    plan = _cv_plan(dataset)
    result = saber.benchmark(
        datasets={"prepared": dataset},
        algorithms=("logistic_regression",),
        config=BenchmarkConfig(
            metrics=("accuracy",),
            seeds=(42, 43),
            include_baselines=False,
        ),
        partitions={"cv": plan},
    )
    assert result.n_runs == 2
    assert len(result.successes) == 2
    assert set(result.aggregate_metrics_frame()["seed"]) == {42, 43}


def test_public_api_regression_end_to_end():
    X, y = make_regression(n_samples=60, n_features=5, noise=0.2, random_state=3)
    dataset = DatasetBundle(
        pd.DataFrame(X, columns=[f"x{i}" for i in range(5)]),
        y,
        sample_ids=[f"r{i}" for i in range(60)],
    )
    result = saber.train(dataset=dataset, algorithm="ridge_regressor", random_state=42)
    evaluation = saber.evaluate(dataset=dataset, model=result, metrics=("rmse", "mae"))
    assert evaluation.metrics["rmse"] >= 0.0
    assert evaluation.metrics["mae"] >= 0.0
