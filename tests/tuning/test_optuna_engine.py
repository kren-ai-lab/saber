from __future__ import annotations

from typing import Any

import pytest
from sklearn.datasets import make_classification

pytest.importorskip("optuna")

from saber.core.search_space import LogFloat, SearchSpace
from saber.datasets import DatasetBundle, PartitionPlan
from saber.tuning import TuningConfig, TuningEngine


def _inputs():
    X, y = make_classification(
        n_samples=54,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    ids = [f"sample_{i}" for i in range(len(y))]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=[i % 3 for i in range(len(y))],
        dataset_fingerprint=dataset.fingerprint,
    )
    return dataset, plan


def test_optuna_uses_typed_space_pipeline_and_multiple_metrics() -> None:
    dataset, plan = _inputs()
    result = TuningEngine().run(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(
            optimizer="optuna",
            metrics=("accuracy", "balanced_accuracy"),
            refit_metric="accuracy",
            n_trials=3,
            random_state=42,
            n_jobs=1,
        ),
        partition_plan=plan,
        search_space=SearchSpace("lr", {"C": LogFloat(1e-3, 10.0)}),
    )
    assert result.best_model is not None
    assert set(result.best_scores) == {"accuracy", "balanced_accuracy"}
    assert len(result.history) == 3
    assert all("status" in entry for entry in result.history)


def test_seeded_optuna_is_reproducible() -> None:
    dataset, plan = _inputs()
    config = TuningConfig(
        optimizer="optuna",
        metrics=("accuracy",),
        n_trials=4,
        random_state=91,
        n_jobs=1,
    )
    space = SearchSpace("lr", {"C": LogFloat(1e-3, 10.0)})
    first = TuningEngine().run(
        dataset=dataset,
        algorithm="logistic_regression",
        config=config,
        partition_plan=plan,
        search_space=space,
    )
    second = TuningEngine().run(
        dataset=dataset,
        algorithm="logistic_regression",
        config=config,
        partition_plan=plan,
        search_space=space,
    )
    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)


def test_optuna_storage_can_resume_existing_study(tmp_path) -> None:
    dataset, plan = _inputs()
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    base: dict[str, Any] = {
        "optimizer": "optuna",
        "metrics": ("accuracy",),
        "n_trials": 2,
        "random_state": 17,
        "n_jobs": 1,
        "optuna_storage": storage,
        "optuna_study_name": "phase5-resume",
        "optuna_load_if_exists": True,
    }
    space = SearchSpace("lr", {"C": LogFloat(1e-3, 10.0)})
    first = TuningEngine().run(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(**base),
        partition_plan=plan,
        search_space=space,
    )
    assert len(first.history) == 2
    first_values = [entry["params"]["C"] for entry in first.history]
    second = TuningEngine().run(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(**base),
        partition_plan=plan,
        search_space=space,
    )
    assert len(second.history) == 4
    second_new_values = [entry["params"]["C"] for entry in second.history[2:]]
    assert second_new_values != first_values
