from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from saber.benchmark import BenchmarkConfig, benchmark
from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import BenchmarkContractError


def _classification(n=60, seed=5):
    X, y = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=seed)
    ids = [f"s{i}" for i in range(n)]
    return DatasetBundle(X=X, y=y, sample_ids=ids)


def _cv(dataset, n=3):
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=np.arange(dataset.n_samples) % n,
        dataset=dataset,
    )


def test_representation_with_missing_sample_id_fails_before_runs():
    base = _classification()
    assert base.sample_ids is not None
    second = DatasetBundle(X=np.asarray(base.X)[:-1], y=base.y[:-1], sample_ids=base.sample_ids[:-1])
    with pytest.raises(BenchmarkContractError, match="same sample IDs"):
        benchmark(
            datasets={"a": base, "b": second},
            algorithms=("logistic_regression",),
            config=BenchmarkConfig(include_baselines=False),
            partitions=_cv(base),
            metrics=("accuracy",),
        )


def test_mixed_classification_and_regression_algorithms_are_rejected_globally():
    dataset = _classification()
    with pytest.raises(BenchmarkContractError, match="mix supervised tasks"):
        benchmark(
            datasets=dataset,
            algorithms=("logistic_regression", "ridge_regressor"),
            config=BenchmarkConfig(include_baselines=False),
            partitions=_cv(dataset),
            metrics=("accuracy",),
        )
