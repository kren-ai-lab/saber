"""Tests for the canonical AlgorithmSpec contract."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from saber.core.capabilities import EstimatorRequirements
from saber.core.search_space import SearchSpace
from saber.core.specs import AlgorithmSpec


def test_algorithm_spec_builds_factory_and_metadata() -> None:
    spec = AlgorithmSpec(
        provider="sklearn",
        task="classification",
        name="logistic_regression",
        estimator_cls=LogisticRegression,
        tags=("classification", "linear"),
        default_params={"max_iter": 250},
        search_space=SearchSpace({"C": [0.1, 1.0]}),
        requirements=EstimatorRequirements(scaling="recommended"),
    )

    estimator = spec.build_estimator(random_state=17, C=2.0)
    metadata = spec.metadata()

    assert estimator.C == 2.0
    assert estimator.max_iter == 250
    assert estimator.random_state == 17
    assert metadata["provider"] == "sklearn"
    assert metadata["capabilities"]["predict_proba"] is True
    assert metadata["requirements"]["scaling"] == "recommended"
    assert "backend" not in metadata
