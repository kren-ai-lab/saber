"""Tests for the canonical AlgorithmSpec contract."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
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
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
        requirements=EstimatorRequirements(scaling="recommended"),
        supports_cv=True,
    )

    estimator = spec.build_estimator(random_state=17, C=2.0)
    metadata = spec.metadata()

    assert estimator.C == 2.0
    assert estimator.max_iter == 250
    assert estimator.random_state == 17
    assert spec.has_tag("linear")
    assert spec.has_estimator_factory()
    assert spec.get_search_space() is spec.search_space
    assert metadata["provider"] == "sklearn"
    assert metadata["capabilities"]["predict_proba"] is True
    assert metadata["requirements"]["scaling"] == "recommended"
    assert "backend" not in metadata


def test_explicit_factory_class_mismatch_is_rejected() -> None:
    from saber.core.estimator import EstimatorFactory

    factory = EstimatorFactory(RandomForestClassifier)
    try:
        AlgorithmSpec(
            provider="sklearn",
            task="classification",
            name="bad",
            estimator_cls=LogisticRegression,
            estimator_factory=factory,
        )
    except ValueError as exc:
        assert "same estimator class" in str(exc)
    else:
        raise AssertionError("AlgorithmSpec accepted mismatched estimator construction.")
