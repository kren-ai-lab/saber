"""Unit tests for canonical estimator construction."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlcore.core.estimator import EstimatorFactory


def test_factory_applies_default_and_override_params() -> None:
    factory = EstimatorFactory(
        LogisticRegression,
        default_params={"C": 2.0, "max_iter": 250},
    )

    estimator = factory.build(C=3.0)

    assert estimator.C == 3.0
    assert estimator.max_iter == 250


def test_factory_injects_random_state_when_supported() -> None:
    factory = EstimatorFactory(RandomForestClassifier)

    estimator = factory.build(random_state=17, n_estimators=5)

    assert estimator.random_state == 17
    assert estimator.n_estimators == 5


def test_explicit_random_state_overrides_factory_default() -> None:
    factory = EstimatorFactory(
        RandomForestClassifier,
        default_params={"random_state": 9},
    )

    estimator = factory.build(random_state=17)

    assert estimator.random_state == 17


def test_factory_does_not_inject_unsupported_random_state() -> None:
    class SimpleEstimator:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    factory = EstimatorFactory(SimpleEstimator)
    estimator = factory.build(random_state=17, value=4)

    assert estimator.value == 4
