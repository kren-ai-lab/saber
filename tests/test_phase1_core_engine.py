"""Phase 1 acceptance tests for the unified estimator engine."""

from __future__ import annotations

from sklearn.datasets import make_classification, make_regression

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.trainer import Trainer
from mlcore.core.search_space import SearchSpace
from mlcore.tuning.sklearn.grid import GridSearchOptimizer


def test_registered_specs_have_canonical_factories() -> None:
    for spec in MODEL_REGISTRY:
        assert spec.has_estimator_factory()
        assert spec.estimator_factory is not None
        assert spec.estimator_cls is spec.estimator_factory.estimator_cls


def test_registry_metadata_exposes_capabilities_and_requirements() -> None:
    metadata = MODEL_REGISTRY.describe("random_forest")

    assert metadata["provider"] == "sklearn"
    assert metadata["has_estimator_factory"] is True
    assert "capabilities" in metadata
    assert "requirements" in metadata
    assert metadata["capabilities"]["predict_proba"] is True


def test_training_uses_factory_defaults_and_seed() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        random_state=7,
    )

    trainer = Trainer(MODEL_REGISTRY)
    result = trainer.fit(
        "random_forest",
        X,
        y,
        random_state=23,
        n_estimators=7,
    )

    assert result.model.random_state == 23
    assert result.model.n_estimators == 7
    assert result.backend is not None
    assert result.backend.get_model() is result.model


def test_grid_search_and_training_share_estimator_construction() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        random_state=11,
    )

    trainer = Trainer(MODEL_REGISTRY)
    trained = trainer.fit(
        "logistic_regression",
        X,
        y,
        random_state=31,
        C=1.5,
        max_iter=200,
    )

    optimizer = GridSearchOptimizer(MODEL_REGISTRY)
    optimized = optimizer.optimize(
        "logistic_regression",
        X,
        y,
        metric="accuracy",
        cv=3,
        random_state=31,
        search_space=SearchSpace("phase1", {"C": [1.5], "max_iter": [200]}),
    )

    direct_params = trained.model.get_params(deep=False)
    tuned_params = optimized.best_model.get_params(deep=False)

    assert direct_params["C"] == tuned_params["C"] == 1.5
    assert direct_params["max_iter"] == tuned_params["max_iter"] == 200
    assert direct_params["random_state"] == tuned_params["random_state"] == 31


def test_regression_trainer_uses_same_canonical_factory() -> None:
    X, y = make_regression(
        n_samples=120,
        n_features=8,
        random_state=13,
    )

    trainer = Trainer(MODEL_REGISTRY)
    result = trainer.fit(
        "random_forest_regressor",
        X,
        y,
        random_state=29,
        n_estimators=6,
    )

    spec = MODEL_REGISTRY.get("random_forest_regressor")
    fresh = spec.build_estimator(random_state=29, n_estimators=6)

    assert type(result.model) is type(fresh)
    assert result.model.random_state == fresh.random_state == 29
    assert result.model.n_estimators == fresh.n_estimators == 6
