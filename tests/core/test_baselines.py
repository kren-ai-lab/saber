from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor

from saber import MODEL_REGISTRY


def test_classification_baseline_is_registered_with_deterministic_default() -> None:
    spec = MODEL_REGISTRY.get("dummy_classifier")
    assert spec.task == "classification"
    assert "baseline" in spec.tags
    assert spec.default_params["strategy"] == "prior"
    model = spec.build_estimator(random_state=42)
    assert isinstance(model, DummyClassifier)
    # sklearn stubs omit this constructor-stored attribute.
    assert model.strategy == "prior"  # pyrefly: ignore[missing-attribute]


def test_regression_baseline_is_registered_with_deterministic_default() -> None:
    spec = MODEL_REGISTRY.get("dummy_regressor")
    assert spec.task == "regression"
    assert "baseline" in spec.tags
    assert spec.default_params["strategy"] == "mean"
    model = spec.build_estimator(random_state=42)
    assert isinstance(model, DummyRegressor)
    # sklearn stubs omit this constructor-stored attribute.
    assert model.strategy == "mean"  # pyrefly: ignore[missing-attribute]
