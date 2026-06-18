"""
mlcore.classification.sklearn_models
=====================================

Scikit-learn based classification algorithms and registry wiring.

This module defines:
- model runners
- algorithm specifications
- registry registration
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.naive_bayes import (
    GaussianNB,
    BernoulliNB,
    CategoricalNB,
    MultinomialNB,
    ComplementNB
)

from sklearn.svm import (
    SVC, 
    LinearSVC, 
    NuSVC
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
    NearestCentroid
)

from sklearn.tree import (
    DecisionTreeClassifier,
    ExtraTreeClassifier
)

from sklearn.linear_model import (
    RidgeClassifier,
    LogisticRegression,
    SGDClassifier
)

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from sklearn.gaussian_process import GaussianProcessClassifier

from mlcore.core.base import BackendBase
from mlcore.core.registry import AlgorithmRegistry
from mlcore.core.specs import AlgorithmSpec


# ============================================================
# Backend
# ============================================================

class SklearnBackend(BackendBase):
    """
    Backend for sklearn classifiers.
    """

    pass  # BackendBase already stores everything needed


# ============================================================
# Unified fit helper
# ============================================================

def _fit_model(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """
    Fit sklearn model safely.
    """

    model.fit(X, y)
    return model


# ============================================================
# Runner function factory
# ============================================================

def _make_runner(model_cls: Any):
    """
    Create a runner for a given sklearn estimator.
    """

    def runner(
        backend: SklearnBackend,
        X: np.ndarray,
        y: np.ndarray,
        **params: Any,
    ) -> None:

        model = model_cls(**params)

        model = _fit_model(model, X, y)

        backend.set_model(model)

        preds = model.predict(X)

        backend.set_predictions(preds)

        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                backend.set_probabilities(probs)
            except Exception:
                pass

    return runner


# ============================================================
# Registry instance (local)
# ============================================================

CLF_REGISTRY = AlgorithmRegistry()


# ============================================================
# Algorithm definitions
# ============================================================

def register_sklearn_classification_models() -> None:
    """
    Register all sklearn classification models.
    """

    models = [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("extra_trees", ExtraTreesClassifier),
        ("gradient_boosting", GradientBoostingClassifier),
        ("svc", SVC),
        ("knn", KNeighborsClassifier),
        ("decision_tree", DecisionTreeClassifier),
        ("adaboost", AdaBoostClassifier),
        ("bagging", BaggingClassifier),
        ("histgradient", HistGradientBoostingClassifier),
        ("linear_svc", LinearSVC),
        ("Nusvc", NuSVC),
        ("radius_neighbors", RadiusNeighborsClassifier),
        ("LDA", LinearDiscriminantAnalysis),
        ("QDA", QuadraticDiscriminantAnalysis),
        ("ridge", RidgeClassifier),
        ("SGD", SGDClassifier),
        ("extra_tree", ExtraTreeClassifier),
        ("nearest_centroid", NearestCentroid),
        ("gaussian_process", GaussianProcessClassifier),
        ("gaussian_nb", GaussianNB),
        ("bernoulli_nb", BernoulliNB),
        ("categorical_nb", CategoricalNB),
        ("multinomial_nb", MultinomialNB),
        ("complement_nb", ComplementNB)
    ]

    specs = []

    for name, model_cls in models:

        spec = AlgorithmSpec(
            backend="sklearn",
            task="classification",
            name=name,
            runner=_make_runner(model_cls),
            backend_cls=SklearnBackend,
            tags=("classification", "sklearn"),
            supports_proba=hasattr(model_cls(), "predict_proba"),
        )

        specs.append(spec)

    CLF_REGISTRY.register_many(specs)


# ============================================================
# Auto-register on import
# ============================================================

register_sklearn_classification_models()