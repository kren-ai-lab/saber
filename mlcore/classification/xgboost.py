"""
mlcore.classification.xgboost_models
=====================================

XGBoost based classification models and registry wiring.

This module defines:
- XGBoost classifier runners
- AlgorithmSpec definitions
- automatic registry registration
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from xgboost import XGBClassifier, XGBRFClassifier

from mlcore.core.base import BackendBase
from mlcore.core.registry import AlgorithmRegistry
from mlcore.core.specs import AlgorithmSpec


# ============================================================
# Backend
# ============================================================

class XGBoostBackend(BackendBase):
    """
    Backend for XGBoost models.
    """

    pass  # inherits full state management from BackendBase


# ============================================================
# Runner factory
# ============================================================

def _make_runner(model_cls: Any):
    """
    Create a unified XGBoost runner.
    """

    def runner(
        backend: XGBoostBackend,
        X: np.ndarray,
        y: np.ndarray,
        **params: Any,
    ) -> None:

        model = model_cls(**params)

        model.fit(X, y)

        backend.set_model(model)

        preds = model.predict(X)
        backend.set_predictions(preds)

        # probability support
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                backend.set_probabilities(probs)
            except Exception:
                pass

    return runner


# ============================================================
# Registry
# ============================================================

XGB_REGISTRY = AlgorithmRegistry()


# ============================================================
# Registration function
# ============================================================

def register_xgboost_classification_models() -> None:
    """
    Register all XGBoost classification models.
    """

    models = [
        ("xgb_classifier", XGBClassifier),
        ("xgbr_classifier", XGBRFClassifier),
    ]

    specs = []

    for name, model_cls in models:

        spec = AlgorithmSpec(
            backend="xgboost",
            task="classification",
            name=name,
            runner=_make_runner(model_cls),
            backend_cls=XGBoostBackend,
            tags=("classification", "xgboost", "tree"),
            supports_proba=True,
            description="XGBoost classifier wrapper",
        )

        specs.append(spec)

    XGB_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_xgboost_classification_models()