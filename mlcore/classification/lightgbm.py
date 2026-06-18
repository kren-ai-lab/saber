"""
mlcore.classification.lightgbm_models
======================================

LightGBM based classification models and registry wiring.

This module defines:
- LightGBM classifier runners
- AlgorithmSpec definitions
- automatic registry registration
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from lightgbm import LGBMClassifier

from mlcore.core.base import BackendBase
from mlcore.core.registry import AlgorithmRegistry
from mlcore.core.specs import AlgorithmSpec


# ============================================================
# Backend
# ============================================================

class LightGBMBackend(BackendBase):
    """
    Backend for LightGBM models.
    """

    pass  # inherits full state handling from BackendBase


# ============================================================
# Runner factory
# ============================================================

def _make_runner(model_cls: Any):
    """
    Create a unified LightGBM runner.
    """

    def runner(
        backend: LightGBMBackend,
        X: np.ndarray,
        y: np.ndarray,
        **params: Any,
    ) -> None:

        model = model_cls(**params)

        model.fit(X, y)

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
# Registry
# ============================================================

LGBM_REGISTRY = AlgorithmRegistry()


# ============================================================
# Registration
# ============================================================

def register_lightgbm_classification_models() -> None:
    """
    Register all LightGBM classification models.
    """

    models = [
        ("lgbm_classifier", LGBMClassifier),
    ]

    specs = []

    for name, model_cls in models:

        spec = AlgorithmSpec(
            backend="lightgbm",
            task="classification",
            name=name,
            runner=_make_runner(model_cls),
            backend_cls=LightGBMBackend,
            tags=("classification", "lightgbm", "tree", "boosting"),
            supports_proba=True,
            description="LightGBM classifier wrapper",
        )

        specs.append(spec)

    LGBM_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_lightgbm_classification_models()