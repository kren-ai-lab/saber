"""
mlcore.classification.xgboost_models
====================================

XGBoost classification models and registry wiring.
"""

from __future__ import annotations

from xgboost import (
    XGBClassifier,
    XGBRFClassifier,
)

from mlcore.classification.backend import ClassificationBackend
from mlcore.classification.runners import make_classifier_runner

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec
from mlcore.classification import search_spaces

# ============================================================
# Registration
# ============================================================

def register_xgboost_classification_models() -> None:
    """
    Register all XGBoost classification models.
    """

    models = [
        ("xgb_classifier", XGBClassifier, ("classification", "xgboost", "tree", "boosting"), search_spaces.XGB_CLASSIFIER),
        ("xgb_rf_classifier", XGBRFClassifier, ("classification", "xgboost", "tree", "boosting"), search_spaces.XGB_RF_CLASSIFIER),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls, tags, search_space in models:

        try:
            supports_proba = hasattr(
                model_cls(),
                "predict_proba",
            )

        except Exception:
            supports_proba = False

        spec = AlgorithmSpec(
            backend="xgboost",
            task="classification",
            name=name,
            runner=make_classifier_runner(model_cls),
            backend_cls=ClassificationBackend,
            tags=tags,
            supports_proba=supports_proba,
            search_space=search_space
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_xgboost_classification_models()