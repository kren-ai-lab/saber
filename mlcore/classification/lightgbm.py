"""
mlcore.classification.lightgbm_models
=====================================

LightGBM classification models and registry wiring.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier

from mlcore.classification.backend import ClassificationBackend
from mlcore.classification.runners import make_classifier_runner

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec
from mlcore.classification import search_spaces

# ============================================================
# Registration
# ============================================================

def register_lightgbm_classification_models() -> None:
    """
    Register all LightGBM classification models.
    """

    models = [
        ("lgbm_classifier", LGBMClassifier, ("classification", "lightgbm", "tree", "boosting"), search_spaces.LGBM_CLASSIFIER),
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
            backend="lightgbm",
            task="classification",
            name=name,
            estimator_cls=model_cls,
            runner=make_classifier_runner(model_cls),
            backend_cls=ClassificationBackend,
            supports_proba=supports_proba,
            search_space=search_space,
            tags=tags
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_lightgbm_classification_models()