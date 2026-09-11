"""
mlcore.classification.lightgbm_models
=====================================

LightGBM classification models and registry wiring.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier

from mlcore.classification.backend import ClassificationBackend
from mlcore.classification.runners import make_classifier_runner

from mlcore.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec
from mlcore.classification import search_spaces

_ALIASES: dict[str, tuple[str, ...]] = {"lgbm_classifier": ("lightgbm_classifier",)}


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

        spec = AlgorithmSpec(
            backend="lightgbm",
            task="classification",
            name=name,
            estimator_cls=model_cls,
            runner=make_classifier_runner(model_cls),
            backend_cls=ClassificationBackend,
            aliases=_ALIASES.get(name, tuple()),
            capabilities=infer_estimator_capabilities(
                model_cls,
                native_missing_values=True,
            ),
            requirements=EstimatorRequirements(scaling="not_required"),
            supports_cv=True,
            search_space=search_space,
            tags=tags,
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_lightgbm_classification_models()