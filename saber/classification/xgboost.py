"""saber.classification.xgboost_models
====================================

XGBoost classification models and registry wiring.
"""

from __future__ import annotations

from xgboost import (
    XGBClassifier,
    XGBRFClassifier,
)

from saber.classification import search_spaces
from saber.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.registry import MODEL_REGISTRY
from saber.core.specs import AlgorithmSpec

_ALIASES: dict[str, tuple[str, ...]] = {
    "xgb_classifier": ("xgboost_classifier",),
    "xgb_rf_classifier": ("xgboost_rf_classifier",),
}


# ============================================================
# Registration
# ============================================================


def register_xgboost_classification_models() -> None:
    """Register all XGBoost classification models.
    """
    models = [
        (
            "xgb_classifier",
            XGBClassifier,
            ("classification", "xgboost", "tree", "boosting"),
            search_spaces.XGB_CLASSIFIER,
        ),
        (
            "xgb_rf_classifier",
            XGBRFClassifier,
            ("classification", "xgboost", "tree", "boosting"),
            search_spaces.XGB_RF_CLASSIFIER,
        ),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls, tags, search_space in models:
        spec = AlgorithmSpec(
            provider="xgboost",
            task="classification",
            name=name,
            estimator_cls=model_cls,
            aliases=_ALIASES.get(name, tuple()),
            tags=tags,
            capabilities=infer_estimator_capabilities(
                model_cls,
                native_missing_values=True,
            ),
            requirements=EstimatorRequirements(scaling="not_required"),
            supports_cv=True,
            search_space=search_space,
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_xgboost_classification_models()
