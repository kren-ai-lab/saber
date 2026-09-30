"""saber.classification.lightgbm_models.

LightGBM classification model specs.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier

from saber.classification import search_spaces
from saber.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.specs import AlgorithmSpec

# ============================================================
# Specs
# ============================================================


def _build_specs() -> tuple[AlgorithmSpec, ...]:
    """Build all LightGBM classification models."""
    models = [
        (
            "lgbm_classifier",
            LGBMClassifier,
            ("classification", "lightgbm", "tree", "boosting"),
            search_spaces.LGBM_CLASSIFIER,
        ),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls, tags, search_space in models:
        spec = AlgorithmSpec(
            provider="lightgbm",
            task="classification",
            name=name,
            estimator_cls=model_cls,
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

    return tuple(specs)


# ============================================================
# Catalog
# ============================================================

SPECS: tuple[AlgorithmSpec, ...] = _build_specs()
