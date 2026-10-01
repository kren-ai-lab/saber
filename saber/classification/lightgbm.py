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

SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        provider="lightgbm",
        task="classification",
        name="lgbm_classifier",
        estimator_cls=LGBMClassifier,
        tags=("tree", "ensemble", "boosting"),
        capabilities=infer_estimator_capabilities(LGBMClassifier, native_missing_values=True),
        requirements=EstimatorRequirements(scaling="not_required"),
        search_space=search_spaces.LGBM_CLASSIFIER,
    ),
)
