"""saber.classification.xgboost_models.

XGBoost classification model specs.
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
from saber.core.specs import AlgorithmSpec

SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        provider="xgboost",
        task="classification",
        name="xgb_classifier",
        estimator_cls=XGBClassifier,
        tags=("tree", "ensemble", "boosting"),
        capabilities=infer_estimator_capabilities(XGBClassifier, native_missing_values=True),
        requirements=EstimatorRequirements(scaling="not_required"),
        search_space=search_spaces.XGB_CLASSIFIER,
    ),
    AlgorithmSpec(
        provider="xgboost",
        task="classification",
        name="xgb_rf_classifier",
        estimator_cls=XGBRFClassifier,
        tags=("tree", "ensemble", "bagging"),
        capabilities=infer_estimator_capabilities(XGBRFClassifier, native_missing_values=True),
        requirements=EstimatorRequirements(scaling="not_required"),
        search_space=search_spaces.XGB_RF_CLASSIFIER,
    ),
)
