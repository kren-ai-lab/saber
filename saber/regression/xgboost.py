"""XGBoost regression model specs."""

from __future__ import annotations

from xgboost import (
    XGBRegressor,
    XGBRFRegressor,
)

from saber.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.specs import AlgorithmSpec
from saber.regression import search_spaces

SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        provider="xgboost",
        task="regression",
        name="xgb_regressor",
        estimator_cls=XGBRegressor,
        tags=("tree", "ensemble", "boosting"),
        capabilities=infer_estimator_capabilities(XGBRegressor, native_missing_values=True),
        requirements=EstimatorRequirements(scaling="not_required"),
        search_space=search_spaces.XGB_REGRESSOR,
    ),
    AlgorithmSpec(
        provider="xgboost",
        task="regression",
        name="xgb_rf_regressor",
        estimator_cls=XGBRFRegressor,
        tags=("tree", "ensemble", "bagging"),
        capabilities=infer_estimator_capabilities(XGBRFRegressor, native_missing_values=True),
        requirements=EstimatorRequirements(scaling="not_required"),
        search_space=search_spaces.XGBRF_REGRESSOR,
    ),
)
