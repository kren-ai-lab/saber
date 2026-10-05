"""LightGBM regression model specs."""

from __future__ import annotations

from lightgbm import LGBMRegressor

from saber.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.specs import AlgorithmSpec
from saber.regression import search_spaces

SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        provider="lightgbm",
        task="regression",
        name="lgbm_regressor",
        estimator_cls=LGBMRegressor,
        tags=("tree", "ensemble", "boosting"),
        capabilities=infer_estimator_capabilities(LGBMRegressor, native_missing_values=True),
        requirements=EstimatorRequirements(scaling="not_required"),
        search_space=search_spaces.LGBM_REGRESSOR,
    ),
)
