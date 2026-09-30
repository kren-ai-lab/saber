"""saber.regression.xgboost.

XGBoost regression model specs.
"""

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


def _build_specs() -> tuple[AlgorithmSpec, ...]:
    """Build all XGBoost regression models."""
    models = [
        (
            "xgb_regressor",
            XGBRegressor,
            (
                "regression",
                "xgboost",
                "tree",
                "boosting",
            ),
            search_spaces.XGB_REGRESSOR,
        ),
        (
            "xgbrf_regressor",
            XGBRFRegressor,
            (
                "regression",
                "xgboost",
                "tree",
                "random_forest",
            ),
            search_spaces.XGBRF_REGRESSOR,
        ),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_rgx, tags, search_space in models:
        spec = AlgorithmSpec(
            provider="xgboost",
            task="regression",
            name=name,
            estimator_cls=model_rgx,
            tags=tags,
            capabilities=infer_estimator_capabilities(
                model_rgx,
                native_missing_values=True,
            ),
            requirements=EstimatorRequirements(scaling="not_required"),
            search_space=search_space,
        )

        specs.append(spec)

    return tuple(specs)


SPECS: tuple[AlgorithmSpec, ...] = _build_specs()
