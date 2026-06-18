"""
mlcore.regression.xgboost
=========================

XGBoost regression models and registry wiring.
"""

from __future__ import annotations

from xgboost import (
    XGBRegressor,
    XGBRFRegressor,
)

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec

from mlcore.regression.backend import RegressionBackend
from mlcore.regression.runners import make_regression_runner
from mlcore.regression import search_spaces

def register_xgboost_regression_models() -> None:
    """
    Register all XGBoost regression models.
    """

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

    for name, model_cls, tags, search_space in models:

        spec = AlgorithmSpec(
            backend="xgboost",
            task="regression",
            name=name,
            runner=make_regression_runner(
                model_cls,
            ),
            backend_cls=RegressionBackend,
            tags=tags,
            search_space=search_space
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(
        specs,
    )


register_xgboost_regression_models()