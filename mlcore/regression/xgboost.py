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

from mlcore.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec

from mlcore.regression.backend import RegressionBackend
from mlcore.regression.runners import make_regression_runner
from mlcore.regression import search_spaces

_ALIASES: dict[str, tuple[str, ...]] = {"xgb_regressor": ("xgboost_regressor",), "xgbrf_regressor": ("xgboost_rf_regressor",)}


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

    for name, model_rgx, tags, search_space in models:

        spec = AlgorithmSpec(
            backend="xgboost",
            task="regression",
            name=name,
            estimator_cls=model_rgx,
            runner=make_regression_runner(
                model_rgx,
            ),
            backend_cls=RegressionBackend,
            aliases=_ALIASES.get(name, tuple()),
            tags=tags,
            capabilities=infer_estimator_capabilities(
                model_rgx,
                native_missing_values=True,
            ),
            requirements=EstimatorRequirements(scaling="not_required"),
            supports_cv=True,
            search_space=search_space,
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(
        specs,
    )


register_xgboost_regression_models()