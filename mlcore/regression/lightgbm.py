"""
mlcore.regression.lightgbm
==========================

LightGBM regression models and registry wiring.
"""

from __future__ import annotations

from lightgbm import LGBMRegressor

from mlcore.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec

from mlcore.regression.backend import RegressionBackend
from mlcore.regression.runners import make_regression_runner
from mlcore.regression import search_spaces

_ALIASES: dict[str, tuple[str, ...]] = {"lgbm_regressor": ("lightgbm_regressor",)}


def register_lightgbm_regression_models() -> None:
    """
    Register all LightGBM regression models.
    """

    models = [
        (
            "lgbm_regressor",
            LGBMRegressor,
            (
                "regression",
                "lightgbm",
                "tree",
                "boosting",
            ),
            search_spaces.LGBM_REGRESSOR
        ),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_rgx, tags, search_space in models:

        spec = AlgorithmSpec(
            backend="lightgbm",
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


register_lightgbm_regression_models()