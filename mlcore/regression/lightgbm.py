"""
mlcore.regression.lightgbm
==========================

LightGBM regression models and registry wiring.
"""

from __future__ import annotations

from lightgbm import LGBMRegressor

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec

from mlcore.regression.backend import RegressionBackend
from mlcore.regression.runners import make_regression_runner


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
        ),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls, tags in models:

        spec = AlgorithmSpec(
            backend="lightgbm",
            task="regression",
            name=name,
            runner=make_regression_runner(
                model_cls,
            ),
            backend_cls=RegressionBackend,
            tags=tags,
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(
        specs,
    )


register_lightgbm_regression_models()