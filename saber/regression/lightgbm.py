"""saber.regression.lightgbm.

LightGBM regression models and registry wiring.
"""

from __future__ import annotations

from lightgbm import LGBMRegressor

from saber.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.registry import MODEL_REGISTRY
from saber.core.specs import AlgorithmSpec
from saber.regression import search_spaces

_ALIASES: dict[str, tuple[str, ...]] = {"lgbm_regressor": ("lightgbm_regressor",)}


def register_lightgbm_regression_models() -> None:
    """Register all LightGBM regression models."""
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
            search_spaces.LGBM_REGRESSOR,
        ),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_rgx, tags, search_space in models:
        spec = AlgorithmSpec(
            provider="lightgbm",
            task="regression",
            name=name,
            estimator_cls=model_rgx,
            aliases=_ALIASES.get(name, ()),
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
