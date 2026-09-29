"""Regression algorithm registration."""

from __future__ import annotations

from saber._optional import is_dependency_available
from saber.regression.sklearn import register_sklearn_regression_models

__all__ = ["register_sklearn_regression_models"]

if is_dependency_available("xgboost"):
    from saber.regression.xgboost import register_xgboost_regression_models

    __all__ += ["register_xgboost_regression_models"]

if is_dependency_available("lightgbm"):
    from saber.regression.lightgbm import register_lightgbm_regression_models

    __all__ += ["register_lightgbm_regression_models"]
