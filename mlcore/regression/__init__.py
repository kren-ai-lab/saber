"""Regression algorithm registration."""

from __future__ import annotations

from mlcore._optional import is_dependency_available
from mlcore.regression.sklearn import register_sklearn_regression_models

__all__ = ["register_sklearn_regression_models"]

if is_dependency_available("xgboost"):
    from mlcore.regression.xgboost import register_xgboost_regression_models

    __all__.append("register_xgboost_regression_models")

if is_dependency_available("lightgbm"):
    from mlcore.regression.lightgbm import register_lightgbm_regression_models

    __all__.append("register_lightgbm_regression_models")
