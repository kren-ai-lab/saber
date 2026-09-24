"""Classification algorithm registration."""

from __future__ import annotations

from saber._optional import is_dependency_available
from saber.classification.sklearn import register_sklearn_classification_models

__all__ = ["register_sklearn_classification_models"]

if is_dependency_available("xgboost"):
    from saber.classification.xgboost import register_xgboost_classification_models

    __all__.append("register_xgboost_classification_models")

if is_dependency_available("lightgbm"):
    from saber.classification.lightgbm import register_lightgbm_classification_models

    __all__.append("register_lightgbm_classification_models")
