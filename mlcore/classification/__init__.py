"""Classification algorithm registration."""

from __future__ import annotations

from mlcore._optional import is_dependency_available
from mlcore.classification.sklearn import register_sklearn_classification_models

__all__ = ["register_sklearn_classification_models"]

if is_dependency_available("xgboost"):
    from mlcore.classification.xgboost import register_xgboost_classification_models

    __all__.append("register_xgboost_classification_models")

if is_dependency_available("lightgbm"):
    from mlcore.classification.lightgbm import register_lightgbm_classification_models

    __all__.append("register_lightgbm_classification_models")
