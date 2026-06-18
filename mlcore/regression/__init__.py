from mlcore.regression.lightgbm import (
    register_lightgbm_regression_models,
)

from mlcore.regression.sklearn import (
    register_sklearn_regression_models,
)

from mlcore.regression.xgboost import (
    register_xgboost_regression_models,
)

__all__ = [
    "register_sklearn_regression_models",
    "register_xgboost_regression_models",
    "register_lightgbm_regression_models",
]