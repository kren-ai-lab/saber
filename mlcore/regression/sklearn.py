"""
mlcore.regression.sklearn
=========================

Scikit-learn based regression algorithms and registry wiring.
"""

from __future__ import annotations

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    BaggingRegressor
)

from sklearn.gaussian_process import (
    GaussianProcessRegressor,
)

from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    ARDRegression,
    GammaRegressor,
    HuberRegressor,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
)

from sklearn.neighbors import (
    KNeighborsRegressor,
    RadiusNeighborsRegressor
)

from sklearn.neural_network import (
    MLPRegressor,
)

from sklearn.svm import (
    LinearSVR,
    NuSVR,
    SVR,
)

from sklearn.tree import (
    DecisionTreeRegressor,
    ExtraTreeRegressor,
)

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec

from mlcore.regression.backend import RegressionBackend
from mlcore.regression.runners import make_regression_runner


def register_sklearn_regression_models() -> None:
    """
    Register all scikit-learn regression models.
    """

    models = [
        (
            "linear_regression",
            LinearRegression,
            ("regression", "linear"),
        ),
        (
            "ridge_regressor",
            Ridge,
            ("regression", "linear", "regularized"),
        ),
        (
            "lasso_regressor",
            Lasso,
            ("regression", "linear", "regularized"),
        ),
        (
            "elastic_net",
            ElasticNet,
            ("regression", "linear", "regularized"),
        ),
        (
            "bayesian_ridge",
            BayesianRidge,
            ("regression", "linear", "bayesian"),
        ),
        (
            "random_forest_regressor",
            RandomForestRegressor,
            ("regression", "tree", "ensemble"),
        ),
        (
            "extra_trees_regressor",
            ExtraTreesRegressor,
            ("regression", "tree", "ensemble"),
        ),
        (
            "gradient_boosting_regressor",
            GradientBoostingRegressor,
            ("regression", "boosting"),
        ),
        (
            "hist_gradient_boosting_regressor",
            HistGradientBoostingRegressor,
            ("regression", "boosting"),
        ),
        (
            "adaboost_regressor",
            AdaBoostRegressor,
            ("regression", "boosting"),
        ),
        (
            "knn_regressor",
            KNeighborsRegressor,
            ("regression", "neighbors"),
        ),
        (
            "svr",
            SVR,
            ("regression", "svm"),
        ),
        (
            "linear_svr",
            LinearSVR,
            ("regression", "svm"),
        ),
        (
            "nu_svr",
            NuSVR,
            ("regression", "svm"),
        ),
        (
            "decision_tree_regressor",
            DecisionTreeRegressor,
            ("regression", "tree"),
        ),
        (
            "extra_tree_regressor",
            ExtraTreeRegressor,
            ("regression", "tree"),
        ),
        (
            "gaussian_process_regressor",
            GaussianProcessRegressor,
            ("regression", "gaussian_process"),
        ),
        (
            "mlp_regressor",
            MLPRegressor,
            ("regression", "neural_network"),
        ),
        (
            "bagging_regressor",
            BaggingRegressor,
            ("regression", "enssemble")
        ), 

        (
            "ard_regression",
            ARDRegression,
            ("regression", "linear method"),
        ),

        (
            "gamma_regression",
            GammaRegressor,
            ("regression", "linear method"),
        ),

        (
            "huber_regression",
            HuberRegressor,
            ("regression", "linear method"),
        ),

        (
            "lars_regressor",
            Lars,
            ("regression", "linear method"),
        ),

        (
            "lasso_lars_regressor",
            LassoLars,
            ("regression", "linear method"),
        ),

        (
            "orthogonal_matching_pursuit",
            OrthogonalMatchingPursuit,
            ("regression", "linear method"),
        ),

        (
            "radius_neighbors_regressor",
            RadiusNeighborsRegressor,
            ("regression", "KNN-based")
        )
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls, tags in models:

        spec = AlgorithmSpec(
            backend="sklearn",
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


register_sklearn_regression_models()