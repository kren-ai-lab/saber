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
from mlcore.regression import search_spaces

def register_sklearn_regression_models() -> None:
    """
    Register all scikit-learn regression models.
    """

    models = [
        (
            "linear_regression",
            LinearRegression,
            ("regression", "linear"),
            search_spaces.LINEAR_REGRESSION,
        ),
        (
            "ridge_regressor",
            Ridge,
            ("regression", "linear", "regularized"),
            search_spaces.RIDGE_REGRESSOR,
        ),
        (
            "lasso_regressor",
            Lasso,
            ("regression", "linear", "regularized"),
            search_spaces.LASSO_REGRESSOR,
        ),
        (
            "elastic_net",
            ElasticNet,
            ("regression", "linear", "regularized"),
            search_spaces.ELASTIC_NET,
        ),
        (
            "bayesian_ridge",
            BayesianRidge,
            ("regression", "linear", "bayesian"),
            search_spaces.BAYESIAN_RIDGE,
        ),
        (
            "random_forest_regressor",
            RandomForestRegressor,
            ("regression", "tree", "ensemble"),
            search_spaces.RANDOM_FOREST_REGRESSOR,
        ),
        (
            "extra_trees_regressor",
            ExtraTreesRegressor,
            ("regression", "tree", "ensemble"),
            search_spaces.EXTRA_TREES_REGRESSOR,
        ),
        (
            "gradient_boosting_regressor",
            GradientBoostingRegressor,
            ("regression", "boosting"),
            search_spaces.GRADIENT_BOOSTING_REGRESSOR,
        ),
        (
            "hist_gradient_boosting_regressor",
            HistGradientBoostingRegressor,
            ("regression", "boosting"),
            search_spaces.HIST_GRADIENT_BOOSTING_REGRESSOR,
        ),
        (
            "adaboost_regressor",
            AdaBoostRegressor,
            ("regression", "boosting"),
            search_spaces.ADABOOST_REGRESSOR,
        ),
        (
            "knn_regressor",
            KNeighborsRegressor,
            ("regression", "neighbors"),
            search_spaces.KNN_REGRESSOR,
        ),
        (
            "svr",
            SVR,
            ("regression", "svm"),
            search_spaces.SVR_SPACE,
        ),
        (
            "linear_svr",
            LinearSVR,
            ("regression", "svm"),
            search_spaces.LINEAR_SVR,
        ),
        (
            "nu_svr",
            NuSVR,
            ("regression", "svm"),
            search_spaces.NU_SVR,
        ),
        (
            "decision_tree_regressor",
            DecisionTreeRegressor,
            ("regression", "tree"),
            search_spaces.DECISION_TREE_REGRESSOR,
        ),
        (
            "extra_tree_regressor",
            ExtraTreeRegressor,
            ("regression", "tree"),
            search_spaces.EXTRA_TREE_REGRESSOR,
        ),
        (
            "gaussian_process_regressor",
            GaussianProcessRegressor,
            ("regression", "gaussian_process"),
            search_spaces.GAUSSIAN_PROCESS_REGRESSOR,
        ),
        (
            "mlp_regressor",
            MLPRegressor,
            ("regression", "neural_network"),
            search_spaces.MLP_REGRESSOR,
        ),
        (
            "bagging_regressor",
            BaggingRegressor,
            ("regression", "enssemble"),
            search_spaces.BAGGING_REGRESSOR,
        ), 

        (
            "ard_regression",
            ARDRegression,
            ("regression", "linear method"),
            search_spaces.ARD_REGRESSION,
        ),

        (
            "gamma_regression",
            GammaRegressor,
            ("regression", "linear method"),
            search_spaces.GAMMA_REGRESSION,
        ),

        (
            "huber_regression",
            HuberRegressor,
            ("regression", "linear method"),
            search_spaces.HUBER_REGRESSION,
        ),

        (
            "lars_regressor",
            Lars,
            ("regression", "linear method"),
            search_spaces.LARS_REGRESSOR,
        ),

        (
            "lasso_lars_regressor",
            LassoLars,
            ("regression", "linear method"),
            search_spaces.LASSO_LARS_REGRESSOR,
        ),

        (
            "orthogonal_matching_pursuit",
            OrthogonalMatchingPursuit,
            ("regression", "linear method"),
            search_spaces.ORTHOGONAL_MATCHING_PURSUIT,
        ),

        (
            "radius_neighbors_regressor",
            RadiusNeighborsRegressor,
            ("regression", "KNN-based"),
            search_spaces.RADIUS_NEIGHBORS_REGRESSOR,
        )
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls, tags, search_space in models:

        spec = AlgorithmSpec(
            backend="sklearn",
            task="regression",
            name=name,
            runner=make_regression_runner(
                model_cls,
            ),
            backend_cls=RegressionBackend,
            tags=tags,
            search_space=search_space
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(
        specs,
    )


register_sklearn_regression_models()