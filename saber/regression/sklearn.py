"""
saber.regression.sklearn
=========================

Scikit-learn based regression algorithms and registry wiring.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor

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


from sklearn.svm import (
    LinearSVR,
    NuSVR,
    SVR,
)

from sklearn.tree import (
    DecisionTreeRegressor,
    ExtraTreeRegressor,
)

from saber.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.registry import MODEL_REGISTRY
from saber.core.specs import AlgorithmSpec

from saber.regression import search_spaces


_ALIASES: dict[str, tuple[str, ...]] = {
    "dummy_regressor": ("regression_baseline",),
    "linear_regression": ("ols",),
    "ridge_regressor": ("ridge",),
    "random_forest_regressor": ("rf_regressor",),
    "knn_regressor": ("knn_regression",),
    "svr": ("svm_regressor",),
    "decision_tree_regressor": ("decision_tree_regression",),
}

_DEFAULT_PARAMS: dict[str, dict[str, object]] = {
    "dummy_regressor": {"strategy": "mean"},
}

_SCALING_RECOMMENDED = {
    "ridge_regressor",
    "lasso_regressor",
    "elastic_net",
    "bayesian_ridge",
    "knn_regressor",
    "svr",
    "linear_svr",
    "nu_svr",
    "gaussian_process_regressor",
    "ard_regression",
    "huber_regression",
    "lars_regressor",
    "lasso_lars_regressor",
    "orthogonal_matching_pursuit",
    "radius_neighbors_regressor",
}


def _requirements_for(name: str) -> EstimatorRequirements:
    return EstimatorRequirements(
        positive_y=(name == "gamma_regression"),
        scaling=(
            "recommended"
            if name in _SCALING_RECOMMENDED
            else "not_required"
        ),
    )


def register_sklearn_regression_models() -> None:
    """
    Register all scikit-learn regression models.
    """

    models = [
        (
            "dummy_regressor",
            DummyRegressor,
            ("regression", "baseline", "dummy"),
            None,
        ),
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
            "bagging_regressor",
            BaggingRegressor,
            ("regression", "ensemble"),
            search_spaces.BAGGING_REGRESSOR,
        ), 

        (
            "ard_regression",
            ARDRegression,
            ("regression", "linear"),
            search_spaces.ARD_REGRESSION,
        ),

        (
            "gamma_regression",
            GammaRegressor,
            ("regression", "linear"),
            search_spaces.GAMMA_REGRESSION,
        ),

        (
            "huber_regression",
            HuberRegressor,
            ("regression", "linear"),
            search_spaces.HUBER_REGRESSION,
        ),

        (
            "lars_regressor",
            Lars,
            ("regression", "linear"),
            search_spaces.LARS_REGRESSOR,
        ),

        (
            "lasso_lars_regressor",
            LassoLars,
            ("regression", "linear"),
            search_spaces.LASSO_LARS_REGRESSOR,
        ),

        (
            "orthogonal_matching_pursuit",
            OrthogonalMatchingPursuit,
            ("regression", "linear"),
            search_spaces.ORTHOGONAL_MATCHING_PURSUIT,
        ),

        (
            "radius_neighbors_regressor",
            RadiusNeighborsRegressor,
            ("regression", "knn"),
            search_spaces.RADIUS_NEIGHBORS_REGRESSOR,
        )
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_rgx, tags, search_space in models:

        capabilities = infer_estimator_capabilities(
            model_rgx,
            native_missing_values=(name == "hist_gradient_boosting_regressor"),
        )

        spec = AlgorithmSpec(
            provider="sklearn",
            task="regression",
            name=name,
            estimator_cls=model_rgx,
            aliases=_ALIASES.get(name, tuple()),
            default_params=_DEFAULT_PARAMS.get(name, {}),
            tags=tags,
            capabilities=capabilities,
            requirements=_requirements_for(name),
            supports_cv=True,
            search_space=search_space,
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(
        specs,
    )


register_sklearn_regression_models()
