"""saber.regression.search_spaces.

Default hyperparameter search spaces for regression models.
"""

from __future__ import annotations

from saber.core.search_space import SearchSpace

# ============================================================
# Linear Models
# ============================================================

LINEAR_REGRESSION = SearchSpace(
    name="linear_regression",
    parameters={},
)

RIDGE_REGRESSOR = SearchSpace(
    name="ridge_regressor",
    parameters={
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
)

LASSO_REGRESSOR = SearchSpace(
    name="lasso",
    parameters={
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    },
)

ELASTIC_NET = SearchSpace(
    name="elastic_net",
    parameters={
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
)

BAYESIAN_RIDGE = SearchSpace(
    name="bayesian_ridge",
    parameters={
        "alpha_1": [1e-7, 1e-6, 1e-5],
        "alpha_2": [1e-7, 1e-6, 1e-5],
        "lambda_1": [1e-7, 1e-6, 1e-5],
        "lambda_2": [1e-7, 1e-6, 1e-5],
    },
)

ARD_REGRESSION = SearchSpace(
    name="ard_regression",
    parameters={
        "alpha_1": [1e-7, 1e-6, 1e-5],
        "alpha_2": [1e-7, 1e-6, 1e-5],
        "lambda_1": [1e-7, 1e-6, 1e-5],
        "lambda_2": [1e-7, 1e-6, 1e-5],
    },
)

GAMMA_REGRESSION = SearchSpace(
    name="gamma_regressor",
    parameters={
        "alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
    },
)

HUBER_REGRESSION = SearchSpace(
    name="huber_regressor",
    parameters={
        "epsilon": [1.1, 1.2, 1.35, 1.5, 2.0],
        "alpha": [0.0001, 0.001, 0.01],
    },
)

LARS_REGRESSOR = SearchSpace(
    name="lars",
    parameters={
        "n_nonzero_coefs": [100, 250, 500],
    },
)

LASSO_LARS_REGRESSOR = SearchSpace(
    name="lasso_lars",
    parameters={
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    },
)

ORTHOGONAL_MATCHING_PURSUIT = SearchSpace(
    name="orthogonal_matching_pursuit",
    parameters={
        "n_nonzero_coefs": [5, 10, 20, 50],
    },
)


# ============================================================
# Tree Models
# ============================================================

DECISION_TREE_REGRESSOR = SearchSpace(
    name="decision_tree_regressor",
    parameters={
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
)

EXTRA_TREE_REGRESSOR = SearchSpace(
    name="extra_tree_regressor",
    parameters={
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
)

RANDOM_FOREST_REGRESSOR = SearchSpace(
    name="random_forest_regressor",
    parameters={
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
)

EXTRA_TREES_REGRESSOR = SearchSpace(
    name="extra_trees_regressor",
    parameters={
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
)


# ============================================================
# Boosting
# ============================================================

GRADIENT_BOOSTING_REGRESSOR = SearchSpace(
    name="gradient_boosting_regressor",
    parameters={
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    },
)

HIST_GRADIENT_BOOSTING_REGRESSOR = SearchSpace(
    name="hist_gradient_boosting_regressor",
    parameters={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 10, None],
        "max_iter": [100, 200, 500],
    },
)

ADABOOST_REGRESSOR = SearchSpace(
    name="adaboost_regressor",
    parameters={
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 1.0],
    },
)

BAGGING_REGRESSOR = SearchSpace(
    name="bagging_regressor",
    parameters={
        "n_estimators": [10, 50, 100, 200],
    },
)


# ============================================================
# Neighbors
# ============================================================

KNN_REGRESSOR = SearchSpace(
    name="knn_regressor",
    parameters={
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
)

RADIUS_NEIGHBORS_REGRESSOR = SearchSpace(
    name="radius_neighbors_regressor",
    parameters={
        "radius": [0.5, 1.0, 2.0, 5.0],
        "weights": ["uniform", "distance"],
    },
)


# ============================================================
# SVM
# ============================================================

SVR_SPACE = SearchSpace(
    name="svr",
    parameters={
        "C": [0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.1, 0.5],
        "kernel": ["linear", "rbf"],
    },
)

LINEAR_SVR = SearchSpace(
    name="linear_svr",
    parameters={
        "C": [0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.1, 0.5],
    },
)

NU_SVR = SearchSpace(
    name="nu_svr",
    parameters={
        "C": [0.1, 1.0, 10.0],
        "nu": [0.25, 0.5, 0.75],
    },
)


# ============================================================
# Gaussian Process
# ============================================================

GAUSSIAN_PROCESS_REGRESSOR = SearchSpace(
    name="gaussian_process_regressor",
    parameters={},
)


# ============================================================
# XGBoost
# ============================================================

XGB_REGRESSOR = SearchSpace(
    name="xgb_regressor",
    parameters={
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
)

XGBRF_REGRESSOR = SearchSpace(
    name="xgb_rf_regressor",
    parameters={
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bynode": [0.7, 0.8, 1.0],
    },
)


# ============================================================
# LightGBM
# ============================================================

LGBM_REGRESSOR = SearchSpace(
    name="lgbm_regressor",
    parameters={
        "n_estimators": [100, 200, 500],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
)
