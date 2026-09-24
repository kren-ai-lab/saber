"""saber.classification.search_spaces.

Default hyperparameter search spaces for classification models.
"""

from __future__ import annotations

from saber.core.search_space import SearchSpace

# ============================================================
# Linear Models
# ============================================================

LOGISTIC_REGRESSION = SearchSpace(
    name="logistic_regression",
    parameters={
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs", "liblinear"],
        "penalty": ["l2"],
    },
)

RIDGE_CLASSIFIER = SearchSpace(
    name="ridge_classifier",
    parameters={
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
)

SGD_CLASSIFIER = SearchSpace(
    name="sgd_classifier",
    parameters={
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "loss": ["hinge", "log_loss", "modified_huber"],
        "penalty": ["l2", "l1", "elasticnet"],
    },
)


# ============================================================
# Tree-based Models
# ============================================================

DECISION_TREE = SearchSpace(
    name="decision_tree",
    parameters={
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
)

EXTRA_TREE = SearchSpace(
    name="extra_tree",
    parameters={
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
)

RANDOM_FOREST = SearchSpace(
    name="random_forest",
    parameters={
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
)

EXTRA_TREES = SearchSpace(
    name="extra_trees",
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

GRADIENT_BOOSTING = SearchSpace(
    name="gradient_boosting",
    parameters={
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    },
)

ADABOOST = SearchSpace(
    name="adaboost",
    parameters={
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 1.0],
    },
)

HIST_GRADIENT_BOOSTING = SearchSpace(
    name="hist_gradient_boosting",
    parameters={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 10, None],
        "max_iter": [100, 200, 500],
    },
)

BAGGING = SearchSpace(
    name="bagging",
    parameters={
        "n_estimators": [10, 50, 100, 200],
    },
)


# ============================================================
# SVM
# ============================================================

SVC_SPACE = SearchSpace(
    name="svc",
    parameters={
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
)

LINEAR_SVC = SearchSpace(
    name="linear_svc",
    parameters={
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
)

NU_SVC = SearchSpace(
    name="nu_svc",
    parameters={
        "nu": [0.1, 0.25, 0.5, 0.75],
        "kernel": ["linear", "rbf"],
    },
)


# ============================================================
# Neighbors
# ============================================================

KNN = SearchSpace(
    name="knn",
    parameters={
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
)

RADIUS_NEIGHBORS = SearchSpace(
    name="radius_neighbors",
    parameters={
        "radius": [0.5, 1.0, 2.0, 5.0],
        "weights": ["uniform", "distance"],
    },
)

NEAREST_CENTROID = SearchSpace(
    name="nearest_centroid",
    parameters={},
)


# ============================================================
# Discriminant Analysis
# ============================================================

LDA = SearchSpace(
    name="lda",
    parameters={
        "solver": ["svd", "lsqr", "eigen"],
    },
)

QDA = SearchSpace(
    name="qda",
    parameters={
        "reg_param": [0.0, 0.1, 0.25, 0.5],
    },
)


# ============================================================
# Gaussian Process
# ============================================================

GAUSSIAN_PROCESS = SearchSpace(
    name="gaussian_process",
    parameters={},
)


# ============================================================
# Naive Bayes
# ============================================================

GAUSSIAN_NB = SearchSpace(
    name="gaussian_nb",
    parameters={
        "var_smoothing": [
            1e-12,
            1e-10,
            1e-9,
            1e-8,
        ],
    },
)

BERNOULLI_NB = SearchSpace(
    name="bernoulli_nb",
    parameters={
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
)

CATEGORICAL_NB = SearchSpace(
    name="categorical_nb",
    parameters={
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
)

MULTINOMIAL_NB = SearchSpace(
    name="multinomial_nb",
    parameters={
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
)

COMPLEMENT_NB = SearchSpace(
    name="complement_nb",
    parameters={
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
)


# ============================================================
# XGBoost
# ============================================================

XGB_CLASSIFIER = SearchSpace(
    name="xgb_classifier",
    parameters={
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
)

XGB_RF_CLASSIFIER = SearchSpace(
    name="xgb_rf_classifier",
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

LGBM_CLASSIFIER = SearchSpace(
    name="lgbm_classifier",
    parameters={
        "n_estimators": [100, 200, 500],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
)
