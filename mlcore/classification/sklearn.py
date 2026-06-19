"""
mlcore.classification.sklearn_models
====================================

Scikit-learn classification models and registry wiring.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)

from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)

from sklearn.svm import (
    LinearSVC,
    NuSVC,
    SVC,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    ExtraTreeClassifier,
)

from mlcore.classification.backend import ClassificationBackend
from mlcore.classification.runners import make_classifier_runner

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.specs import AlgorithmSpec

from mlcore.classification import search_spaces

# ============================================================
# Registration
# ============================================================

def register_sklearn_classification_models() -> None:
    """
    Register all scikit-learn classification models.
    """

    models = [
        ("logistic_regression", LogisticRegression, ("classification", "logistic regression"), search_spaces.LOGISTIC_REGRESSION),
        ("random_forest", RandomForestClassifier, ("classification", "random forest", "ensseemble", "bagging"), search_spaces.RANDOM_FOREST),
        ("extra_trees", ExtraTreesClassifier, ("classification", "extra trees", "tree", "bagging"), search_spaces.EXTRA_TREES),
        ("gradient_boosting", GradientBoostingClassifier, ("classification", "gradient boosting", "tree", "boosting"), search_spaces.GRADIENT_BOOSTING),
        ("svc", SVC, ("classification", "svc", "SVM"), search_spaces.SVC_SPACE),
        ("linear_svc", LinearSVC, ("classification", "svc", "Linear SVM"), search_spaces.LINEAR_SVC),
        ("nu_svc", NuSVC, ("classification", "svc", "Nu SVC"), search_spaces.NU_SVC),
        ("knn", KNeighborsClassifier, ("classification", "KNN", "distance-based"), search_spaces.KNN),
        ("radius_neighbors", RadiusNeighborsClassifier, ("classification", "Radius Neighbors", "distance-based"), search_spaces.RADIUS_NEIGHBORS),
        ("nearest_centroid", NearestCentroid, ("classification", "Nearest centroid", "distance-based"), search_spaces.NEAREST_CENTROID),
        ("decision_tree", DecisionTreeClassifier, ("classification", "decision tree", "tree"), search_spaces.DECISION_TREE),
        ("extra_tree", ExtraTreeClassifier, ("classification", "extra-tree", "tree"), search_spaces.EXTRA_TREE),
        ("adaboost", AdaBoostClassifier, ("classification", "adaboosting", "tree", "boosting"), search_spaces.ADABOOST),
        ("bagging", BaggingClassifier, ("classification", "bagging", "tree", "bagging-based"), search_spaces.BAGGING),
        ("hist_gradient_boosting", HistGradientBoostingClassifier, ("classification", "hist-gradient", "tree", "boosting"), search_spaces.HIST_GRADIENT_BOOSTING),
        ("ridge_classifier", RidgeClassifier, ("classification", "ridge", "linear"), search_spaces.RIDGE_CLASSIFIER),
        ("sgd_classifier", SGDClassifier, ("classification", "SGD", "linear"), search_spaces.SGD_CLASSIFIER),
        ("lda", LinearDiscriminantAnalysis, ("classification", "LDA", "linear-based"), search_spaces.LDA),
        ("qda", QuadraticDiscriminantAnalysis, ("classification", "QDA", "quadratic-based"), search_spaces.QDA),
        ("gaussian_process", GaussianProcessClassifier, ("classification", "Gaussian Process"), search_spaces.GAUSSIAN_PROCESS),
        ("gaussian_nb", GaussianNB, ("classification", "Gaussian", "naive bayes"), search_spaces.GAUSSIAN_NB),
        ("bernoulli_nb", BernoulliNB, ("classification", "Bernoulli", "naive bayes"), search_spaces.BERNOULLI_NB),
        ("categorical_nb", CategoricalNB, ("classification", "Categorical", "naive bayes"), search_spaces.CATEGORICAL_NB),
        ("multinomial_nb", MultinomialNB, ("classification", "Multinomial", "naive bayes"), search_spaces.MULTINOMIAL_NB),
        ("complement_nb", ComplementNB, ("classification", "Complement", "naive bayes"), search_spaces.COMPLEMENT_NB),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls, tags, search_space in models:

        try:
            supports_proba = hasattr(
                model_cls(),
                "predict_proba",
            )

        except Exception:
            supports_proba = False

        spec = AlgorithmSpec(
            backend="sklearn",
            task="classification",
            name=name,
            tags=tags,
            estimator_cls=model_cls,
            runner=make_classifier_runner(model_cls),
            backend_cls=ClassificationBackend,
            supports_proba=supports_proba,
            search_space=search_space
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_sklearn_classification_models()