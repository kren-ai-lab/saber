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


# ============================================================
# Registration
# ============================================================

def register_sklearn_classification_models() -> None:
    """
    Register all scikit-learn classification models.
    """

    models = [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("extra_trees", ExtraTreesClassifier),
        ("gradient_boosting", GradientBoostingClassifier),
        ("svc", SVC),
        ("linear_svc", LinearSVC),
        ("nu_svc", NuSVC),
        ("knn", KNeighborsClassifier),
        ("radius_neighbors", RadiusNeighborsClassifier),
        ("nearest_centroid", NearestCentroid),
        ("decision_tree", DecisionTreeClassifier),
        ("extra_tree", ExtraTreeClassifier),
        ("adaboost", AdaBoostClassifier),
        ("bagging", BaggingClassifier),
        ("hist_gradient_boosting", HistGradientBoostingClassifier),
        ("ridge_classifier", RidgeClassifier),
        ("sgd_classifier", SGDClassifier),
        ("lda", LinearDiscriminantAnalysis),
        ("qda", QuadraticDiscriminantAnalysis),
        ("gaussian_process", GaussianProcessClassifier),
        ("gaussian_nb", GaussianNB),
        ("bernoulli_nb", BernoulliNB),
        ("categorical_nb", CategoricalNB),
        ("multinomial_nb", MultinomialNB),
        ("complement_nb", ComplementNB),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls in models:

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
            runner=make_classifier_runner(model_cls),
            backend_cls=ClassificationBackend,
            tags=("classification", "sklearn"),
            supports_proba=supports_proba,
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_sklearn_classification_models()