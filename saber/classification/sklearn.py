"""saber.classification.sklearn_models
====================================

Scikit-learn classification models and registry wiring.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
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
    SVC,
    LinearSVC,
    NuSVC,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    ExtraTreeClassifier,
)

from saber.classification import search_spaces
from saber.core.capabilities import (
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.registry import MODEL_REGISTRY
from saber.core.specs import AlgorithmSpec

_ALIASES: dict[str, tuple[str, ...]] = {
    "dummy_classifier": ("classification_baseline",),
    "logistic_regression": ("logreg", "lr_classifier"),
    "random_forest": ("rf_classifier", "rf_clf"),
    "extra_trees": ("extra_trees_classifier",),
    "svc": ("svm_classifier",),
    "knn": ("knn_classifier",),
    "decision_tree": ("decision_tree_classifier",),
}

_DEFAULT_PARAMS: dict[str, dict[str, object]] = {
    "dummy_classifier": {"strategy": "prior"},
}

_NON_NEGATIVE_X = {
    "categorical_nb",
    "multinomial_nb",
    "complement_nb",
}

_SCALING_RECOMMENDED = {
    "logistic_regression",
    "svc",
    "linear_svc",
    "nu_svc",
    "knn",
    "radius_neighbors",
    "nearest_centroid",
    "ridge_classifier",
    "sgd_classifier",
    "gaussian_process",
}


def _requirements_for(name: str) -> EstimatorRequirements:
    return EstimatorRequirements(
        non_negative_X=name in _NON_NEGATIVE_X,
        scaling=("recommended" if name in _SCALING_RECOMMENDED else "not_required"),
    )


# ============================================================
# Registration
# ============================================================


def register_sklearn_classification_models() -> None:
    """Register all scikit-learn classification models.
    """
    models = [
        ("dummy_classifier", DummyClassifier, ("classification", "baseline", "dummy"), None),
        (
            "logistic_regression",
            LogisticRegression,
            ("classification", "logistic_regression"),
            search_spaces.LOGISTIC_REGRESSION,
        ),
        (
            "random_forest",
            RandomForestClassifier,
            ("classification", "random_forest", "ensemble", "bagging"),
            search_spaces.RANDOM_FOREST,
        ),
        (
            "extra_trees",
            ExtraTreesClassifier,
            ("classification", "extra_trees", "tree", "bagging"),
            search_spaces.EXTRA_TREES,
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier,
            ("classification", "gradient_boosting", "tree", "boosting"),
            search_spaces.GRADIENT_BOOSTING,
        ),
        ("svc", SVC, ("classification", "svc", "svm"), search_spaces.SVC_SPACE),
        ("linear_svc", LinearSVC, ("classification", "svc", "linear_svm"), search_spaces.LINEAR_SVC),
        ("nu_svc", NuSVC, ("classification", "svc", "nu_svc"), search_spaces.NU_SVC),
        ("knn", KNeighborsClassifier, ("classification", "knn", "distance_based"), search_spaces.KNN),
        (
            "radius_neighbors",
            RadiusNeighborsClassifier,
            ("classification", "radius_neighbors", "distance_based"),
            search_spaces.RADIUS_NEIGHBORS,
        ),
        (
            "nearest_centroid",
            NearestCentroid,
            ("classification", "nearest_centroid", "distance_based"),
            search_spaces.NEAREST_CENTROID,
        ),
        (
            "decision_tree",
            DecisionTreeClassifier,
            ("classification", "decision_tree", "tree"),
            search_spaces.DECISION_TREE,
        ),
        (
            "extra_tree",
            ExtraTreeClassifier,
            ("classification", "extra_tree", "tree"),
            search_spaces.EXTRA_TREE,
        ),
        (
            "adaboost",
            AdaBoostClassifier,
            ("classification", "adaboost", "tree", "boosting"),
            search_spaces.ADABOOST,
        ),
        (
            "bagging",
            BaggingClassifier,
            ("classification", "bagging", "tree", "bagging"),
            search_spaces.BAGGING,
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier,
            ("classification", "hist_gradient_boosting", "tree", "boosting"),
            search_spaces.HIST_GRADIENT_BOOSTING,
        ),
        (
            "ridge_classifier",
            RidgeClassifier,
            ("classification", "ridge", "linear"),
            search_spaces.RIDGE_CLASSIFIER,
        ),
        ("sgd_classifier", SGDClassifier, ("classification", "sgd", "linear"), search_spaces.SGD_CLASSIFIER),
        ("lda", LinearDiscriminantAnalysis, ("classification", "lda", "linear"), search_spaces.LDA),
        ("qda", QuadraticDiscriminantAnalysis, ("classification", "qda", "quadratic"), search_spaces.QDA),
        (
            "gaussian_process",
            GaussianProcessClassifier,
            ("classification", "gaussian_process"),
            search_spaces.GAUSSIAN_PROCESS,
        ),
        ("gaussian_nb", GaussianNB, ("classification", "gaussian", "naive_bayes"), search_spaces.GAUSSIAN_NB),
        (
            "bernoulli_nb",
            BernoulliNB,
            ("classification", "bernoulli", "naive_bayes"),
            search_spaces.BERNOULLI_NB,
        ),
        (
            "categorical_nb",
            CategoricalNB,
            ("classification", "categorical", "naive_bayes"),
            search_spaces.CATEGORICAL_NB,
        ),
        (
            "multinomial_nb",
            MultinomialNB,
            ("classification", "multinomial", "naive_bayes"),
            search_spaces.MULTINOMIAL_NB,
        ),
        (
            "complement_nb",
            ComplementNB,
            ("classification", "complement", "naive_bayes"),
            search_spaces.COMPLEMENT_NB,
        ),
    ]

    specs: list[AlgorithmSpec] = []

    for name, model_cls, tags, search_space in models:
        capabilities = infer_estimator_capabilities(
            model_cls,
            native_missing_values=(name == "hist_gradient_boosting"),
        )

        spec = AlgorithmSpec(
            provider="sklearn",
            task="classification",
            name=name,
            tags=tags,
            aliases=_ALIASES.get(name, tuple()),
            default_params=_DEFAULT_PARAMS.get(name, {}),
            estimator_cls=model_cls,
            capabilities=capabilities,
            requirements=_requirements_for(name),
            supports_cv=True,
            search_space=search_space,
        )

        specs.append(spec)

    MODEL_REGISTRY.register_many(specs)


# ============================================================
# Auto-registration
# ============================================================

register_sklearn_classification_models()
