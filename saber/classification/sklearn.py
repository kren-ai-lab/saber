"""saber.classification.sklearn.

Scikit-learn classification model specs.
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
from saber.core.specs import AlgorithmSpec

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
    "knn_classifier",
    "radius_neighbors_classifier",
    "nearest_centroid",
    "ridge_classifier",
    "sgd_classifier",
    "gaussian_process_classifier",
}


def _requirements_for(name: str) -> EstimatorRequirements:
    return EstimatorRequirements(
        non_negative_X=name in _NON_NEGATIVE_X,
        scaling=("recommended" if name in _SCALING_RECOMMENDED else "not_required"),
    )


def _build_specs() -> tuple[AlgorithmSpec, ...]:
    """Build all scikit-learn classification models."""
    models = [
        ("dummy_classifier", DummyClassifier, ("baseline",), None),
        (
            "logistic_regression",
            LogisticRegression,
            ("linear",),
            search_spaces.LOGISTIC_REGRESSION,
        ),
        (
            "random_forest_classifier",
            RandomForestClassifier,
            ("tree", "ensemble", "bagging"),
            search_spaces.RANDOM_FOREST,
        ),
        (
            "extra_trees_classifier",
            ExtraTreesClassifier,
            ("tree", "ensemble", "bagging"),
            search_spaces.EXTRA_TREES,
        ),
        (
            "gradient_boosting_classifier",
            GradientBoostingClassifier,
            ("tree", "ensemble", "boosting"),
            search_spaces.GRADIENT_BOOSTING,
        ),
        ("svc", SVC, ("svm",), search_spaces.SVC_SPACE),
        ("linear_svc", LinearSVC, ("linear", "svm"), search_spaces.LINEAR_SVC),
        ("nu_svc", NuSVC, ("svm",), search_spaces.NU_SVC),
        (
            "knn_classifier",
            KNeighborsClassifier,
            ("neighbors",),
            search_spaces.KNN,
        ),
        (
            "radius_neighbors_classifier",
            RadiusNeighborsClassifier,
            ("neighbors",),
            search_spaces.RADIUS_NEIGHBORS,
        ),
        (
            "nearest_centroid",
            NearestCentroid,
            ("neighbors",),
            search_spaces.NEAREST_CENTROID,
        ),
        (
            "decision_tree_classifier",
            DecisionTreeClassifier,
            ("tree",),
            search_spaces.DECISION_TREE,
        ),
        (
            "extra_tree_classifier",
            ExtraTreeClassifier,
            ("tree",),
            search_spaces.EXTRA_TREE,
        ),
        (
            "adaboost_classifier",
            AdaBoostClassifier,
            ("ensemble", "boosting"),
            search_spaces.ADABOOST,
        ),
        (
            "bagging_classifier",
            BaggingClassifier,
            ("ensemble", "bagging"),
            search_spaces.BAGGING,
        ),
        (
            "hist_gradient_boosting_classifier",
            HistGradientBoostingClassifier,
            ("tree", "ensemble", "boosting"),
            search_spaces.HIST_GRADIENT_BOOSTING,
        ),
        (
            "ridge_classifier",
            RidgeClassifier,
            ("linear",),
            search_spaces.RIDGE_CLASSIFIER,
        ),
        ("sgd_classifier", SGDClassifier, ("linear",), search_spaces.SGD_CLASSIFIER),
        ("lda", LinearDiscriminantAnalysis, ("linear", "discriminant_analysis"), search_spaces.LDA),
        ("qda", QuadraticDiscriminantAnalysis, ("discriminant_analysis",), search_spaces.QDA),
        (
            "gaussian_process_classifier",
            GaussianProcessClassifier,
            ("gaussian_process",),
            search_spaces.GAUSSIAN_PROCESS,
        ),
        ("gaussian_nb", GaussianNB, ("naive_bayes",), search_spaces.GAUSSIAN_NB),
        (
            "bernoulli_nb",
            BernoulliNB,
            ("naive_bayes",),
            search_spaces.BERNOULLI_NB,
        ),
        (
            "categorical_nb",
            CategoricalNB,
            ("naive_bayes",),
            search_spaces.CATEGORICAL_NB,
        ),
        (
            "multinomial_nb",
            MultinomialNB,
            ("naive_bayes",),
            search_spaces.MULTINOMIAL_NB,
        ),
        (
            "complement_nb",
            ComplementNB,
            ("naive_bayes",),
            search_spaces.COMPLEMENT_NB,
        ),
    ]
    return tuple(
        AlgorithmSpec(
            provider="sklearn",
            task="classification",
            name=name,
            estimator_cls=model_cls,
            default_params=_DEFAULT_PARAMS.get(name, {}),
            tags=tags,
            capabilities=infer_estimator_capabilities(
                model_cls,
                native_missing_values=(name == "hist_gradient_boosting_classifier"),
            ),
            requirements=_requirements_for(name),
            search_space=search_space,
        )
        for name, model_cls, tags, search_space in models
    )


SPECS: tuple[AlgorithmSpec, ...] = _build_specs()
