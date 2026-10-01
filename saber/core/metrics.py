"""Canonical metric specifications used by optimization and evaluation layers.

The central contract separates a metric's natural interpretation from the
score representation used by scikit-learn during optimization. Loss metrics
such as MAE are naturally minimized, while scikit-learn scorer objects negate
them so every search backend can maximize a common objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from saber.exceptions import (
    MetricIncompatibleError,
    MetricNotFoundError,
    ValidationContractError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from saber.core.task import TaskType

ProblemType = Literal["binary", "multiclass", "regression"]
ResponseMethod = str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Declarative metric contract.

    Parameters
    ----------
    name
        Canonical metric identifier.
    task
        Supported supervised task.
    score_func
        Metric callable consumed by ``sklearn.metrics.make_scorer``.
    greater_is_better
        Natural direction of the metric. Loss metrics set this to ``False``.
    response_method
        Estimator response consumed by the scorer.
    scorer_kwargs
        Extra keyword arguments forwarded to the metric callable.
    problem_types
        Classification regimes supported by the metric. Regression metrics
        use ``("regression",)``.
    positive_class_aware
        Whether a binary scorer must be computed for an explicit positive
        class (as evaluation does) instead of by class order or averaging.

    """

    name: str
    task: TaskType
    # sklearn metric functions return a mix of float/np.floating/ndarray depending
    # on overload; callers normalize via to_natural_score()/float(), so the wider
    # return type is accurate rather than a workaround.
    score_func: Callable[..., Any]
    greater_is_better: bool = True
    response_method: ResponseMethod = "predict"
    scorer_kwargs: dict[str, Any] = field(default_factory=dict)
    problem_types: tuple[ProblemType, ...] = field(default_factory=tuple)
    positive_class_aware: bool = False

    def make_scorer(self, *, positive_class: Any | None = None) -> Any:
        """Build a scikit-learn compatible scorer.

        ``positive_class`` is only meaningful for binary targets: positive-class
        aware metrics are then scored for that class, matching evaluation,
        rather than averaged across classes.
        """
        kwargs = dict(self.scorer_kwargs)
        if positive_class is not None and self.positive_class_aware:
            kwargs.pop("average", None)
            kwargs["pos_label"] = positive_class
        return make_scorer(
            self.score_func,
            response_method=self.response_method,
            greater_is_better=self.greater_is_better,
            **kwargs,
        )

    @property
    def is_loss(self) -> bool:
        """Whether lower natural values are better."""
        return not self.greater_is_better

    def to_natural_score(self, score: float) -> float:
        """Convert an optimization score into its natural user-facing value."""
        value = float(score)
        if self.is_loss:
            return -value
        return value

    def validate_task(self, task: str) -> None:
        """Validate task compatibility."""
        if task != self.task:
            raise MetricIncompatibleError(f"Metric '{self.name}' supports task '{self.task}', not '{task}'.")

    def validate_problem_type(self, problem_type: ProblemType) -> None:
        """Validate binary/multiclass/regression compatibility."""
        if self.problem_types and problem_type not in self.problem_types:
            raise MetricIncompatibleError(
                f"Metric '{self.name}' does not support problem type '{problem_type}'. "
                f"Supported: {', '.join(self.problem_types)}."
            )


def _roc_auc_for_positive_class(y_true: Any, y_score: Any, *, pos_label: Any | None = None) -> float:
    """ROC AUC where ``y_score`` ranks ``pos_label`` (sklearn orients the response to it)."""
    if pos_label is None:
        return float(roc_auc_score(y_true, y_score))
    return float(roc_auc_score(np.asarray(y_true) == pos_label, y_score))


_clf = partial(MetricSpec, task="classification", problem_types=("binary", "multiclass"))
_reg = partial(MetricSpec, task="regression", problem_types=("regression",))
_AVERAGED = {"average": "weighted", "zero_division": 0}

_CLASSIFICATION_SPECS = (
    _clf(name="accuracy", score_func=accuracy_score),
    _clf(name="balanced_accuracy", score_func=balanced_accuracy_score),
    _clf(name="precision", score_func=precision_score, scorer_kwargs=_AVERAGED, positive_class_aware=True),
    _clf(name="recall", score_func=recall_score, scorer_kwargs=_AVERAGED, positive_class_aware=True),
    _clf(name="f1", score_func=f1_score, scorer_kwargs=_AVERAGED, positive_class_aware=True),
    _clf(name="mcc", score_func=matthews_corrcoef),
    _clf(
        name="roc_auc",
        score_func=_roc_auc_for_positive_class,
        response_method=("decision_function", "predict_proba"),
        problem_types=("binary",),
        positive_class_aware=True,
    ),
)

_REGRESSION_SPECS = (
    _reg(name="mae", score_func=mean_absolute_error, greater_is_better=False),
    _reg(name="mse", score_func=mean_squared_error, greater_is_better=False),
    _reg(name="rmse", score_func=root_mean_squared_error, greater_is_better=False),
    _reg(name="median_ae", score_func=median_absolute_error, greater_is_better=False),
    _reg(name="r2", score_func=r2_score),
    _reg(name="explained_variance", score_func=explained_variance_score),
)

METRIC_SPECS: dict[str, MetricSpec] = {
    spec.name: spec for spec in (*_CLASSIFICATION_SPECS, *_REGRESSION_SPECS)
}


def get_metric_spec(name: str) -> MetricSpec:
    """Retrieve a metric specification by canonical name."""
    try:
        return METRIC_SPECS[name]
    except KeyError as exc:
        raise MetricNotFoundError(name) from exc


def infer_problem_type(*, task: str, y: Sequence[Any] | np.ndarray | None = None) -> ProblemType:
    """Infer the supervised problem regime needed by metric validation."""
    if task == "regression":
        return "regression"

    if task != "classification":
        raise ValueError(f"Unknown task: {task}")

    if y is None:
        # Problem-type validation is deferred when targets are not available.
        return "binary"

    # A single-class target is invalid, but dataset-level validation belongs
    # to a later phase; "binary" keeps metric validation on its own contract.
    return "multiclass" if np.unique(np.asarray(y)).size > 2 else "binary"


def resolve_positive_class(
    *, task: str, y: Sequence[Any] | np.ndarray, positive_class: Any | None = None
) -> Any:
    """Return the binary positive class used for scoring, or ``None`` when not binary.

    The default is the last class in sorted order, the same rule
    ``PredictionResult`` applies to a fitted estimator's ``classes_``.
    """
    classes = np.unique(np.asarray(y)) if task == "classification" else np.asarray([])
    if classes.size != 2:
        if positive_class is not None:
            raise ValidationContractError("positive_class is only valid for binary classification.")
        return None
    if positive_class is None:
        return classes[-1].item()
    if positive_class not in classes.tolist():
        raise ValidationContractError(
            f"positive_class {positive_class!r} is not one of the target classes {classes.tolist()}."
        )
    return positive_class


def validate_metric(
    name: str,
    *,
    task: str,
    y: Sequence[Any] | np.ndarray | None = None,
) -> MetricSpec:
    """Retrieve and validate a metric against task and target regime."""
    spec = get_metric_spec(name)
    spec.validate_task(task)

    if y is not None:
        spec.validate_problem_type(
            infer_problem_type(task=task, y=y),
        )

    return spec
