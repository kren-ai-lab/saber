"""Canonical metric specifications used by optimization and evaluation layers.

The central contract separates a metric's natural interpretation from the
score representation used by scikit-learn during optimization. Loss metrics
such as MAE are naturally minimized, while scikit-learn scorer objects negate
them so every search backend can maximize a common objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    description
        Human-readable summary.

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
    description: str | None = None
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


_CLASSIFICATION_SPECS = (
    MetricSpec(
        name="accuracy",
        task="classification",
        score_func=accuracy_score,
        problem_types=("binary", "multiclass"),
        description="Fraction of correctly classified samples.",
    ),
    MetricSpec(
        name="balanced_accuracy",
        task="classification",
        score_func=balanced_accuracy_score,
        problem_types=("binary", "multiclass"),
        description="Mean recall across classes.",
    ),
    MetricSpec(
        name="precision",
        task="classification",
        score_func=precision_score,
        scorer_kwargs={"average": "weighted", "zero_division": 0},
        problem_types=("binary", "multiclass"),
        description="Positive-class precision (binary); weighted across classes (multiclass).",
        positive_class_aware=True,
    ),
    MetricSpec(
        name="recall",
        task="classification",
        score_func=recall_score,
        scorer_kwargs={"average": "weighted", "zero_division": 0},
        problem_types=("binary", "multiclass"),
        description="Positive-class recall (binary); weighted across classes (multiclass).",
        positive_class_aware=True,
    ),
    MetricSpec(
        name="f1",
        task="classification",
        score_func=f1_score,
        scorer_kwargs={"average": "weighted", "zero_division": 0},
        problem_types=("binary", "multiclass"),
        description="Positive-class F1 (binary); weighted across classes (multiclass).",
        positive_class_aware=True,
    ),
    MetricSpec(
        name="mcc",
        task="classification",
        score_func=matthews_corrcoef,
        problem_types=("binary", "multiclass"),
        description="Matthews correlation coefficient.",
    ),
    MetricSpec(
        name="roc_auc",
        task="classification",
        score_func=_roc_auc_for_positive_class,
        response_method=("decision_function", "predict_proba"),
        problem_types=("binary",),
        description="Area under the ROC curve for binary classification.",
        positive_class_aware=True,
    ),
)

_REGRESSION_SPECS = (
    MetricSpec(
        name="mae",
        task="regression",
        score_func=mean_absolute_error,
        greater_is_better=False,
        problem_types=("regression",),
        description="Mean absolute error.",
    ),
    MetricSpec(
        name="mse",
        task="regression",
        score_func=mean_squared_error,
        greater_is_better=False,
        problem_types=("regression",),
        description="Mean squared error.",
    ),
    MetricSpec(
        name="rmse",
        task="regression",
        score_func=root_mean_squared_error,
        greater_is_better=False,
        problem_types=("regression",),
        description="Root mean squared error.",
    ),
    MetricSpec(
        name="median_ae",
        task="regression",
        score_func=median_absolute_error,
        greater_is_better=False,
        problem_types=("regression",),
        description="Median absolute error.",
    ),
    MetricSpec(
        name="r2",
        task="regression",
        score_func=r2_score,
        problem_types=("regression",),
        description="Coefficient of determination.",
    ),
    MetricSpec(
        name="explained_variance",
        task="regression",
        score_func=explained_variance_score,
        problem_types=("regression",),
        description="Explained variance regression score.",
    ),
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


def list_metric_specs(*, task: str | None = None) -> list[MetricSpec]:
    """List metric specifications, optionally filtered by task."""
    specs = list(METRIC_SPECS.values())
    if task is not None:
        specs = [spec for spec in specs if spec.task == task]
    return sorted(specs, key=lambda spec: spec.name)


def infer_problem_type(*, task: str, y: Sequence[Any] | np.ndarray | None = None) -> ProblemType:
    """Infer the supervised problem regime needed by metric validation."""
    if task == "regression":
        return "regression"

    if task != "classification":
        raise ValueError(f"Unknown task: {task}")

    if y is None:
        # Problem-type validation is deferred when targets are not available.
        return "binary"

    n_classes = np.unique(np.asarray(y)).size
    if n_classes == 2:
        return "binary"
    if n_classes > 2:
        return "multiclass"

    # A single-class target is invalid for supervised classification, but
    # dataset-level validation belongs to a later phase. Returning binary here
    # keeps metric validation focused on its own contract.
    return "binary"


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
