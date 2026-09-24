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
    MetricNotFoundError,
    MetricProblemTypeError,
    MetricTaskMismatchError,
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
    description
        Human-readable summary.

    """

    name: str
    task: TaskType
    score_func: Callable[..., float]
    greater_is_better: bool = True
    response_method: ResponseMethod = "predict"
    scorer_kwargs: dict[str, Any] = field(default_factory=dict)
    problem_types: tuple[ProblemType, ...] = field(default_factory=tuple)
    description: str | None = None

    def make_scorer(self) -> Any:
        """Build a scikit-learn compatible scorer."""
        return make_scorer(
            self.score_func,
            response_method=self.response_method,
            greater_is_better=self.greater_is_better,
            **dict(self.scorer_kwargs),
        )

    @property
    def is_loss(self) -> bool:
        """Whether lower natural values are better."""
        return not self.greater_is_better

    @property
    def optimization_direction(self) -> Literal["maximize"]:
        """Optimization direction for scorer outputs.

        Scikit-learn scorer objects negate loss metrics, therefore every
        scorer exposed by saber is optimized by maximization.
        """
        return "maximize"

    def to_natural_score(self, score: float) -> float:
        """Convert an optimization score into its natural user-facing value."""
        value = float(score)
        if self.is_loss:
            return -value
        return value

    def validate_task(self, task: str) -> None:
        """Validate task compatibility."""
        if task != self.task:
            raise MetricTaskMismatchError(
                metric=self.name,
                metric_task=self.task,
                requested_task=task,
            )

    def validate_problem_type(self, problem_type: ProblemType) -> None:
        """Validate binary/multiclass/regression compatibility."""
        if self.problem_types and problem_type not in self.problem_types:
            raise MetricProblemTypeError(
                metric=self.name,
                problem_type=problem_type,
                supported=self.problem_types,
            )


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
        description="Weighted precision across classes.",
    ),
    MetricSpec(
        name="recall",
        task="classification",
        score_func=recall_score,
        scorer_kwargs={"average": "weighted", "zero_division": 0},
        problem_types=("binary", "multiclass"),
        description="Weighted recall across classes.",
    ),
    MetricSpec(
        name="f1",
        task="classification",
        score_func=f1_score,
        scorer_kwargs={"average": "weighted", "zero_division": 0},
        problem_types=("binary", "multiclass"),
        description="Weighted harmonic mean of precision and recall.",
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
        score_func=roc_auc_score,
        response_method=("decision_function", "predict_proba"),
        problem_types=("binary",),
        description="Area under the ROC curve for binary classification.",
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
