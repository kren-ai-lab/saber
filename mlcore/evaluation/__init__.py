from mlcore.evaluation.classification import (
    CLASSIFICATION_METRICS,
    evaluate_classification,
)

from mlcore.evaluation.regression import (
    REGRESSION_METRICS,
    evaluate_regression,
)

from mlcore.evaluation.registry import (
    METRIC_REGISTRY,
    available_tasks,
    get_metrics,
    has_metric,
)

__all__ = [
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "METRIC_REGISTRY",
    "evaluate_classification",
    "evaluate_regression",
    "available_tasks",
    "get_metrics",
    "has_metric",
]