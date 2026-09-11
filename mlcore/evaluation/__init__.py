from mlcore.evaluation.classification import (
    CLASSIFICATION_METRICS,
    evaluate_binary_classification,
    evaluate_classification,
    evaluate_multiclass_classification,
)
from mlcore.evaluation.evaluator import evaluate_prediction
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
from mlcore.evaluation.results import EvaluationResult

__all__ = [
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "METRIC_REGISTRY",
    "EvaluationResult",
    "evaluate_binary_classification",
    "evaluate_classification",
    "evaluate_multiclass_classification",
    "evaluate_prediction",
    "evaluate_regression",
    "available_tasks",
    "get_metrics",
    "has_metric",
]
