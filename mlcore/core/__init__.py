"""Stable core contracts for mlcore."""

from mlcore.core.capabilities import (
    EstimatorCapabilities,
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from mlcore.core.estimator import EstimatorFactory
from mlcore.core.metrics import (
    METRIC_SPECS,
    MetricSpec,
    get_metric_spec,
    list_metric_specs,
    validate_metric,
)
from mlcore.core.prediction import PredictionResult
from mlcore.core.registry import AlgorithmRegistry, MODEL_REGISTRY
from mlcore.core.results import TrainResult
from mlcore.core.search_space import Categorical, Float, Integer, LogFloat, SearchSpace
from mlcore.core.specs import AlgorithmSpec
from mlcore.core.task import TaskType

__all__ = [
    "AlgorithmRegistry",
    "AlgorithmSpec",
    "Categorical",
    "EstimatorCapabilities",
    "EstimatorFactory",
    "EstimatorRequirements",
    "Float",
    "Integer",
    "LogFloat",
    "METRIC_SPECS",
    "MODEL_REGISTRY",
    "MetricSpec",
    "PredictionResult",
    "SearchSpace",
    "TaskType",
    "TrainResult",
    "get_metric_spec",
    "infer_estimator_capabilities",
    "list_metric_specs",
    "validate_metric",
]
