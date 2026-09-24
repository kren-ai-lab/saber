"""Stable core contracts for saber."""

from saber.core.capabilities import (
    EstimatorCapabilities,
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.estimator import EstimatorFactory
from saber.core.metrics import (
    METRIC_SPECS,
    MetricSpec,
    get_metric_spec,
    list_metric_specs,
    validate_metric,
)
from saber.core.prediction import PredictionResult
from saber.core.registry import MODEL_REGISTRY, AlgorithmRegistry
from saber.core.results import TrainResult
from saber.core.search_space import Categorical, Float, Integer, LogFloat, SearchSpace
from saber.core.specs import AlgorithmSpec
from saber.core.task import TaskType

__all__ = [
    "METRIC_SPECS",
    "MODEL_REGISTRY",
    "AlgorithmRegistry",
    "AlgorithmSpec",
    "Categorical",
    "EstimatorCapabilities",
    "EstimatorFactory",
    "EstimatorRequirements",
    "Float",
    "Integer",
    "LogFloat",
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
