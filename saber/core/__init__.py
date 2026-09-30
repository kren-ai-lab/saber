"""Stable core contracts for saber."""

from saber.core.prediction import PredictionResult
from saber.core.registry import ALGORITHMS, get_algorithm
from saber.core.results import TrainResult
from saber.core.search_space import Categorical, Float, Integer, LogFloat, SearchSpace
from saber.core.specs import AlgorithmSpec
from saber.core.task import TaskType

__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "Categorical",
    "Float",
    "Integer",
    "LogFloat",
    "PredictionResult",
    "SearchSpace",
    "TaskType",
    "TrainResult",
    "get_algorithm",
]
