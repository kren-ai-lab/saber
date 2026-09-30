"""saber — classical supervised machine learning infrastructure."""

from saber._api import evaluate, predict, train
from saber._version import __version__
from saber.benchmark import BenchmarkConfig, BenchmarkResult, benchmark
from saber.config import run_config
from saber.core import (
    ALGORITHMS,
    Categorical,
    Float,
    Integer,
    LogFloat,
    PredictionResult,
    SearchSpace,
    TrainResult,
)
from saber.datasets import BioSievePartitionConfig, DatasetBundle, PartitionPlan
from saber.evaluation import EvaluationResult
from saber.exceptions import SaberError
from saber.persistence import (
    LoadedModelArtifact,
    inspect_artifact,
    load_benchmark,
    load_model,
    save_benchmark,
    save_model,
)
from saber.preprocessing import PreprocessingConfig
from saber.tuning import OptimizationResult, TuningConfig, tune
from saber.validation import ValidationResult, validate

__all__ = [  # noqa: RUF022  # grouped by purpose, not alphabetical
    # Workflows
    "train",
    "validate",
    "tune",
    "benchmark",
    "evaluate",
    "predict",
    "run_config",
    # Persistence
    "save_model",
    "load_model",
    "save_benchmark",
    "load_benchmark",
    "inspect_artifact",
    # Inputs
    "DatasetBundle",
    "PartitionPlan",
    "BioSievePartitionConfig",
    "PreprocessingConfig",
    "TuningConfig",
    "BenchmarkConfig",
    "SearchSpace",
    "Categorical",
    "Integer",
    "Float",
    "LogFloat",
    # Results
    "TrainResult",
    "PredictionResult",
    "EvaluationResult",
    "ValidationResult",
    "OptimizationResult",
    "BenchmarkResult",
    "LoadedModelArtifact",
    # Catalog / misc
    "ALGORITHMS",
    "SaberError",
    "__version__",
]
