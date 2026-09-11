"""mlcore — classical supervised machine learning infrastructure."""

from mlcore.core.registry import MODEL_REGISTRY

# Force core algorithm registration. Optional providers register lazily when available.
import mlcore.classification
import mlcore.regression

from mlcore.api import (
    benchmark,
    evaluate,
    inspect_artifact,
    load_benchmark,
    load_model,
    optimize,
    predict,
    save_benchmark,
    save_model,
    train,
    tune,
    validate,
    verify_artifact,
)
from mlcore.config import dump_config, load_config, run_config

__all__ = [
    "MODEL_REGISTRY",
    "benchmark",
    "dump_config",
    "evaluate",
    "inspect_artifact",
    "load_benchmark",
    "load_config",
    "load_model",
    "optimize",
    "predict",
    "run_config",
    "save_benchmark",
    "save_model",
    "train",
    "tune",
    "validate",
    "verify_artifact",
]

__version__ = "0.1.0"
