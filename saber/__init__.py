"""saber — classical supervised machine learning infrastructure."""

from saber._version import __version__
from saber.core.registry import MODEL_REGISTRY

# Force core algorithm registration. Optional providers register lazily when available.
import saber.classification
import saber.regression

from saber.api import (
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
from saber.config import dump_config, load_config, run_config

__all__ = [
    "__version__",
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
