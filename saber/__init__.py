"""saber — classical supervised machine learning infrastructure."""

from saber._version import __version__
from saber.api import (
    benchmark,
    evaluate,
    inspect_artifact,
    load_benchmark,
    load_model,
    predict,
    save_benchmark,
    save_model,
    train,
    tune,
    validate,
    verify_artifact,
)
from saber.config import dump_config, load_config, run_config
from saber.core.registry import ALGORITHMS, get_algorithm

__all__ = [
    "ALGORITHMS",
    "__version__",
    "benchmark",
    "dump_config",
    "evaluate",
    "get_algorithm",
    "inspect_artifact",
    "load_benchmark",
    "load_config",
    "load_model",
    "predict",
    "run_config",
    "save_benchmark",
    "save_model",
    "train",
    "tune",
    "validate",
    "verify_artifact",
]
