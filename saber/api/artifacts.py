"""Thin high-level aliases for persistence operations."""

from saber.persistence import (
    inspect_artifact,
    load_benchmark_artifact,
    load_model_artifact,
    save_benchmark_artifact,
    save_model_artifact,
    verify_artifact,
)

save_model = save_model_artifact
load_model = load_model_artifact
save_benchmark = save_benchmark_artifact
load_benchmark = load_benchmark_artifact

__all__ = [
    "inspect_artifact",
    "load_benchmark",
    "load_model",
    "save_benchmark",
    "save_model",
    "verify_artifact",
]
