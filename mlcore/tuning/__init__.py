"""Hyperparameter optimization public API."""

from mlcore.tuning.engine import TuningConfig, TuningEngine, tune_model
from mlcore.tuning.results import OptimizationResult
from mlcore.tuning.sklearn.grid import GridSearchOptimizer
from mlcore.tuning.sklearn.halving_grid import HalvingGridSearchOptimizer
from mlcore.tuning.sklearn.halving_random import HalvingRandomSearchOptimizer
from mlcore.tuning.sklearn.random import RandomSearchOptimizer

__all__ = [
    "GridSearchOptimizer",
    "HalvingGridSearchOptimizer",
    "HalvingRandomSearchOptimizer",
    "OptimizationResult",
    "OptunaOptimizer",
    "RandomSearchOptimizer",
    "TuningConfig",
    "TuningEngine",
    "tune_model",
]


def __getattr__(name: str):
    """Load the optional Optuna adapter only when explicitly requested."""

    if name == "OptunaOptimizer":
        from mlcore.tuning.optuna import OptunaOptimizer

        return OptunaOptimizer
    raise AttributeError(name)
