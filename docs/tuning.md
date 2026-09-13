# Hyperparameter optimization

`TuningEngine` uses the same dataset, partition, preprocessing, estimator factory, and metric semantics as validation.

## Optimizers

Supported optimizer names are:

```text
grid
random
halving_grid
halving_random
optuna
```

## Typed search spaces

```python
from mlcore.core.search_space import (
    SearchSpace,
    Categorical,
    Integer,
    Float,
    LogFloat,
)

space = SearchSpace(
    name="example",
    parameters={
        "n_estimators": Integer(100, 500, step=100),
        "max_features": Categorical(["sqrt", "log2", None]),
        "learning_rate": LogFloat(1e-3, 1e-1),
    },
)
```

Finite lists remain valid categorical domains. Grid search requires enumerable domains; a continuous `Float` without a step is therefore rejected instead of being discretized silently.

## Multi-metric tuning

```python
from mlcore.tuning import TuningConfig

config = TuningConfig(
    optimizer="optuna",
    metrics=("mcc", "roc_auc", "balanced_accuracy"),
    refit_metric="mcc",
    n_trials=50,
    random_state=42,
)
```

One metric is the explicit optimization/refit objective. Additional requested metrics are retained for the same candidate/folds.

## Leakage safety

The searched object is the complete preprocessing + estimator pipeline. Imputation/scaling are therefore fitted independently inside every training fold for every candidate.

## Protected test semantics

For train/validation/test holdouts, search uses train/validation. The selected configuration can be refit on train+validation while the final test remains untouched by selection.

Ordinary CV cannot simultaneously serve as both tuning data and unbiased final performance reporting. Tuned benchmarking therefore requires a protected test or a scientifically appropriate nested design upstream.

## Result inspection

`OptimizationResult` exposes best parameters/scores, failures, history, the fitted model when `refit=True`, partition provenance, and a long-form `history_frame()` suitable for convergence/candidate analysis.

See the executable notebooks:

- [`examples/tuning/01_hyperparameter_optimization.ipynb`](../examples/tuning/01_hyperparameter_optimization.ipynb)
- [`examples/tuning/02_optuna_optimization.ipynb`](../examples/tuning/02_optuna_optimization.ipynb)
- [`examples/tuning/03_optimizer_comparison.ipynb`](../examples/tuning/03_optimizer_comparison.ipynb)
