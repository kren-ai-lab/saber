# Hyperparameter optimization

`saber.tune(...)` searches over the whole preprocessing + estimator pipeline,
using the same partitions and metrics as `saber.validate(...)`. Optimizers:
`grid`, `random`, `halving_grid`, `halving_random` and `optuna`
(`saberlib[optuna]`).

## Search spaces

```python
from saber.core.search_space import Categorical, Integer, LogFloat, SearchSpace

space = SearchSpace(
    name="forest",
    parameters={
        "n_estimators": Integer(100, 500, step=100),
        "max_features": Categorical(["sqrt", "log2", None]),
    },
)
```

Domains are `Categorical`, `Integer`, `Float` and `LogFloat`, and a plain list
works as a categorical. Grid search rejects a `Float` without a `step`. A
stepped `Float` must span a whole number of steps (`Float(0.0, 0.9, step=0.3)`,
not `Float(0.0, 1.0, step=0.3)`).

## Running a search

```python
import saber
from saber.tuning import TuningConfig

config = TuningConfig(
    optimizer="optuna",
    metrics=("mcc", "roc_auc", "balanced_accuracy"),
    refit_metric="mcc",
    n_trials=50,
    random_state=42,
)
result = saber.tune(
    dataset=dataset,
    algorithm="random_forest",
    partition_plan=plan,
    search_space=space,
    config=config,
)
print(result.best_params)
print(result.history_frame())
```

`refit_metric` is the objective; the other metrics are recorded for every
candidate. For binary targets, `precision`, `recall`, `f1` and `roc_auc` are
scored for the positive class; pass `positive_class=` to change it.

## Protected test

With a train/validation/test holdout, the search uses train/validation, the
best configuration is refit on train+validation, and the test set is only used
for the final report. Ordinary cross-validation can't be used both to pick
hyperparameters and to report unbiased performance, so tuned benchmarks require
a protected test (or a nested design done upstream).

## Examples

- [`05_hyperparameter_optimization.py`](../examples/05_hyperparameter_optimization.py)
- [`06_optuna_optimization.py`](../examples/06_optuna_optimization.py)
- [`07_optimizer_comparison.py`](../examples/07_optimizer_comparison.py)
