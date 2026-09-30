# Benchmarking

`saber.benchmark(...)` runs validation and tuning over a grid of
representations × partitions × algorithms × seeds × tuned/untuned modes.

```python
import saber
from saber import BenchmarkConfig

result = saber.benchmark(
    datasets={"repr_a": dataset_a, "repr_b": dataset_b},
    algorithms=("logistic_regression", "random_forest_classifier", "svc"),
    partitions={"shared_cv": plan},
    metrics=("mcc", "balanced_accuracy"),
    config=BenchmarkConfig(
        seeds=(42, 123, 456),
        modes=("untuned",),
        include_baselines=True,
    ),
)
```

`metrics` is required; `evaluation_role`, `require_complete` and `return_estimators` are also keyword arguments of `benchmark`. `partitions` accepts a `PartitionPlan`, a mapping of label to plan, or a `BioSievePartitionConfig` (the first dataset is the reference). In tuned mode, tuning uses the benchmark `metrics` and each seed as its `random_state`.

All representations must have the same sample IDs and targets, so one
partition plan can be reused across them. `include_baselines=True` adds dummy
classifier/regressor runs.

When `tuned` and `untuned` modes are mixed on a train/validation/test holdout,
every mode is evaluated on the same protected test, and untuned models are
refit on train+validation first, so rows stay comparable.

## Results

`BenchmarkResult` exposes long-form Polars tables:

```python
result.runs_frame()
result.aggregate_metrics_frame()
result.fold_metrics_frame()
result.predictions_frame()
result.failures_frame()
result.optimization_history_frame()
```

Every row carries a `run_id` that links it to its representation, partition,
algorithm, mode, seed and parameters. Failed runs go to `failures_frame()`
instead of stopping the benchmark, unless you set `fail_fast=True`.

Saber doesn't plot; feed these tables to your plotting or reporting tool of
choice.

## Examples

- [`10_multi_representation_benchmark.py`](../examples/10_multi_representation_benchmark.py)
- [`11_benchmark_reporting.py`](../examples/11_benchmark_reporting.py)
- [`13_data_centric_benchmark.py`](../examples/13_data_centric_benchmark.py)
