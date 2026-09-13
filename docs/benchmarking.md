# Benchmarking

Benchmarking is a scientific orchestration layer over validation and tuning. It does not create a separate model-execution path.

## Experiment matrix

A benchmark can cross:

```text
prepared representation(s)
× partition scenario(s)
× registered algorithm(s)
× random seed(s)
× untuned / tuned mode(s)
```

Multiple representations must contain the same sample IDs and targets. Partition memberships can then be reused across representations without regenerating folds, while dataset fingerprints remain representation-specific.

## Configuration

```python
from mlcore.benchmark import BenchmarkConfig

config = BenchmarkConfig(
    metrics=("mcc", "balanced_accuracy"),
    seeds=(42, 123, 456),
    modes=("untuned",),
    include_baselines=True,
)
```

Dummy classifier/regressor baselines can be included automatically.

## Tuned versus untuned comparability

When a benchmark mixes `untuned` and `tuned` modes on a holdout with train/validation/test memberships, all reported modes are evaluated on the **same protected test**. Untuned/baseline models are refit on train+validation before test evaluation so leaderboard rows remain scientifically comparable.

## Structured outputs

`BenchmarkResult` retains individual runs and exposes analysis-ready tables:

```python
runs = result.runs_frame()
aggregate = result.aggregate_metrics_frame()
folds = result.fold_metrics_frame()
predictions = result.predictions_frame()
failures = result.failures_frame()
optimization = result.optimization_history_frame()
```

Every metric/prediction row carries a deterministic `run_id` and configuration identity, connecting the reported score back to representation, partition, algorithm, mode, seed, parameters, fold, and sample-level prediction.

## Failure isolation

A model failure does not need to invalidate an entire benchmark matrix. Failed runs are retained separately unless `fail_fast=True` is requested.

## Reporting

Plotting and report layout stay outside the core. The long-form tables are designed to feed pandas, statistical analyses, matplotlib, R, or custom reporting systems.

Recommended notebooks:

- [`examples/benchmark/01_multi_representation_multi_partition.ipynb`](../examples/benchmark/01_multi_representation_multi_partition.ipynb)
- [`examples/reporting/01_benchmark_reporting.ipynb`](../examples/reporting/01_benchmark_reporting.ipynb)
- [`examples/end_to_end/01_data_centric_benchmark.ipynb`](../examples/end_to_end/01_data_centric_benchmark.ipynb)
