# Workflow recipes

This page is a compact recipe book. For concepts and guarantees, follow the links in [`docs/README.md`](README.md).

## Validate prepared data

```python
result = saber.validate(
    dataset=dataset,
    algorithm="logistic_regression",
    partition_plan=plan,
    metrics=("mcc", "balanced_accuracy", "roc_auc"),
    random_state=42,
)
```

Use `partitioning=BioSievePartitionConfig(...)` instead of `partition_plan` when memberships must be generated.

## Final fit

```python
trained = saber.train(
    dataset=dataset,
    algorithm="random_forest",
    random_state=42,
)
```

Assessment belongs to `validate`/`evaluate`; `train` fits a final model on the supplied samples.

## Tune

```python
optimized = saber.tune(
    dataset=dataset,
    algorithm="logistic_regression",
    config=tuning_config,
    partition_plan=plan,
    search_space=space,
)
```

## Benchmark

```python
result = saber.benchmark(
    datasets={"repr_a": dataset_a, "repr_b": dataset_b},
    algorithms=("logistic_regression", "random_forest"),
    partitions={"shared_cv": plan},
    config=benchmark_config,
)

metrics = result.aggregate_metrics_frame()
predictions = result.predictions_frame()
```

## Persist and reload

```python
saber.save_model(
    "artifacts/model",
    model=trained.model,
    algorithm=trained.spec.name,
    task=trained.spec.task,
    dataset=dataset,
    provider=trained.spec.provider,
    parameters=trained.parameters,
)

artifact = saber.load_model("artifacts/model")
y_pred = artifact.predict(X_new, feature_names=feature_names)
```

## Declarative workflow

```bash
saber validate experiment.yaml --dry-run
saber validate experiment.yaml
```

Or from Python:

```python
config = saber.load_config("experiment.yaml")
execution = saber.run_config(config)
```

## Next

- [Data and partitions](data_and_partitions.md)
- [Metrics and evaluation](metrics_and_evaluation.md)
- [Tuning](tuning.md)
- [Benchmarking](benchmarking.md)
- [Persistence](persistence.md)
- [CLI](cli.md)
