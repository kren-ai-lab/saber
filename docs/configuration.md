# YAML/JSON configuration

A config file describes one workflow with the same options as the Python API. Run it with `saber <workflow> config.yaml` or `saber.run_config(saber.load_config(path))`.

## Common structure

```yaml
schema_version: "1.0"
workflow: validate

dataset:
  path: data/features.csv
  target: label
  sample_id: sample_id

algorithm: logistic_regression

partitioning:
  strategy: stratified_kfold
  params:
    n_splits: 5
    seed: 42

preprocessing:
  imputation: auto
  scaler: auto

metrics: [mcc, balanced_accuracy, roc_auc]
random_state: 42
```

Relative paths are resolved against the config file location.

## Supported workflows

```text
train
evaluate
validate
tune
benchmark
predict
```

## Existing versus generated partitions

Use exactly one of:

```yaml
partition:
  path: partitions.json
```

or:

```yaml
partitioning:
  strategy: stratified_kfold
  params:
    n_splits: 5
    seed: 42
```

The second form requires BioSieve.

## Tuning

```yaml
workflow: tune
algorithm: ridge_regressor

tuning:
  optimizer: grid
  metrics: [rmse, mae]
  refit_metric: rmse
  random_state: 42

search_space:
  alpha:
    type: categorical
    values: [0.01, 0.1, 1.0, 10.0]
```

Typed domains: `categorical`, `integer`, `float`, `log_float`.

## Benchmark

```yaml
workflow: benchmark

datasets:
  representation_a:
    path: repr_a.csv
    target: label
    sample_id: sample_id
  representation_b:
    path: repr_b.csv
    target: label
    sample_id: sample_id

algorithms: [logistic_regression, random_forest_classifier]

partitions:
  external_cv:
    path: folds.json

benchmark:
  metrics: [mcc, balanced_accuracy]
  seeds: [42, 123]
  modes: [untuned]
  include_baselines: true
```

## Validation

```bash
saber run experiment.yaml --dry-run
```

`--dry-run` validates the config and renders the plan without executing the experiment.

Complete minimal files live in [`examples/configs/`](../examples/configs/).
