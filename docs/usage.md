# mlcore Usage

Phase 8 exposes the same workflow through Python, YAML/JSON, and the CLI. The CLI/config layers are thin adapters over the public Python API; they do not implement separate model-training logic.

## Python API

```python
import mlcore

result = mlcore.validate(
    dataset=dataset,
    algorithm="logistic_regression",
    partition_plan=plan,
    metrics=("mcc", "balanced_accuracy", "roc_auc"),
    random_state=42,
)
```

The package root exposes:

```text
train
validate
evaluate
tune / optimize
benchmark
predict
save_model / load_model
save_benchmark / load_benchmark
inspect_artifact / verify_artifact
load_config / dump_config / run_config
```

`train` is a final-fit operation. Model assessment belongs to `validate`/`evaluate`; model selection belongs to `tune`; systematic comparisons belong to `benchmark`.

## Config schema

Configs are versioned independently from internal implementation details:

```yaml
schema_version: "1.0"
workflow: validate

dataset:
  path: data.csv
  target: label
  sample_id: sample_id

algorithm: logistic_regression

partition:
  path: folds.json

preprocessing:
  imputation: auto
  scaler: auto

metrics:
  - mcc
  - balanced_accuracy
  - roc_auc

random_state: 42
```

When data are not already partitioned, use BioSieve instead of defining an internal mlcore splitter:

```yaml
partitioning:
  strategy: stratified_kfold
  params:
    n_splits: 5
    seed: 42
```

External `partition` and BioSieve `partitioning` are mutually exclusive.

Relative paths in YAML/JSON are resolved relative to the config file. In-memory mapping configs resolve relative paths against the current working directory.

## CLI

Equivalent execution:

```bash
mlcore validate experiment.yaml
```

or:

```bash
mlcore run experiment.yaml
```

Available workflow commands:

```text
mlcore train CONFIG
mlcore evaluate CONFIG
mlcore validate CONFIG
mlcore tune CONFIG
mlcore optimize CONFIG   # alias for tune
mlcore benchmark CONFIG
mlcore predict CONFIG
```

Discovery and reproducibility utilities:

```bash
mlcore models list
mlcore models list --task classification
mlcore models show random_forest

mlcore config validate experiment.yaml
mlcore config normalize experiment.yaml -o normalized.yaml

mlcore artifact inspect artifacts/model
mlcore artifact verify artifacts/model
```

Exit-code policy:

```text
0  success
2  invalid configuration / CLI contract
3  mlcore workflow/domain failure
4  unexpected internal CLI failure
```

## Dataset files

The Phase 8 declarative loader consumes prepared numerical CSV/TSV/TXT tables. Configure identity/target columns explicitly when present:

```yaml
dataset:
  path: representation.csv
  sample_id: sequence_id
  target: activity
  groups: cluster_id
  sample_weight: weight
```

If `features` is omitted, all columns except configured target, ID, group, and weight columns are treated as numerical features.

mlcore does not generate molecular/protein representations and does not perform redundancy reduction. Those steps remain upstream. If mlcore must generate partitions, it delegates them to BioSieve.

## Tuning config

```yaml
workflow: tune
algorithm: logistic_regression

# dataset + partition/partitioning omitted here

tuning:
  optimizer: optuna
  metrics: [mcc, roc_auc]
  refit_metric: mcc
  n_trials: 50
  random_state: 42

search_space:
  C:
    type: log_float
    low: 1.0e-5
    high: 100.0
```

Supported typed domains are `categorical`, `integer`, `float`, and `log_float`. Finite YAML lists remain valid categorical domains.

## Benchmark config

A benchmark can consume multiple prepared representations sharing the same sample IDs and targets:

```yaml
workflow: benchmark

datasets:
  roxy:
    path: roxy.csv
    target: label
    sample_id: sample_id
  sylphy:
    path: sylphy.csv
    target: label
    sample_id: sample_id

algorithms:
  - logistic_regression
  - random_forest

partitions:
  cluster_disjoint:
    path: cluster_folds.json

benchmark:
  metrics: [mcc, balanced_accuracy]
  seeds: [42, 123]
  modes: [untuned]
  include_baselines: true
```

## Prediction artifacts

```yaml
workflow: predict

artifact: artifacts/final_model

dataset:
  path: inference.csv
  sample_id: sample_id

output:
  path: predictions.csv
```

The persisted `FeatureSchema` is checked before prediction, including feature names and order.
