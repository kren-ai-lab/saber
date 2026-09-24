# Saber documentation

Start with the [quickstart](../README.md) to install Saber and run a workflow.
Use the guides below for input requirements, method choices and result interpretation.

| I want to… | Read |
| --- | --- |
| Run a first Python workflow end to end | [Getting started](getting_started.md) |
| See what problems Saber supports and doesn't | [Scope](scope.md) |
| Understand subsystem boundaries and invariants | [Architecture](architecture.md) |
| Build `DatasetBundle`/`PartitionPlan` and use BioSieve | [Data & partitions](data_and_partitions.md) |
| Choose metrics and read `PredictionResult`/OOF outputs | [Metrics & evaluation](metrics_and_evaluation.md) |
| Optimize hyperparameters with a protected test | [Tuning](tuning.md) |
| Run multi-representation/multi-partition experiment matrices | [Benchmarking](benchmarking.md) |
| Save and reload auditable model artifacts | [Persistence](persistence.md) |
| Write YAML/JSON workflow configs | [Configuration](configuration.md) |
| Run Saber from the terminal | [CLI reference](cli.md) |
| Look up the high-level Python functions and contracts | [Python API](api_reference.md) |
| Copy compact workflow recipes | [Usage recipes](usage.md) |
| Explore complete executable examples | [Example notebooks](../examples/README.md) |
| Set up a checkout, change code and run checks | [Development guide](../DEVELOPMENT.md) |

## Python API help

Import public workflow functions and data contracts from `saber`. Use Python
help for the signatures and defaults of your installed version:

```python
import saber
from saber.datasets import DatasetBundle

help(saber.validate)
help(DatasetBundle)
```

The CLI reference links every command to its Python/config equivalent. Result
tables are Polars DataFrames; inspect their column names and types through
`.schema` — see the [results section](benchmarking.md#structured-outputs) for
a working example.
