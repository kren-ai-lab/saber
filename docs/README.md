# Saber documentation

Start with the [quickstart](../README.md) to install Saber and validate a model.

| I want to… | Read |
| --- | --- |
| Build a dataset, attach partitions or generate them with BioSieve | [Data and partitions](data_and_partitions.md) |
| Choose metrics and read predictions | [Metrics and evaluation](metrics_and_evaluation.md) |
| Optimize hyperparameters | [Tuning](tuning.md) |
| Compare representations, partitions and algorithms | [Benchmarking](benchmarking.md) |
| Save a model and predict with it later | [Persistence](persistence.md) |
| Describe a workflow in YAML or JSON | [Configuration](configuration.md) |
| Run Saber from the terminal | [CLI reference](cli.md) |
| Explore complete workflows and plots | [Example notebooks](../examples/README.md) |
| Set up a checkout, change code and run checks | [Development guide](../DEVELOPMENT.md) |

## Python API help

The workflow functions live in the package root: `saber.train`,
`saber.validate`, `saber.evaluate`, `saber.tune`,
`saber.benchmark` and `saber.predict`, plus `save_model`, `load_model`,
`load_config` and `run_config`. Use Python help for the signatures and defaults
of your installed version:

```python
import saber
from saber.datasets import DatasetBundle

help(saber.validate)
help(DatasetBundle)
```

To find an algorithm, browse the algorithm catalog, or run `saber models list`:

```python
from saber import ALGORITHMS, get_algorithm

[s.name for s in ALGORITHMS.values() if s.task == "classification" and s.provider == "sklearn"]
get_algorithm("random_forest_classifier").metadata()
```

Modules and names that start with `_` are internal.
