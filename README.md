# Saber

[![PyPI](https://img.shields.io/pypi/v/saberlib?style=flat-square)](https://pypi.org/project/saberlib/)
[![PyVersions](https://img.shields.io/pypi/pyversions/saberlib?style=flat-square)](https://github.com/kren-ai-lab/saber)
[![Tests](https://img.shields.io/github/actions/workflow/status/kren-ai-lab/saber/tests.yml?style=flat-square)](https://github.com/kren-ai-lab/saber/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/kren-ai-lab/saber?style=flat-square)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23165090-blue?style=flat-square)](https://doi.org/10.5281/zenodo.23165090)

Saber is a Python library for classical supervised machine learning
(classification and regression) on numerical tabular features. You give it a
feature matrix and a target; it trains, validates, tunes, benchmarks and saves
scikit-learn, XGBoost and LightGBM models. Preprocessing is fitted inside each
fold, and every result records which samples, partition and parameters
produced it.

Computing the features is up to you. Deep learning, AutoML, multilabel and
multi-output problems are out of scope.

## Installation

Saber supports Python 3.11 to 3.14.

```bash
pip install saberlib
# or with uv:
uv add saberlib
```

Optional extras:

```bash
pip install "saberlib[biosieve]"   # partition generation
pip install "saberlib[optuna]"     # Optuna tuning
pip install "saberlib[xgboost]"    # XGBoost models
pip install "saberlib[lightgbm]"   # LightGBM models
pip install "saberlib[all]"
```

For a development setup, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Validate a model

```python
import numpy as np
from sklearn.datasets import make_classification

import saber
from saber import DatasetBundle, PartitionPlan

X, y = make_classification(n_samples=120, n_features=12, random_state=42)
dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(len(y))])

plan = PartitionPlan.from_predefined_folds(
    sample_ids=dataset.sample_ids,
    fold_assignments=np.arange(len(y)) % 4,
    dataset=dataset,
)

result = saber.validate(
    dataset=dataset,
    algorithm="logistic_regression",
    partition_plan=plan,
    metrics=("mcc", "balanced_accuracy", "roc_auc"),
    random_state=42,
)
print(result.aggregate_metrics)
```

`result.oof_prediction` holds the out-of-fold predictions, aligned by sample
ID. `X` can be a NumPy array, a Polars DataFrame or a pandas DataFrame, and
partitions refer to samples by ID rather than by row position. If your data
isn't split yet, [BioSieve](https://pypi.org/project/biosieve/) can generate
the partitions:

```python
from saber import BioSievePartitionConfig

result = saber.validate(
    dataset=dataset,
    algorithm="random_forest_classifier",
    partition_plan=BioSievePartitionConfig(
        strategy="stratified_kfold",
        params={"n_splits": 5, "seed": 42},
    ),
    metrics=("mcc", "roc_auc"),
)
```

The [data and partitions](docs/data_and_partitions.md) guide covers holdouts,
external partition files and BioSieve.

## Tune, benchmark and save

```python
from saber import Categorical, LogFloat, SearchSpace, TuningConfig

tuned = saber.tune(
    dataset=dataset,
    algorithm="svc",
    partition_plan=plan,
    search_space=SearchSpace(
        parameters={"C": LogFloat(1e-3, 1e2), "kernel": Categorical(["linear", "rbf"])},
    ),
    config=TuningConfig(optimizer="random", n_trials=20),
    metrics=("mcc",),
    random_state=42,
)
print(tuned.best_params)
```

`saber.benchmark(...)` runs every combination of the representations,
partitions, algorithms, seeds and tuning modes you list, and returns long-form
Polars tables with the metrics and per-sample predictions. `saber.train(...)`
fits a final model. `saber.save_model(path, result, dataset=dataset)` writes a
`train` or `tune` result to a directory together with its feature schema,
provenance and checksums, and `saber.load_model(...)` checks those checksums
before loading.
See the [tuning](docs/tuning.md), [benchmarking](docs/benchmarking.md) and
[persistence](docs/persistence.md) guides.

## Command line

```bash
saber models list --task classification
saber run experiment.yaml --dry-run
saber run study.yaml --json
saber artifact verify artifacts/model
```

Every workflow can be written as a YAML or JSON file and run from the CLI or
with `saber.run_config("experiment.yaml")`. See the [configuration](docs/configuration.md)
and [CLI](docs/cli.md) references.

## Design principles

- Imputation and scaling are fitted on each training fold only.
- Partitions you provide are used exactly as given. When Saber has to create
  them, it delegates to BioSieve.
- A tuned model is scored on a held-out test set that played no part in
  choosing its hyperparameters.
- In binary problems you name the positive class; Saber maps it to the right
  probability column using the fitted estimator's class order.
- Saved models carry dataset and partition fingerprints, and their checksums
  are verified before loading. They are serialized with joblib, so only load
  models from sources you trust.

These rules prevent the common leakage mistakes inside Saber. Whether your
partitions and metrics fit your scientific question, and whether leakage crept
in before the data reached Saber, is still your call.

## Documentation and examples

The [documentation index](docs/README.md) links all the guides. There are also
[13 marimo example notebooks](examples/README.md). To open one:

```bash
uv sync --all-extras --group examples
uv run marimo edit examples/01_binary_classification.py
```

## Citing and license

If you use Saber in published work, please cite it with
[`CITATION.cff`](https://github.com/kren-ai-lab/saber/blob/main/CITATION.cff)
or GitHub's "Cite this repository" button. Saber is released under the [MIT license](https://github.com/kren-ai-lab/saber/blob/main/LICENSE).
