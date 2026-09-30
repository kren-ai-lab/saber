# Saber

[![PyPI](https://img.shields.io/pypi/v/saberlib?style=flat-square)](https://pypi.org/project/saberlib/)
[![PyVersions](https://img.shields.io/pypi/pyversions/saberlib?style=flat-square)](https://github.com/kren-ai-lab/saber)
[![Tests](https://img.shields.io/github/actions/workflow/status/kren-ai-lab/saber/tests.yml?style=flat-square)](https://github.com/kren-ai-lab/saber/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/kren-ai-lab/saber?style=flat-square)

Saber is a Python library for classical supervised machine learning
(classification and regression) on numerical tabular features. It trains,
validates, tunes, benchmarks and persists scikit-learn, XGBoost and LightGBM
models, with preprocessing fitted inside each fold and every result tied back
to its samples, partition and parameters.

Saber doesn't compute features. Bring a descriptor table, an embedding or any
numeric matrix, and Saber models it. Deep learning, AutoML, multilabel and
multi-output problems are out of scope.

## Installation

Saber supports Python 3.11 to 3.14.

```bash
python -m pip install saberlib
```

Optional extras:

```bash
python -m pip install "saberlib[biosieve]"   # partition generation
python -m pip install "saberlib[optuna]"     # Optuna tuning
python -m pip install "saberlib[xgboost]"    # XGBoost models
python -m pip install "saberlib[lightgbm]"   # LightGBM models
python -m pip install "saberlib[all]"
```

For development setup with
`uv`, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Validate a model

```python
import numpy as np
from sklearn.datasets import make_classification

import saber
from saber.datasets import DatasetBundle, PartitionPlan

X, y = make_classification(n_samples=120, n_features=12, random_state=42)
dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(len(y))])

plan = PartitionPlan.from_predefined_folds(
    sample_ids=dataset.sample_ids,
    fold_assignments=np.arange(len(y)) % 4,
    dataset_fingerprint=dataset.fingerprint,
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
ID. `X` can be a NumPy array, a Polars DataFrame or a pandas DataFrame. Partitions
refer to samples by ID, never by row position. If your data isn't split yet,
let BioSieve generate the partitions:

```python
from saber.datasets import BioSievePartitionConfig

result = saber.validate(
    dataset=dataset,
    algorithm="random_forest_classifier",
    partitioning=BioSievePartitionConfig(
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
from saber.core.search_space import Categorical, LogFloat, SearchSpace
from saber.tuning import TuningConfig

tuned = saber.tune(
    dataset=dataset,
    algorithm="svc",
    partition_plan=plan,
    search_space=SearchSpace(
        name="svc",
        parameters={"C": LogFloat(1e-3, 1e2), "kernel": Categorical(["linear", "rbf"])},
    ),
    config=TuningConfig(optimizer="random", metrics=("mcc",), n_trials=20, random_state=42),
)
print(tuned.best_params)
```

`saber.benchmark(...)` crosses representations, partitions, algorithms, seeds
and tuned/untuned modes, and returns long-form Polars tables of metrics and
per-sample predictions. `saber.train(...)` fits a final model, and
`saber.save_model(...)` writes it as a directory with a feature schema,
provenance and checksums that `saber.load_model(...)` verifies before loading.
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
with `saber.run_config(...)`. See the [configuration](docs/configuration.md)
and [CLI](docs/cli.md) references.

## Design principles

- Imputation and scaling are fitted inside each training fold, never on the
  whole dataset.
- Existing partitions are used exactly as given. New ones come from BioSieve;
  Saber has no splitters of its own.
- Tuned results are reported on a protected test set, not on the folds used to
  pick the hyperparameters.
- For binary problems the positive class is explicit, never inferred from a
  probability column's position.
- Saved models carry dataset and partition fingerprints and are checksum
  verified before loading. They use joblib, so load them only from trusted
  sources.

Saber keeps the workflow honest, but it can't tell whether your partitions or
metrics answer your scientific question, or catch leakage that happened
upstream.

## Documentation and examples

The [documentation index](docs/README.md) links all the guides. There are also
[13 example notebooks](examples/README.md) written in marimo. To open one:

```bash
uv sync --all-extras --group examples
uv run marimo edit examples/01_binary_classification.py
```

## Citing and license

If you use Saber in published work, please cite it using
[`CITATION.cff`](https://github.com/kren-ai-lab/saber/blob/main/CITATION.cff), or the "Cite this repository" button on
GitHub. Saber is released under the [MIT license](https://github.com/kren-ai-lab/saber/blob/main/LICENSE).
