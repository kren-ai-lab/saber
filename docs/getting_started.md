# Getting started

This guide uses an explicit fold assignment so it runs with the core installation only. If your data are not already partitioned, use BioSieve as described in [Data and partitions](data_and_partitions.md).

## 1. Prepare numerical features

Saber expects a two-dimensional numerical matrix and one supervised target.

```python
import numpy as np
from sklearn.datasets import make_classification

from saber.datasets import DatasetBundle

X, y = make_classification(
    n_samples=200,
    n_features=20,
    n_informative=10,
    random_state=42,
)

dataset = DatasetBundle(
    X=X,
    y=y,
    sample_ids=[f"sample_{i}" for i in range(len(y))],
)

print(dataset.n_samples, dataset.n_features)
print(dataset.fingerprint)
```

Prefer explicit `sample_ids` in scientific workflows. They are the identity key used by partitions, OOF predictions, benchmark tables, and persistence provenance.

## 2. Attach partitions

```python
from saber.datasets import PartitionPlan

folds = np.arange(dataset.n_samples) % 5
plan = PartitionPlan.from_predefined_folds(
    sample_ids=dataset.sample_ids,
    fold_assignments=folds,
    dataset_fingerprint=dataset.fingerprint,
)
```

The plan stores memberships by sample ID, not by fragile row positions.

## 3. Validate a model

```python
import saber

result = saber.validate(
    dataset=dataset,
    algorithm="logistic_regression",
    partition_plan=plan,
    metrics=("mcc", "balanced_accuracy", "roc_auc"),
    random_state=42,
)

print(result.aggregate_metrics)
print(result.metric_summary)
```

For complete CV, `result.oof_prediction` contains sample-aligned out-of-fold predictions.

## 4. Discover other algorithms

```bash
saber models list --task classification
saber models search forest
saber models show random_forest
```

The registry exposes provider, aliases, capabilities, preprocessing requirements, and search-space metadata.

## 5. Move to tuning or benchmarking

Once dataset identity and partition semantics are stable, continue with:

- [Hyperparameter optimization](tuning.md)
- [Benchmarking](benchmarking.md)
- [Persistence](persistence.md)

The same `DatasetBundle` and `PartitionPlan` contracts are reused throughout.
