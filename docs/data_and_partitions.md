# Data and partitions

## DatasetBundle

`DatasetBundle` holds the features `X`, a single target `y`, and optionally
`sample_ids`, `feature_names`, `groups`, `sample_weight` and free-form
`metadata`.

```python
from saber import DatasetBundle

dataset = DatasetBundle(X=X, y=y, sample_ids=sample_ids)
print(dataset.n_samples, dataset.n_features, dataset.fingerprint)
```

`X` can be a NumPy array, a Polars DataFrame or a pandas DataFrame, and must be
numerical and two-dimensional. Missing values are allowed and are imputed
inside each training fold; infinities are rejected. Always pass `sample_ids`:
partitions, out-of-fold predictions, benchmark tables and saved models all refer
to samples by ID.

`dataset.fingerprint` is a hash of the features, targets, sample IDs, feature
names, groups and weights (not `metadata`). Partition plans and saved models
record it, so a plan built for one dataset can't be applied to another by
mistake.

## PartitionPlan

A `PartitionPlan` is a list of splits, each with `train_ids`, `validation_ids`
and optionally `test_ids`. Plans with overlapping or unknown IDs, or a
mismatched fingerprint, are rejected.

A holdout:

```python
from saber import PartitionPlan

plan = PartitionPlan.holdout(
    train_ids=train_ids,
    validation_ids=validation_ids,
    test_ids=test_ids,
    dataset_fingerprint=dataset.fingerprint,
)
```

With a test set, tuning and tuned benchmarks select on train/validation and
report only on test.

Cross-validation from a fold column:

```python
plan = PartitionPlan.from_predefined_folds(
    sample_ids=dataset.sample_ids,
    fold_assignments=fold_ids,
    dataset_fingerprint=dataset.fingerprint,
)
```

A fold of `-1` means the sample is always in training, as in scikit-learn's
`PredefinedSplit`.

Partitions made by other tools can be loaded from JSON, CSV, TSV or a
DataFrame with `load_partition_plan()` or `partition_plan_from_frame()`.

## Generating partitions with BioSieve

When the data isn't split yet, Saber delegates to BioSieve
(`saberlib[biosieve]`):

```python
from saber import BioSievePartitionConfig

partitioning = BioSievePartitionConfig(
    strategy="stratified_kfold",
    params={"n_splits": 5, "seed": 42},
)
result = saber.validate(dataset=dataset, algorithm="random_forest_classifier", partitioning=partitioning)
```

The resulting plan keeps BioSieve's strategy, parameters and statistics. Saber
has no splitters of its own and doesn't do redundancy reduction; if you need
it, run it before building the `DatasetBundle`.
