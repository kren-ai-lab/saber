# Python API reference

This page documents the intentionally small high-level surface. Lower-level contracts remain available for advanced workflows, but concrete provider modules and helpers beginning with `_` are internal implementation details.

## Package-root functions

```python
import mlcore
```

### `mlcore.train(...)`
Fit one final estimator/pipeline on all samples supplied in a `DatasetBundle`. This is a final-fit operation, not model assessment.

### `mlcore.validate(...)`
Run holdout/CV validation from an explicit `PartitionPlan` or a `BioSievePartitionConfig`. Returns `ValidationResult` with fold results, aggregate summaries, and OOF predictions when defined.

### `mlcore.evaluate(...)`
Evaluate a fitted `TrainResult`, loaded model artifact, or persisted artifact path on labeled data through structured `PredictionResult` semantics.

### `mlcore.tune(...)` / `mlcore.optimize(...)`
Run partition-driven hyperparameter optimization with `TuningConfig` and `SearchSpace`.

### `mlcore.benchmark(...)`
Execute an experiment matrix across prepared datasets/representations, algorithms, partitions, seeds, and tuning modes. Returns `BenchmarkResult`.

### `mlcore.predict(...)`
Run inference through a persisted model artifact with feature-schema validation.

### Persistence helpers

```text
save_model
load_model
save_benchmark
load_benchmark
inspect_artifact
verify_artifact
```

### Configuration helpers

```text
load_config
dump_config
run_config
```

## Important data contracts

```python
from mlcore.datasets import (
    DatasetBundle,
    FeatureSchema,
    PartitionPlan,
    PartitionSplit,
    BioSievePartitionConfig,
)
```

## Preprocessing

```python
from mlcore.preprocessing import PreprocessingConfig
```

Numerical imputation/scaling can be explicit or `auto`; fitted preprocessing remains inside fold-specific pipelines.

## Tuning

```python
from mlcore.tuning import TuningConfig
from mlcore.core.search_space import SearchSpace, Categorical, Integer, Float, LogFloat
```

## Benchmarking

```python
from mlcore.benchmark import BenchmarkConfig, BenchmarkDataset, BenchmarkPartition
```

## Registry

```python
from mlcore import MODEL_REGISTRY

MODEL_REGISTRY.providers()
MODEL_REGISTRY.filter(task="classification", provider="sklearn")
MODEL_REGISTRY.describe("random_forest")
```

Prefer registry discovery over importing provider classes directly.

## Public/internal stability

The package-root workflow functions and major semantic contracts above are the intended public surface. Modules/classes prefixed with `_`, concrete serialization helpers, and provider-registration implementation details are not guaranteed stable APIs.
