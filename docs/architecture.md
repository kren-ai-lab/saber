# mlcore Architecture

## Status

This document is the **Phase 0 architecture freeze**. It defines the target boundaries that later implementation phases must follow. Existing code may temporarily retain legacy runner/backend objects until the Core Engine migration is completed, but new functionality must not deepen those legacy abstractions.

## Architectural target

```text
DatasetBundle
     │
     ├──────────────► PartitionPlan
     │                     │
     ▼                     ▼
Pipeline / Estimator Builder
             │
        AlgorithmSpec
             │
        EstimatorFactory
             │
     sklearn-compatible estimator
             │
       ┌─────┼───────────┐
       ▼     ▼           ▼
    Train  Validate     Tune
       │     │           │
       └─────┴─────┬─────┘
                   ▼
            PredictionResult
                   │
            EvaluationResult
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
 BenchmarkResult  Artifact   Reporting data
```

The principal invariant is that **training and hyperparameter optimization must not construct estimators through different mechanisms**.

## Layers

### 1. Core contracts

Responsible for stable semantic objects:

- tasks;
- algorithm specifications;
- estimator construction;
- search-space definitions;
- common result contracts;
- registry/discovery.

The core must not depend on persistence, CLI, plotting, domain-specific data code, or optional third-party estimator packages.

### 2. Estimator providers

Canonical provider modules are:

```text
mlcore/classification/sklearn.py
mlcore/classification/xgboost.py
mlcore/classification/lightgbm.py
mlcore/regression/sklearn.py
mlcore/regression/xgboost.py
mlcore/regression/lightgbm.py
```

The previous empty `mlcore/models/` duplication is retired. Provider registration belongs beside the task-specific estimator definitions until a future migration has a concrete need for another layout.

`scikit-learn` is mandatory. XGBoost and LightGBM are optional providers and must not prevent a core-only import.

### 3. Data and partitions

`mlcore.datasets` will define validated dataset contracts. `mlcore.validation` will define split/validation execution. Partition generation and partition consumption are distinct concerns.

The target dataset object will preserve at minimum:

```text
X
y
sample_ids
feature_names
groups
sample_weight
metadata
```

The target partition object will preserve sample identity and explicit train/validation/test or fold membership. BioSieve interoperability is file/contract based; BioSieve is not a dependency of `mlcore`.

### 4. Preprocessing

Preprocessing is based on scikit-learn compatible transformers and pipelines. Phase 1+ implementations must guarantee that fitted transforms used for model selection live inside the fold-specific pipeline.

The first supported preprocessing scope is intentionally numerical:

- passthrough;
- simple imputation;
- standard scaling;
- robust scaling;
- min-max scaling;
- user-supplied compatible pipelines.

Categorical inference and complex automatic feature engineering are not required for the initial stable core.

### 5. Training, validation, tuning, and benchmarking

These are orchestration layers over the same estimator-construction contract.

Phase 1 establishes one canonical construction path:

```text
AlgorithmSpec -> EstimatorFactory -> estimator/pipeline
```

`Trainer`, Grid Search, Random Search, Halving Search, and Optuna all construct
estimators through `AlgorithmSpec.build_estimator()`. Default parameters and
explicit random-state propagation are therefore resolved by the same factory.

The historical fields `runner` and `backend_cls` remain temporarily for source
compatibility. They are not part of the canonical execution path: `Trainer`
only mirrors fitted state into a legacy backend container, while direct legacy
runner calls remain supported for existing callers/tests. No new subsystem may
execute models through the runner/backend path.

### 6. Evaluation

Evaluation consumes structured prediction outputs rather than guessing probability semantics. Binary positive-class identity and multiclass class ordering are explicit.

Phase 2 establishes two canonical semantic contracts:

```text
MetricSpec
├── task
├── natural optimization direction
├── estimator response method
├── supported problem regime
└── scorer construction

PredictionResult
├── predictions
├── probabilities
├── decision scores
├── ordered classes
├── positive class
└── sample identity/metadata
```

Loss metrics retain their natural user-facing value while scikit-learn scorer objects transform them into maximization objectives internally. Metric/task and binary/multiclass compatibility is validated before optimization. ROC-AUC uses the supported `response_method` scorer API rather than deprecated probability flags.

For binary classifiers, a two-column `predict_proba` output is never interpreted by column position alone. The probability column is resolved through the estimator's explicit `classes_` order and `positive_class`. `Trainer.predict_result()` therefore produces an object that can be passed directly to the evaluation layer without manual `[:, 1]` extraction.

Optimization results reject non-finite best scores. Search backends preserve best parameters/history when `refit=False` while correctly reporting that no fitted best model is available.

Target result hierarchy:

```text
TrainingResult
PredictionResult
ValidationResult
OptimizationResult
EvaluationResult
BenchmarkResult
```

Each object owns only data relevant to its stage and can be serialized into downstream artifacts.

### 7. Persistence

Persistence saves fitted estimators/pipelines together with sufficient metadata to validate future inference. Persistence must not contain training logic.

### 8. Public surfaces

The final public surface will be thin wrappers over shared internal orchestration:

```text
Python API
YAML config ──► Python API
CLI         ──► Python API
```

CLI and configuration files must not implement a second execution engine.

## Public versus internal API

The long-term public API is intentionally small:

```python
import mlcore
from mlcore import MODEL_REGISTRY
```

Later phases may expose high-level functions such as `train`, `optimize`, `benchmark`, `evaluate`, and `predict` from the package root after their contracts stabilize.

Modules beginning with `_` are internal. Concrete provider modules, runners, backend state containers, serializers, and implementation helpers are not guaranteed public APIs.

## AlgorithmSpec contract

Phase 1 migrated `AlgorithmSpec` to a provider-neutral construction contract containing:

```text
name
task
provider
estimator factory/class
aliases
tags
description
default parameters
search space
capabilities
requirements
```

Capabilities/requirements describe behavior instead of requiring orchestration layers to probe fitted estimators. Phase 1 records:

```text
predict_proba
decision_function
sample_weight
native_missing_values
non_negative_X
positive_y
scaling recommendation
```

Additional capabilities can be added when they have reliable provider-level semantics.
The fields `runner` and `backend_cls` are legacy compatibility fields scheduled for removal only after downstream code no longer depends on them.

## Dependency boundaries

Mandatory dependency modules may never import optional packages at package import time. Optional provider discovery is performed lazily/conditionally.

Optuna is an optional optimization backend. XGBoost and LightGBM are optional estimator providers. TPOT is outside scope.

## Versioning and stability

Until the architecture, result contracts, validation engine, benchmarking, persistence, and integration gates are complete, the project remains **Alpha**. Reaching a high unit-test coverage percentage alone does not change stability status.

A stable release is gated by scientific/integration behavior, not a calendar date.
