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

`mlcore.datasets` defines validated dataset and partition contracts. `mlcore.validation` remains responsible for executing model-validation workflows in Phase 4. Partition generation and partition consumption are distinct concerns.

Phase 3 establishes `DatasetBundle` as the canonical supervised data container:

```text
DatasetBundle
├── X
├── y
├── sample_ids
├── FeatureSchema / feature_names
├── groups
├── sample_weight
└── metadata
```

Feature matrices are numerical and two-dimensional. Missing feature values may be preserved so Phase 4 can impute them inside training folds; infinite feature values are rejected. Targets remain single-output, and classification/regression semantics can be validated explicitly. Sample IDs are unique and stable across subsets.

Dataset fingerprints are deterministic hashes of scientific content (`X`, `y`, sample identity, feature identity, groups, and sample weights); free-form metadata is intentionally excluded. `FeatureSchema` preserves ordered feature identity and dtypes for later inference validation.

`PartitionPlan` contains one or more explicit `PartitionSplit` objects with train/validation/test membership by sample ID. Overlap inside a split, duplicate membership, unknown sample IDs, incomplete coverage (when required), and dataset-fingerprint mismatches fail explicitly. Membership ordering is non-semantic, so resolving a split preserves the original dataset order.

Predefined fold assignments are converted once into explicit memberships. External JSON/CSV/TSV/DataFrame partition tables can be ingested through a normalized interchange contract.

BioSieve is the **canonical partition-generation engine** when a dataset is not already partitioned. `mlcore` does not reimplement random, stratified, group, distance-aware, homology-aware, or k-fold split generation. Instead, the optional `mlcore[biosieve]` adapter converts `DatasetBundle` into a BioSieve-compatible table, executes the requested BioSieve splitter, preserves BioSieve strategy/parameter/statistics provenance, and converts returned memberships into `PartitionPlan`. Already-partitioned datasets bypass BioSieve and are consumed exactly as supplied. Redundancy reduction remains fully outside `mlcore`; BioSieve reduction, if desired, occurs upstream before `DatasetBundle` creation.

### 4. Preprocessing

Preprocessing is based on scikit-learn compatible transformers and pipelines. Phase 1+ implementations must guarantee that fitted transforms used for model selection live inside the fold-specific pipeline.

The first supported preprocessing scope is intentionally numerical:

- passthrough;
- simple imputation;
- standard scaling;
- robust scaling;
- min-max scaling;
- user-supplied compatible transformers/pipelines.

Phase 4 constructs preprocessing together with the estimator inside one scikit-learn `Pipeline` for every explicit split. Imputation/scaling statistics therefore never see held-out data. `auto` preprocessing respects estimator metadata: models with recommended scaling receive standard scaling; non-negative-input models receive min-max scaling; estimators with native missing-value support can bypass automatic imputation. Categorical inference and complex automatic feature engineering remain outside the initial core.

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

Phase 5 adds a partition-driven `TuningEngine`. It consumes the same
`DatasetBundle`, `PartitionPlan`/BioSieve integration, estimator factory, and
leakage-safe preprocessing pipeline as validation. Hyperparameter search never
generates an internal fallback split. In a holdout containing train, validation,
and final-test memberships, tuning uses train/validation and excludes final test
from both search and refit unless test evaluation is explicitly requested.

Search spaces are backend-agnostic. Finite lists remain compatible, while typed
`Categorical`, `Integer`, `Float`, and `LogFloat` domains translate to sklearn
grid/random/halving representations and Optuna suggestions where mathematically
applicable. Grid search fails explicitly for continuous domains that are not
enumerable. Multi-metric search has one explicit `refit_metric`; successive
halving and Optuna optimize that objective and evaluate the selected candidate
with any additional requested metrics over the same explicit folds.

Phase 6 adds a `BenchmarkEngine` as a pure orchestration layer over validation
and tuning. It does not implement a fourth model-execution path. A benchmark
matrix crosses prepared numerical representations, registered algorithms, named
partition scenarios, seeds, and untuned/tuned modes. `DummyClassifier` and
`DummyRegressor` provide deterministic scientific baselines.

Prepared representations are described by `BenchmarkDataset`; mlcore never
generates those representations. Multiple representations in one comparison
must contain the same sample IDs and targets. A partition membership generated
from one benchmark representation can then be rebound to another representation
without regenerating splits, preserving fair comparisons while maintaining the
representation-specific dataset fingerprint. Plans from unrelated dataset
fingerprints are rejected.

Tuned benchmark reporting is conservative: hyperparameters are selected on
train/validation, the selected parameters are refit on train+validation, and
performance is reported only on a protected test membership. Reusing ordinary
CV folds for both tuning and final reporting is intentionally rejected; unbiased
tuned CV requires nested CV or an external final test.

`BenchmarkResult` preserves individual `ValidationResult`/`OptimizationResult`
objects and exposes long-form run, aggregate-metric, fold-metric, prediction,
failure, and tuning-history tables. Deterministic run/configuration identifiers
link every score to its representation, partition, seed, explicit parameters,
and held-out sample predictions.

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

Persistence saves fitted estimators/pipelines together with sufficient metadata to validate future inference. Persistence contains no training logic and never generates partitions or representations.

Phase 7 defines a versioned directory artifact rather than an opaque single blob. Model artifacts contain a joblib-serialized fitted estimator/pipeline plus human-readable JSON files for the manifest, environment, feature schema, provenance, parameters, metrics, training configuration, and optional explicit `PartitionPlan`. Every artifact file is covered by a SHA-256 checksum manifest before joblib deserialization. Checksum verification establishes integrity, not authenticity; pickle/joblib artifacts must still originate from a trusted source.

The artifact schema version is independent from the mlcore package version. Loading validates schema compatibility, feature-schema fingerprints, required files, checksums, and environment differences. Environment differences are surfaced as warnings by default and can be promoted to hard compatibility errors. Inference through a loaded model revalidates feature identity/order before calling the persisted pipeline and can reconstruct the canonical `PredictionResult`, including class order and positive-class semantics.

Benchmark persistence is table-first: run, metric, held-out prediction, failure, and optimization-history tables are stored as analysis-ready CSV together with benchmark metadata and environment provenance. Serializing the full Python `BenchmarkResult` is optional because fold estimators and backend-specific studies may be large and less portable.

### 8. Public surfaces

Phase 8 exposes one thin public surface over the frozen orchestration engines:

```text
YAML / JSON ──► WorkflowConfig ──► run_config ──┐
CLI         ──► load_config  ────────────────┐  │
                                             ▼  ▼
                                      public Python API
                                             │
                           ┌─────────────────┼─────────────────┐
                           ▼                 ▼                 ▼
                    ValidationEngine    TuningEngine     BenchmarkEngine
                           │                                   │
                           └──────── persistence / prediction ─┘
```

The root Python API exposes `train`, `validate`, `evaluate`, `tune`/`optimize`,
`benchmark`, `predict`, persistence aliases, and config execution. Declarative
configs use schema version `1.0`; YAML and JSON normalize to the same
`WorkflowConfig` contract. Relative paths are resolved relative to the config
file. The CLI only parses arguments, loads/validates configuration, invokes the
public/config API, and renders outputs; it contains no estimator construction,
partition generation, preprocessing, tuning, or benchmark logic.

Model discovery and artifact inspection/verification are read-only CLI
operations over `MODEL_REGISTRY` and Phase 7 persistence contracts. Config-driven
partition generation continues to use BioSieve exclusively; the public surface
does not reintroduce sklearn splitters.

## Public versus internal API

The long-term public API is intentionally small:

```python
import mlcore
from mlcore import MODEL_REGISTRY
```

Phase 8 exposes the stable high-level functions `train`, `validate`, `evaluate`, `tune`/`optimize`, `benchmark`, and `predict` from the package root together with persistence and config helpers.

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
