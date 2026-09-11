# mlcore Development Roadmap

## Development principle

`mlcore` is developed by **acceptance-gated phases**, not by calendar deadlines. A phase is complete only when its implementation and validation gates pass. The target is a scientifically reproducible classical supervised machine-learning library with a maturity comparable to the other Kren AI Lab core libraries.

The scope contract is defined in `docs/scope.md`; the architecture freeze is defined in `docs/architecture.md`.

---

## Phase 0 — Scope and architecture freeze

**Status:** closed/frozen after repository-side validation (173 tests passed; compileall passed).

### Objectives

- freeze `mlcore` as the canonical distribution/import identity;
- define binary classification, multiclass classification, and single-target regression as the supported problem family;
- explicitly exclude deep learning, representation learning, AutoML, XAI, and domain-specific feature generation;
- freeze the target layered architecture and public/internal boundaries;
- establish optional-dependency policy;
- remove duplicate/out-of-scope placeholders;
- remove `MLPRegressor` from the classical estimator catalogue;
- normalize obvious registry tag inconsistencies;
- document the migration away from dual runner/backend versus direct-estimator execution paths;
- mark project maturity as Alpha until scientific/integration hardening is complete.

### Exit gates

- package imports with only core dependencies;
- package imports with all optional providers installed;
- complete current unit-test suite passes;
- package and tests compile cleanly with `python -m compileall`;
- no TPOT or backport `dataclasses` runtime dependency;
- no neural-network estimator is registered;
- architecture and scope documents exist and agree with package metadata.

---

## Phase 1 — Core Engine 2.0

**Status:** closed/frozen after repository-side validation.

### Objective

Unify estimator construction so training, validation, tuning, benchmarking, and inference cannot instantiate the same algorithm through different execution paths.

### Work

- introduce a canonical estimator factory/builder;
- migrate `AlgorithmSpec` toward provider-neutral estimator construction;
- define explicit capabilities and estimator requirements;
- propagate default parameters and random state consistently;
- migrate `Trainer` to the unified path;
- adapt optimizers to the same construction path;
- deprecate and then remove legacy `runner`/`backend_cls` execution requirements;
- improve registry descriptions, aliases, metadata, and discovery;
- add deterministic estimator-construction tests.

### Exit gates

- a registered algorithm is constructed identically by training and tuning;
- no optimizer directly bypasses the canonical estimator builder;
- existing supported algorithms remain trainable;
- registry discovery and metadata tests pass;
- legacy compatibility behavior is covered during migration.

---

## Phase 2 — Metrics and prediction contracts

**Status:** closed/frozen after repository-side validation.

### Objective

Make output semantics unambiguous across classification, regression, evaluation, and optimization.

### Work

- formal `MetricSpec` registry;
- metric/task compatibility validation;
- optimization direction and response-method metadata;
- correct ROC-AUC scorer implementation for supported scikit-learn versions;
- `PredictionResult` contract;
- explicit class order and binary positive-class semantics;
- support for `predict`, `predict_proba`, and `decision_function` outputs;
- non-finite score rejection and explicit failure handling;
- repair `refit=False` behavior;
- align training and evaluation probability contracts.

### Exit gates

- every public metric declares compatible task and required response;
- binary/multiclass probability semantics are tested;
- known incompatible metric/task combinations fail explicitly;
- ROC-AUC optimization produces finite scores on valid data;
- failed folds/trials cannot silently produce arbitrary best parameters.

---

## Phase 3 — Dataset and partition contracts

**Status:** closed/frozen after repository-side validation.

### Objective

Represent supervised datasets and partitions explicitly enough for reproducible training and external split interoperability.

### Work

- `DatasetBundle` with `X`, `y`, `sample_ids`, `feature_names`, `groups`, `sample_weight`, and metadata;
- validation for alignment, dimensions, finite values, target structure, and feature schema;
- `PartitionPlan` for holdout, train/validation/test, folds, and externally supplied partitions;
- overlap and coverage validation;
- dataset and partition fingerprints;
- predefined-fold support;
- BioSieve-compatible partition ingestion without a BioSieve dependency.

### Exit gates

- sample identity survives all supported partition workflows;
- overlapping or invalid partitions fail loudly;
- external partitions are consumed without silent regeneration;
- dataset/partition fingerprints are deterministic;
- classification and regression contracts are covered by unit tests.

---

## Phase 4 — BioSieve-integrated partitioning, leakage-safe preprocessing, and validation

**Status:** CLOSED / FROZEN after repository-side validation.

### Objective

Provide trustworthy holdout/CV evaluation while making BioSieve the only internal partition-generation engine and guaranteeing that data-dependent preprocessing is fitted only on training data within each explicit split.

### Work

- optional `mlcore[biosieve]` integration;
- accept already-partitioned `PartitionPlan` objects without regeneration;
- require explicit BioSieve configuration when input data are not partitioned;
- adapt BioSieve single-split and k-fold `SplitResult` outputs into `PartitionPlan`;
- preserve BioSieve version, strategy, effective parameters, and statistics as provenance;
- support aligned extra columns required by prepared BioSieve strategies without making `mlcore` domain-specific;
- explicitly keep BioSieve redundancy reduction outside `mlcore`;
- numerical preprocessing pipeline builder;
- passthrough, simple imputation, StandardScaler, RobustScaler, and MinMaxScaler;
- user-supplied compatible preprocessing transformers/pipelines;
- estimator-aware automatic preprocessing;
- holdout and explicit multi-fold validation driven exclusively by `PartitionPlan`;
- structured `ValidationResult` / fold results;
- first-class out-of-fold predictions/probabilities/scores when held-out sample identities are unique;
- fold metrics, aggregate metric summaries, runtime, fitted-estimator metadata, and sample-weight propagation.

### Non-goals

- no duplicate KFold/StratifiedKFold/GroupKFold implementation inside `mlcore`;
- no sequence redundancy reduction, homology clustering, MMseqs2 reduction, representative selection, or similarity computation inside `mlcore`;
- no silent fallback random split when partition configuration is missing.

### Exit gates

- already-partitioned data are consumed without regeneration;
- unpartitioned data can only be partitioned through BioSieve;
- BioSieve is optional when partitions are already supplied;
- BioSieve strategy provenance survives conversion to `PartitionPlan`;
- preprocessing leakage tests pass;
- OOF predictions preserve held-out sample identity and expected coverage;
- group/leakage guarantees expressed by BioSieve memberships survive ingestion unchanged;
- sample weights are propagated only to estimators that explicitly support them;
- deterministic seeds reproduce BioSieve/fitted-model behavior where the underlying components permit it.

---

## Phase 5 — Hyperparameter optimization 2.0

**Status:** closed/frozen after repository-side validation.

### Objective

Unify search spaces and optimizer behavior without turning `mlcore` into AutoML.

### Work

- typed search-space primitives: categorical, integer, float, log-float, while preserving legacy finite lists;
- common translation to Grid, Random, Halving, and Optuna backends;
- partition-driven tuning over the same leakage-safe Pipeline contract used by validation;
- already-partitioned data are consumed exactly; unpartitioned tuning delegates partition generation to BioSieve;
- holdout validation membership is used for search by default while a separate final test set remains protected;
- multi-metric scoring with explicit refit metric;
- group guarantees are inherited from BioSieve/PartitionPlan and sample weights are propagated only where supported;
- deterministic Optuna sampler defaults when requested;
- trial/fold failure representation;
- resource controls and consistent `n_jobs` behavior;
- optimization history with explicit failed-candidate/trial state plus DataFrame export;
- optional Optuna persistence/resume with deterministic seeded continuation where appropriate.

### Exit gates

- one logical search space can be consumed by all supported optimizers where mathematically applicable;
- optimizer results use the same estimator/pipeline contract as validation/direct model construction;
- protected final-test membership is excluded from search/refit when validation membership exists;
- failures and non-finite objectives are explicit;
- sample-weight and multi-metric contracts are covered by integration tests;
- seeded Random/Optuna workflows are reproducible and persisted Optuna studies can resume.

---

## Phase 6 — Benchmark engine

**Status:** CLOSED / FROZEN after repository-side validation.

### Objective

Make systematic algorithm/representation/partition comparisons a core scientific workflow.

### Work

- `BenchmarkResult` contract;
- baseline model support with `DummyClassifier` and `DummyRegressor`;
- multiple algorithms over one or more partition plans;
- tuned and untuned benchmark modes;
- repeated-run/seed support;
- long-form result tables;
- prediction and OOF retention;
- runtime and failure tracking;
- representation labels/metadata without representation generation;
- exportable benchmark tables suitable for downstream statistics/visualization;
- deterministic run/configuration identifiers linking metric rows to sample-level predictions;
- representation datasets must share sample identity and targets before partition memberships can be reused;
- partition plans may be rebound across representations only when they originate from one benchmark representation (or are explicit fingerprint-free membership artifacts);
- tuned benchmark reporting is restricted to train/validation/protected-test holdouts; CV tuning without an external test is not reported as unbiased performance and requires nested CV in a later workflow.

### Exit gates

- benchmark results are deterministic when inputs/estimators are deterministic;
- failed algorithms do not invalidate successful runs;
- every score can be traced to algorithm, split/fold, seed, configuration, and sample predictions;
- baseline comparison is available for classification and regression.

---

## Phase 7 — Persistence and reproducibility

**Status:** CLOSED / FROZEN after repository-side validation.

### Objective

Persist fitted pipelines and their scientific provenance so future inference can be validated.

### Work

- versioned artifact schema;
- fitted pipeline/model persistence;
- training configuration;
- feature schema;
- class order and positive-class metadata;
- dataset/partition fingerprints;
- metrics and optimization history;
- dependency/environment versions;
- checksums and artifact verification;
- compatibility warnings/errors;
- round-trip prediction tests in a fresh Python process.

### Exit gates

- save/load preserves predictions and probabilities;
- feature-schema mismatches are detected;
- corrupted artifacts are rejected;
- manifest/checksum validation passes;
- artifact schema is versioned independently from package internals.

---

## Phase 8 — Public API, YAML, and CLI

**Status:** implemented in the Phase 8 delivery; awaiting repository-side validation.

### Objective

Expose one stable workflow through multiple thin interfaces without duplicating business logic.

### Work

- high-level Python API for train, optimize, validate/evaluate, benchmark, save/load, and predict;
- validated YAML schemas/config parsing;
- CLI wrapping the same Python API;
- model discovery commands;
- artifact inspection/verification commands;
- consistent errors and exit codes;
- config serialization for reproducible reruns.

### Exit gates

- equivalent Python/YAML/CLI workflows produce equivalent configurations and results;
- CLI contains no independent training logic;
- representative classification and regression integration workflows pass end-to-end.

---

## Phase 9 — Documentation, examples, CI, and release engineering

### Objective

Make the library installable, teachable, testable, and releasable without local repository assumptions.

### Work

- complete README and usage documentation;
- classification, multiclass, regression, tuning, external-fold, benchmarking, persistence, and CLI examples;
- notebook smoke tests where practical;
- Python 3.11–3.13 CI matrix when dependency support permits;
- core-only and all-extras installation tests;
- lint/type/build gates;
- wheel/sdist installation tests;
- clean-environment import and CLI smoke tests;
- package metadata and release checklist.

### Exit gates

- wheel and sdist build/install in clean environments;
- core-only install works without optional packages;
- all documented quickstarts are executable;
- CI passes from a clean checkout.

---

## Phase 10 — Scientific torture testing and stable freeze

### Objective

Challenge the complete system with realistic and adversarial supervised-learning scenarios before declaring a stable release.

### Test matrix

- binary and multiclass classification;
- single-target regression;
- balanced and highly imbalanced targets;
- integer and string class labels;
- small-n and high-dimensional data;
- missing values and constant features where compatible;
- dense and supported sparse representations;
- scikit-learn, XGBoost, and LightGBM providers;
- holdout, CV, group CV, predefined/external folds;
- raw and preprocessed workflows;
- tuned and untuned workflows;
- benchmark workflows;
- persistence round trips;
- Python API/YAML/CLI parity.

### Failure matrix

- invalid metric/task pair;
- unknown model;
- unavailable optional provider;
- overlapping partitions;
- malformed fold assignments;
- incompatible feature schema;
- corrupted artifact;
- non-finite optimization objective;
- probability request for unsupported estimator;
- missing/ambiguous class semantics.

### Stable-release gate

A stable release is allowed only after the scientific torture suite, clean-environment packaging gates, documentation workflows, and artifact round-trip tests all pass. Stability is **not** tied to a fixed number of development days.
