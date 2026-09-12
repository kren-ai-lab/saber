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

**Status:** CLOSED / FROZEN after repository-side validation.

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

## Phase 9 — Scientific torture testing and robustness

**Status:** CLOSED / SCIENTIFICALLY FROZEN after repository-side validation.

### Objective

Challenge the frozen scientific architecture with realistic, adversarial, and cross-layer supervised-learning scenarios before building demos or public documentation around it.

### Test matrix

- binary and multiclass classification;
- single-target regression;
- string, integer, boolean, and continuous-target rejection semantics;
- small-n and high-dimensional p >> n data;
- missing values, constant features, outliers, non-negative-feature constraints, and positive-target constraints;
- broad sklearn estimator-family smoke matrix plus XGBoost and LightGBM provider workflows;
- explicit/predefined partitions and BioSieve adapter integration;
- incomplete/repeated held-out coverage and OOF contracts;
- fold-local preprocessing and protected-test behavior;
- Grid, Random, Halving Grid, Halving Random, and Optuna tuning;
- typed search spaces, multi-metric selection, refit=False, partial candidate failure, and total search failure;
- multi-representation benchmarking, baselines, repeated seeds, tuned/untuned workflows, and failure isolation;
- model/benchmark persistence, integrity corruption, feature-schema mismatch, and fresh-process inference;
- Python API, YAML/JSON, and CLI parity.

### Robustness fixes discovered by Phase 9

- continuous targets are rejected at the classification dataset boundary instead of being misclassified as multiclass;
- classification evaluation uses fitted class semantics rather than only the classes observed in one held-out fold;
- requested classification metrics must be valid and computable instead of disappearing silently;
- tuning normalizes incompatible search-space and all-candidate-failure errors into mlcore domain contracts;
- validation and tuning reject scientifically invalid training memberships, such as single-class classification folds, before estimator fitting/search.

### Exit gates

- the full pre-Phase-9 suite remains green;
- all Phase 9 torture tests pass;
- Phase 9 introduces no additional uncaptured warning classes;
- compileall passes;
- no new ML feature path or splitter implementation is introduced;
- scientific behavior is frozen after Phase 9 except for confirmed bug fixes.

---

## Phase 10 — Executable notebook demos and demo-driven robustness

**Status: CLOSED / FROZEN.**

### Objective

Exercise the scientifically frozen workflows through realistic, executable notebooks that double as user demos and integration gates. Visualization and reporting stay outside the mlcore core package. Confirmed bugs discovered by demos may still be fixed, but Phase 10 does not introduce a parallel ML execution engine.

### Advanced demo inventory

Thirteen executable notebooks now cover:

- imbalanced binary classification with sample weights, twelve evaluation metrics, OOF ranking/calibration diagnostics, threshold sensitivity, and sample-level error audit;
- multiclass classification with canonical weighted metrics plus explicit macro/micro metrics, class-level reports, probabilities, and fold stability;
- representation × model × seed benchmark with dummy baseline, rankings, stability, runtime, and prediction provenance;
- regression with missing values, fold-safe imputation/scaling, seven regression metrics, residual structure, target-stratified errors, and worst-sample audit;
- protected-test grid optimization with multi-metric search, candidate ranking, and untuned/tuned final comparison;
- typed Optuna optimization with continuous/log search spaces, convergence and trial diagnostics;
- Grid vs Random vs Optuna comparison on identical memberships and search domain;
- live BioSieve validation when installed, including partition provenance, fold balance, size, and metric diagnostics;
- comparison of externally prepared balanced and group-blocked partition regimes without reimplementing split generation inside mlcore;
- multi-representation × multi-partition × algorithm × seed benchmark matrix with scenario-specific rankings;
- benchmark reporting with leaderboard, stability, runtime, sample-level error audit, and portable CSV/Markdown outputs;
- model persistence, checksum/manifest inspection, strict feature-schema inference, reload, structured prediction, and post-load evaluation;
- end-to-end data-centric study combining prepared representations, hyperparameter tuning, untuned/tuned models, repeated seeds, and a protected final test.

### Demo execution modes

- normal notebook execution uses the fuller datasets, seeds, candidate counts, and Optuna trials;
- the test suite sets `MLCORE_DEMO_TEST=1` to reduce compute while exercising the same workflow graph;
- both modes are executed with a headless matplotlib backend and warnings treated as errors during validation.

### Demo-driven gates

- all thirteen notebooks execute from top to bottom in clean subprocesses;
- all thirteen notebooks were additionally executed successfully in full normal mode;
- every demo contains explicit runtime assertions (`DEMO_CHECKS`);
- notebook contracts require substantive markdown/code content rather than minimal smoke examples;
- notebooks contain no legacy Trainer/optimizer execution path;
- notebooks do not reintroduce sklearn split-generation logic owned by BioSieve;
- plotting/reporting consumes public structured mlcore outputs;
- notebooks are stored without execution outputs/counts;
- no plotting/reporting dependency is added to the mlcore core requirements.

### Confirmed robustness fixes discovered by Phase 10

- canonical multiclass `precision`, `recall`, and `f1` evaluate with the weighted semantics already promised by `MetricSpec`, while explicit `*_macro`, `*_micro`, and `*_weighted` metrics remain available;
- benchmark matrices that mix `untuned` and `tuned` modes on a train/validation/protected-test holdout now report baseline, untuned, and tuned runs on the **same protected final test**. Untuned/baseline models are refit on train+validation before final test evaluation, preventing mixed validation/test scores in one leaderboard;
- notebook inference examples explicitly honor persisted feature-schema names/order instead of bypassing the artifact contract;
- SVC demos no longer request deprecated `probability=True`; ranking metrics use `decision_function` unless explicit calibration is scientifically required.

### Exit gates

- the complete pre-Phase-10 scientific suite remains green;
- the expanded suite totals 484 effective tests (471 non-notebook tests + 13 clean-subprocess notebook executions) with only the 14 pre-existing warning instances;
- all notebook contract, advanced-content, regression, and subprocess-execution tests pass;
- all thirteen notebooks execute successfully in smoke and full normal modes and generate figures/reports;
- compileall passes;
- Phase 10 introduces no core visualization framework, split-generation fallback, or alternate execution engine.

---

## Phase 11 — CLI 2.0

**Status: implemented; awaiting repository-side validation.**

### Objective

Improve the user-facing command-line experience without introducing another execution engine.

### Work

- richer command hierarchy, examples, and contextual help;
- human-oriented execution plans, progress/status output, completion panels, metric tables, and output-path summaries;
- machine-oriented `--json`, quiet, no-progress, and dry-run modes for workflow commands;
- `models search`, capability/requirement-aware model tables, and richer model detail views;
- improved artifact inspection/verification and config inspection;
- `mlcore doctor` for runtime and optional-provider availability;
- strict delegation to `WorkflowConfig`, `run_config`, registry, and persistence APIs.

### Exit gates

- CLI/API configuration parity remains exact;
- CLI contains no independent ML, splitting, tuning, benchmark, or persistence engine;
- dry-run validates without executing or creating workflow outputs;
- JSON mode remains machine-readable and decoration-free;
- representative train/validate/tune/benchmark/predict, model discovery, artifact, config, and doctor commands are covered by subprocess/in-process tests;
- CLI failures preserve the Phase 8 exit-code contract;
- the pre-Phase-11 scientific and notebook suites remain green.

---

## Phase 12 — Project cleanup and pruning

### Objective

Remove migration debris, dead code, duplicated tests, stale placeholders, and outdated project material after the scientific behavior is frozen.

### Work

- identify and remove obsolete/duplicated tests while preserving scientific coverage;
- remove stale roadmap fragments and temporary migration material;
- review and remove legacy execution compatibility layers when no longer required;
- remove dead imports, empty modules/directories, stale examples, and unused aliases;
- ensure the repository tree reflects the final architecture.

### Exit gates

- the reduced suite still passes every Phase 9 scientific contract;
- no public documented API is removed accidentally;
- project tree contains no known placeholder or duplicated execution path.

---

## Phase 13 — Full documentation and README

### Objective

Document the frozen product rather than documenting a moving implementation.

### Work

- complete README and quick start;
- scope and limitations;
- dataset and feature-schema contracts;
- BioSieve partition integration;
- preprocessing, validation, evaluation, tuning, benchmarking, and persistence;
- Python API;
- YAML/JSON configuration;
- CLI 2.0;
- artifact formats and reproducibility;
- links to the final notebook demos.

### Exit gates

- documented workflows match executable behavior;
- no stale legacy API examples remain;
- README provides a complete first-use path.

---

## Phase 14 — Diego handoff package

### Objective

Prepare a constrained engineering handoff for packaging/distribution hardening and migration of the dataframe/tabular implementation to Polars.

### Diego scope

- packaging and distribution hardening;
- wheel/sdist and clean-install validation;
- release metadata/versioning support;
- dataframe/tabular-layer migration from pandas to Polars where appropriate;
- performance/memory validation of that migration.

### Frozen areas that must not be redesigned

- supervised task semantics;
- metric/prediction contracts;
- BioSieve partition ownership;
- leakage-safe preprocessing semantics;
- tuning/benchmark scientific behavior;
- persistence provenance/integrity contracts;
- public workflow behavior.

### Handoff package

- current architecture and package map;
- frozen scientific contracts;
- supported workflows;
- dependency map;
- pandas usage inventory and Polars migration targets;
- modules/behaviors that must not change;
- packaging/distribution task list;
- validation gates required after migration;
- known warnings and limitations;
- release-candidate checklist.

### Exit gate

Diego can perform packaging/Polars hardening without needing to infer or redesign the scientific architecture.

---

## Release candidate gate

A release candidate is considered only after Phases 9–14 are complete and the post-handoff validation suite passes against the packaged artifact.
