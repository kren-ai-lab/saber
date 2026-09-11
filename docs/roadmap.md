# Five-Day Development Roadmap for the Stable Release

## 1. Release objective

The objective is to produce a stable and fully functional release of the classical machine learning library within five intensive development days.

The target release will be:

```text
v0.2.0
```

This will be a stable release suitable for publication on PyPI. It will not be marked as alpha, beta, release candidate, or pre-release.

The release must support the complete machine learning workflow for classification and regression:

```text
Dataset ingestion
    ↓
Dataset validation
    ↓
Partition generation or loading
    ↓
Model training or optimization
    ↓
Model evaluation
    ↓
Metric, curve, and figure generation
    ↓
Artifact persistence
    ↓
Artifact loading
    ↓
Prediction on new data
```

The complete workflow must be available through:

* a Python API;
* a command-line interface;
* YAML configuration files.

---

# 2. Scope of the stable release

## Included

The stable release will include:

* classification and regression;
* the existing algorithm registry;
* scikit-learn, XGBoost, and LightGBM backends;
* model training;
* hyperparameter optimization;
* internal data partitioning;
* externally supplied partitions;
* train, validation, and test workflows;
* cross-validation;
* predefined folds;
* compatibility with BioSieve partition files;
* classification and regression metrics;
* diagnostic curves and plots;
* exportable evaluation reports;
* model and pipeline persistence;
* artifact integrity validation;
* prediction on new data;
* Python API;
* CLI;
* YAML configuration;
* testing;
* documentation;
* packaging and publication readiness.

## Excluded

The release will explicitly exclude:

* explainable artificial intelligence;
* SHAP;
* LIME;
* model interpretation;
* web interfaces;
* dashboards;
* graphical applications;
* deep learning;
* data acquisition;
* biological sequence cleaning;
* biological clustering implemented inside this library;
* model serving through HTTP;
* TPOT or general AutoML;
* experiment tracking platforms;
* advanced statistical benchmarking across repeated experiments.

These capabilities belong to other libraries or later ecosystem components.

---

# 3. Definition of a fully functional release

The release will be considered complete only when all of the following workflows operate successfully.

## Classification workflow

```python
dataset = load_dataset(
    path="classification.csv",
    target="label",
    sample_id="sample_id",
)

result = train(
    dataset=dataset,
    task="classification",
    algorithm="random_forest",
    partition_strategy="stratified",
    test_size=0.20,
    validation_size=0.10,
    metrics=["mcc", "f1", "roc_auc"],
)

evaluation = evaluate(
    result=result,
    output_dir="results/classification",
)

save_artifact(
    result=result,
    path="artifacts/classifier",
)

model = load_artifact("artifacts/classifier")
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
```

## Regression workflow

```python
dataset = load_dataset(
    path="regression.csv",
    target="target",
    sample_id="sample_id",
)

result = optimize(
    dataset=dataset,
    task="regression",
    algorithm="random_forest_regressor",
    method="optuna",
    primary_metric="rmse",
    n_trials=30,
)

evaluation = evaluate(
    result=result,
    output_dir="results/regression",
)

save_artifact(
    result=result,
    path="artifacts/regressor",
)

model = load_artifact("artifacts/regressor")
predictions = model.predict(new_data)
```

## CLI workflow

```bash
mlcore train --config examples/classification.yaml
mlcore tune --config examples/regression_optimization.yaml
mlcore evaluate --artifact artifacts/classifier
mlcore predict --artifact artifacts/classifier --data new_data.csv
mlcore artifact verify artifacts/classifier
```

All these workflows must be covered by automated integration tests.

---

# 4. Day 1 — Core stabilization

## Objective

Resolve every critical defect identified during the previous technical audit.

No new high-level functionality should be developed until the core is stable.

## 4.1. Metrics registry

Implement a formal `MetricSpec` contract:

```python
MetricSpec(
    name="mcc",
    task="classification",
    greater_is_better=True,
    response_method="predict",
    aliases=["matthews_corrcoef"],
    supports_binary=True,
    supports_multiclass=True,
)
```

Each metric must define:

* canonical name;
* aliases;
* compatible task;
* optimization direction;
* required prediction output;
* binary support;
* multiclass support;
* averaging options;
* display name.

## 4.2. Classification metrics

Implement and validate:

* accuracy;
* balanced accuracy;
* precision;
* recall;
* specificity;
* F1 score;
* Matthews correlation coefficient;
* Cohen’s kappa;
* ROC AUC;
* average precision;
* log loss;
* Brier score.

## 4.3. Regression metrics

Implement and validate:

* MAE;
* MSE;
* RMSE;
* median absolute error;
* R²;
* explained variance;
* MAPE;
* Pearson correlation;
* Spearman correlation;
* Kendall correlation.

## 4.4. Critical corrections

Complete all previous audit corrections:

* replace deprecated ROC AUC scorer arguments;
* validate metric and task compatibility;
* reject unknown metrics;
* reject non-finite optimization results;
* prevent searches from returning arbitrary parameters after failed folds;
* support `refit=False`;
* separate internal optimization values from natural metric values;
* handle binary and multiclass probabilities;
* preserve class order;
* preserve the positive class;
* support integer and string labels;
* populate `TrainResult.metrics`;
* remove duplicated prediction generation;
* propagate `random_state`;
* use `n_jobs=1` as a safe default;
* clean algorithm aliases;
* normalize registry tags;
* improve registry error messages.

## 4.5. Optimization result contract

Finalize:

```python
OptimizationResult(
    metric="rmse",
    objective_value=-82.34,
    best_score=82.34,
    greater_is_better=False,
    best_params={...},
    best_model=model,
    refitted=True,
    cv_results={...},
    metadata={...},
)
```

## 4.6. Optuna

For the stable release, Optuna must support:

* categorical parameters;
* integer parameters;
* float parameters;
* logarithmic scales;
* deterministic sampler seeds;
* minimize and maximize directions;
* exportable trial history;
* failed-trial handling.

Conditional search spaces and pruning may be included only when they can be completed without destabilizing the release.

## Day 1 acceptance criteria

* all core unit tests pass;
* classification and regression metrics return finite values;
* incompatible metrics fail before training;
* ROC AUC works for binary classification;
* multiclass probabilities are handled;
* RMSE and MSE display natural positive values;
* `refit=False` returns a valid result;
* LightGBM and XGBoost run using controlled resources.

---

# 5. Day 2 — Data contracts and partitioning

## Objective

Establish the formal contracts required by training, optimization, persistence, and BioSieve interoperability.

## 5.1. `DatasetBundle`

Implement:

```python
DatasetBundle(
    X=X,
    y=y,
    sample_ids=sample_ids,
    feature_names=feature_names,
    groups=groups,
    metadata=metadata,
)
```

Validation must include:

* equal sample counts across inputs;
* unique sample identifiers;
* unique feature names;
* preserved feature order;
* target validation;
* missing-value detection;
* infinite-value detection;
* group alignment;
* sparse-matrix support where compatible;
* classification label validation;
* regression target validation.

## 5.2. Supported input formats

Support:

* NumPy arrays;
* pandas DataFrames;
* SciPy sparse matrices;
* CSV files;
* TSV files;
* Parquet files when pandas supports the required backend.

## 5.3. `PartitionPlan`

Implement:

```python
PartitionPlan(
    train_ids=[...],
    validation_ids=[...],
    test_ids=[...],
    folds=[...],
    strategy="stratified",
    source="internal",
    random_state=42,
    metadata={},
)
```

Validation must detect:

* overlapping train, validation, and test sets;
* repeated identifiers;
* unknown identifiers;
* missing identifiers;
* empty partitions;
* invalid folds;
* leakage between sets;
* inconsistent dataset fingerprints.

## 5.4. Internal partitioning

Implement:

* train/test split;
* train/validation/test split;
* K-fold cross-validation;
* stratified K-fold;
* repeated K-fold;
* repeated stratified K-fold;
* group K-fold;
* stratified group K-fold when available;
* predefined split.

## 5.5. Previously divided data

Support explicit inputs:

```python
train(
    X_train=X_train,
    y_train=y_train,
    X_validation=X_validation,
    y_validation=y_validation,
    X_test=X_test,
    y_test=y_test,
)
```

The library must never repartition these datasets silently.

## 5.6. BioSieve compatibility

Implement an interoperable partition format based on sample identifiers.

The library must consume files containing:

```yaml
partition_schema_version: "1.0"
source: biosieve
strategy: cluster_aware
dataset_fingerprint: "..."
sample_id_column: sample_id
random_state: 42

train_ids:
  - sample_001
  - sample_002

validation_ids:
  - sample_003

test_ids:
  - sample_004

folds:
  - fold: 0
    train_ids: [...]
    validation_ids: [...]
```

For this stable release, BioSieve will not be a mandatory dependency. The connection will operate through a stable file contract.

## 5.7. Leakage rules

The library must enforce:

* test data cannot participate in optimization;
* test data cannot participate in preprocessing fitting;
* test data cannot participate in threshold selection;
* validation data can be used only when explicitly configured;
* preprocessing must be fitted inside each training fold;
* external partitions must remain unchanged.

## Day 2 acceptance criteria

* internal splits work for classification and regression;
* predefined train/validation/test sets work;
* external partitions are joined using sample IDs;
* overlap and leakage are detected;
* partition plans can be exported and loaded;
* folds can be reused across different algorithms;
* partition fingerprints are reproducible.

---

# 6. Day 3 — Pipelines, persistence, and inference

## Objective

Persist a complete reproducible model pipeline and support safe prediction on new data.

## 6.1. Preprocessing pipelines

Use native scikit-learn components:

* `Pipeline`;
* `ColumnTransformer`;
* `SimpleImputer`;
* `StandardScaler`;
* `MinMaxScaler`;
* `RobustScaler`;
* `OneHotEncoder`.

The preprocessing system must support:

* numerical columns;
* categorical columns;
* missing-value handling;
* optional scaling;
* optional encoding;
* passthrough columns;
* user-provided pipelines.

Preprocessing must occur inside cross-validation to prevent leakage.

## 6.2. Artifact structure

```text
model_artifact/
├── model.joblib
├── manifest.json
├── training_config.yaml
├── dataset_schema.json
├── partition_plan.json
├── parameters.json
├── metrics.json
├── optimization.json
├── environment.json
├── evaluation/
└── checksums.sha256
```

## 6.3. Artifact manifest

The manifest must contain:

* artifact schema version;
* library version;
* creation timestamp;
* task;
* algorithm;
* backend;
* estimator class;
* preprocessing information;
* feature names;
* feature order;
* target name;
* class names;
* positive class;
* random seed;
* partition fingerprint;
* dataset fingerprint;
* primary metric;
* artifact files;
* file checksums.

## 6.4. Save and load

Implement:

```python
save_artifact(
    result,
    path="artifacts/model",
    overwrite=False,
)
```

```python
artifact = load_artifact(
    "artifacts/model",
    strict=True,
)
```

Strict loading must validate:

* required files;
* checksums;
* schema version;
* available backend;
* library compatibility;
* estimator compatibility;
* model integrity.

## 6.5. Prediction validation

Before inference, validate:

* missing features;
* unexpected features;
* feature order;
* duplicated features;
* incompatible data types;
* target column accidentally included;
* sample identifier preservation.

Prediction outputs must include:

```text
sample_id
prediction
probability_<class>
```

when probabilities are available.

## 6.6. Security

Documentation and runtime warnings must state that joblib and pickle artifacts must only be loaded from trusted sources.

## Day 3 acceptance criteria

* classification artifact roundtrip works;
* regression artifact roundtrip works;
* optimized model roundtrip works;
* predictions are identical before and after loading;
* probabilities are identical before and after loading;
* corrupted checksums are detected;
* invalid input schemas are rejected;
* complete preprocessing pipelines are preserved.

---

# 7. Day 4 — Evaluation, curves, and reporting

## Objective

Provide complete evaluation outputs for classification and regression.

## 7.1. Classification evaluation

Generate:

* global metrics;
* per-class metrics;
* classification report;
* confusion matrix;
* normalized confusion matrix;
* ROC curve;
* precision–recall curve;
* calibration curve;
* threshold-performance curve;
* learning curve;
* validation curve when a parameter is supplied;
* prediction distribution;
* probability distribution.

## 7.2. Binary classification

Support:

* explicit positive class;
* sensitivity;
* specificity;
* negative predictive value;
* positive predictive value;
* ROC AUC;
* average precision;
* threshold optimization using validation data.

## 7.3. Multiclass classification

Support:

* one-vs-rest ROC curves;
* one-vs-rest precision–recall curves;
* micro average;
* macro average;
* weighted average;
* per-class support;
* complete confusion matrix.

## 7.4. Regression evaluation

Generate:

* observed versus predicted plot;
* residuals versus predicted plot;
* residual distribution;
* absolute error distribution;
* relative error distribution;
* error by target range;
* learning curve;
* validation curve when configured;
* train-versus-validation comparison;
* fold-level metric distributions.

## 7.5. Export structure

Classification:

```text
evaluation/
├── metrics.json
├── metrics.csv
├── fold_metrics.csv
├── predictions.csv
├── probabilities.csv
├── confusion_matrix.csv
├── classification_report.csv
├── curves/
│   ├── roc_curve.csv
│   ├── precision_recall_curve.csv
│   ├── calibration_curve.csv
│   ├── threshold_metrics.csv
│   └── learning_curve.csv
└── figures/
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── calibration_curve.png
    ├── threshold_metrics.png
    └── learning_curve.png
```

Regression:

```text
evaluation/
├── metrics.json
├── metrics.csv
├── fold_metrics.csv
├── predictions.csv
├── residuals.csv
├── error_by_range.csv
├── curves/
│   └── learning_curve.csv
└── figures/
    ├── observed_vs_predicted.png
    ├── residuals_vs_predicted.png
    ├── residual_distribution.png
    ├── absolute_error_distribution.png
    ├── relative_error_distribution.png
    ├── error_by_range.png
    └── learning_curve.png
```

## 7.6. Figure formats

Support:

* PNG;
* SVG;
* PDF.

Every curve must export its underlying numerical data.

Figures must be generated using a non-interactive backend and must not require a graphical environment.

## Day 4 acceptance criteria

* binary classification reports work;
* multiclass classification reports work;
* regression reports work;
* curves produce finite data;
* all required figures are generated;
* figures can be disabled;
* curve data are independently reusable;
* reports can evaluate test sets and cross-validation predictions.

---

# 8. Day 5 — API, CLI, tests, documentation, and PyPI preparation

## Objective

Complete the public interfaces and validate the stable distribution.

## 8.1. Public Python API

Expose:

```python
from mlcore import (
    load_dataset,
    create_partitions,
    load_partitions,
    train,
    optimize,
    evaluate,
    predict,
    save_artifact,
    load_artifact,
)
```

Users must not need to import internal runners or backend-specific components.

## 8.2. CLI commands

Implement:

```bash
mlcore models list
mlcore models show <algorithm>

mlcore metrics list
mlcore metrics show <metric>

mlcore data validate --config data.yaml

mlcore partitions create --config partitions.yaml
mlcore partitions validate partitions.yaml
mlcore partitions inspect partitions.yaml

mlcore train --config training.yaml
mlcore tune --config optimization.yaml
mlcore evaluate --artifact artifacts/model
mlcore predict --artifact artifacts/model --data new_data.csv

mlcore artifact inspect artifacts/model
mlcore artifact verify artifacts/model
```

The CLI must call the public API. It must not contain independent training or evaluation implementations.

## 8.3. CLI behavior

The CLI must provide:

* clear help messages;
* configuration validation;
* readable errors;
* non-zero exit codes on failure;
* configurable verbosity;
* deterministic defaults;
* no mandatory interactive prompts;
* automatic output-directory creation;
* logs written to file when requested.

## 8.4. Test structure

```text
tests/
├── unit/
├── integration/
├── backends/
├── artifacts/
├── cli/
└── regression_tests/
```

Required test markers:

```text
unit
integration
slow
lightgbm
xgboost
optuna
cli
```

## 8.5. Mandatory integration tests

### Classification

* internal stratified split;
* external predefined split;
* optimization;
* report generation;
* artifact save/load;
* prediction parity;
* CLI execution.

### Regression

* internal split;
* external predefined split;
* optimization;
* report generation;
* artifact save/load;
* prediction parity;
* CLI execution.

### BioSieve interoperability

* load BioSieve-style partition;
* validate sample IDs;
* preserve folds;
* train multiple models using the same split;
* reject fingerprint mismatch.

## 8.6. Documentation

Complete:

* README;
* installation guide;
* quick start;
* classification guide;
* regression guide;
* optimization guide;
* persistence guide;
* CLI guide;
* partitioning guide;
* BioSieve interoperability guide;
* metric reference;
* model registry reference;
* artifact format documentation;
* troubleshooting guide.

## 8.7. Package identity

Before publication, decide and freeze:

* PyPI distribution name;
* Python import name;
* CLI executable name;
* repository name;
* public branding.

These names must not remain inconsistent.

## 8.8. Dependencies

Core dependencies should remain minimal.

Recommended core:

```toml
dependencies = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "pyyaml",
]
```

Optional extras:

```toml
[project.optional-dependencies]
xgboost = ["xgboost"]
lightgbm = ["lightgbm"]
optuna = ["optuna"]
all = ["xgboost", "lightgbm", "optuna"]
dev = ["pytest", "pytest-cov", "ruff", "mypy", "build", "twine"]
```

## 8.9. Release validation

The following commands must pass:

```bash
python -m pytest
python -m pytest -m "not slow"
python -m build
twine check dist/*
```

A clean environment must also pass:

```bash
pip install dist/*.whl
python -c "import mlcore"
mlcore --help
mlcore models list
mlcore metrics list
mlcore train --config examples/classification.yaml
mlcore train --config examples/regression.yaml
```

## Day 5 acceptance criteria

* API workflows pass;
* CLI workflows pass;
* wheel and source distribution build;
* package installs in a clean environment;
* examples complete successfully;
* documentation contains no empty sections;
* no public module is an empty placeholder;
* PyPI metadata are complete;
* the release is ready to publish as stable `v0.2.0`.

---

# 9. Release gates

The package must not be released if any of the following conditions remain.

## Core blockers

* ROC AUC produces invalid scores;
* metric/task mismatches are not rejected;
* optimization can return non-finite results;
* `refit=False` crashes;
* probabilities are interpreted incorrectly;
* score semantics remain ambiguous.

## Data blockers

* partitions can overlap without error;
* external partitions are joined by row position;
* test data can leak into training or optimization;
* feature order is not preserved.

## Persistence blockers

* predictions change after loading;
* artifact corruption is not detected;
* preprocessing is not persisted;
* feature schema is not validated.

## Evaluation blockers

* required figures fail for binary or multiclass classification;
* regression residual reports fail;
* curve data are not exportable;
* metrics silently return non-finite values.

## Distribution blockers

* test suite does not complete;
* wheel does not install cleanly;
* CLI does not start;
* documentation examples do not execute;
* package naming remains inconsistent.

---

# 10. Post-release roadmap

The following capabilities can be developed after the stable PyPI release without making the initial version incomplete:

## `v0.3.0`

* direct optional BioSieve Python adapter;
* nested cross-validation;
* benchmark runner;
* repeated-seed experiments;
* tuned versus untuned comparison;
* computational resource reports.

## `v0.4.0`

* advanced Optuna pruning;
* persistent Optuna studies;
* conditional search spaces;
* resume interrupted optimization;
* multi-objective optimization.

## `v0.5.0`

* model comparison reports;
* statistical comparison across repeated runs;
* extended provenance;
* tighter integration with MCS and BDRS artifacts.

These are extensions of a functional product, not corrections required to make the initial release usable.

---

# 11. Final development order

```text
Day 1
    Core stabilization and metrics

Day 2
    Dataset and partition contracts

Day 3
    Pipelines, persistence, loading, and inference

Day 4
    Evaluation, curves, plots, and reports

Day 5
    Public API, CLI, tests, documentation, and packaging
```

The development process must preserve one rule:

> Each day must finish with a working end-to-end state. New components must not break previously completed workflows.

The final result will be a stable classical machine learning library capable of training, optimizing, evaluating, persisting, loading, and applying classification and regression models through both Python and the command line.
