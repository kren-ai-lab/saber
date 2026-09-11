# mlcore Scope Contract

## Purpose

`mlcore` is a domain-agnostic library for **classical supervised machine learning** on numerical tabular feature matrices. It provides a common execution layer for model discovery, training, validation, hyperparameter optimization, benchmarking, persistence, and reproducible inference.

The library consumes already constructed features. It does not know whether those features came from protein embeddings, molecular descriptors, fingerprints, one-hot encodings, experimental measurements, PCA coordinates, or another upstream representation system.

## Canonical identity

- Distribution name: `mlcore`
- Python import package: `mlcore`
- Canonical project name in documentation: **mlcore**
- Current repository directory names are not part of the public API and may be renamed independently.

## Supported learning problems

The stable core targets only:

- binary classification;
- multiclass classification;
- single-target regression.

The canonical task vocabulary is therefore exactly:

```text
classification
regression
```

## Input contract

The core accepts numerical tabular feature matrices represented by NumPy arrays or pandas DataFrames. `DatasetBundle` formalizes sample identifiers, feature names, groups, sample weights, and metadata, while `PartitionPlan` preserves explicit holdout/fold membership by sample identity.

Upstream libraries remain responsible for representation generation and domain-specific preprocessing.

## In scope

- scikit-learn compatible classical estimators;
- XGBoost and LightGBM as optional estimator providers;
- deterministic training where supported by the estimator;
- holdout and cross-validation workflows;
- externally supplied partitions and predefined folds;
- leakage-safe preprocessing inside model-selection folds;
- classification and regression metrics;
- hyperparameter optimization;
- reproducible benchmarking across algorithms and partitions;
- out-of-fold predictions;
- model/pipeline persistence and reproducible inference;
- Python API, YAML configuration, and CLI surfaces built on one execution engine.

## Explicitly out of scope

The core does not implement:

- neural networks or deep learning;
- transfer learning, fine-tuning, zero-shot, or few-shot learning;
- representation learning;
- positive-unlabeled learning;
- multilabel classification;
- multi-output classification or regression;
- ordinal classification;
- AutoML systems such as TPOT;
- molecular, chemical, protein, peptide, or sequence-specific feature generation;
- domain-specific cleaning, clustering, or similarity calculations;
- SHAP, LIME, or a general explainability subsystem;
- dashboards, web interfaces, or hosted services;
- experiment tracking platforms.

These exclusions are architectural boundaries, not missing features for the first stable release.

## Dependency policy

The mandatory installation remains intentionally small:

```text
numpy
pandas
scipy
scikit-learn
joblib
rich
```

Optional providers/features are installed explicitly:

```text
mlcore[xgboost]
mlcore[lightgbm]
mlcore[optuna]
mlcore[all]
```

Importing `mlcore` must succeed without any optional dependency installed. Optional estimator providers register only when their Python package is available.

## Scientific design principles

1. **One execution path.** Training, validation, tuning, benchmarking, and inference must eventually build estimators through the same canonical factory/pipeline path.
2. **No hidden leakage.** Any data-dependent preprocessing used during model selection must be fitted inside each training fold.
3. **Explicit partitions.** Externally supplied splits are first-class inputs and are never silently regenerated.
4. **Structured results.** Metrics, predictions, probabilities, fold assignments, runtime, configuration, and provenance are reusable data, not only printed output.
5. **Reproducibility by construction.** Seeds, versions, schemas, parameters, and partition identity are persisted when relevant.
6. **Domain agnosticism.** The core operates on supervised numerical data without embedding biological or chemical assumptions.
7. **Minimal magic.** Defaults may be convenient, but all scientifically relevant behavior must remain inspectable and configurable.
