# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

**Saber** is a domain-agnostic Python library for **classical supervised
machine learning** (classification and regression) on numerical tabular
feature matrices. It sits alongside three sibling libraries from the same
lab, each with a distinct scope:

- **Roxy** — classical protein sequence descriptors
- **Sylphy** — sequence encoders, embeddings and dimensionality reductions
- **Ruddy** — statistical exploratory data analysis (EDA)

Saber does **not** duplicate any of that functionality. It consumes prepared
numerical features — a Roxy descriptor table, a Sylphy embedding, or any
other numeric matrix — through its own `DatasetBundle` contract; representation
generation stays upstream. Unlike Ruddy, Saber uses **BioSieve** as an
optional extra (`saberlib[biosieve]`) to generate partitions when a dataset is
not already split, and it does not reimplement splitters of its own.

This project uses `uv` and `taskipy` — see `DEVELOPMENT.md` for setup and
task commands.

## Package Layout

The package is a **flat layout** at `saber/`. `tests/` mostly mirrors it one
directory per package, plus the cross-cutting suites `integration/`,
`robustness/`, and `examples/`.

## Development Workflow

Package management is `uv`; task running is `taskipy`. See `DEVELOPMENT.md`
for the full command list (`uv sync --all-extras`, `uv run task
format|lint|lint-fix|test|test-v|test-cov|pyrefly`).

## Scientific Invariants (read before touching any scientific module)

These are the most important rules for an agent working in this repository.

- **Leakage-safe preprocessing.** Imputation and scaling are fitted inside
  each training fold, never on the full dataset before folds are created.
- **BioSieve is the only partition-generation engine.** Saber never
  reimplements KFold/StratifiedKFold/GroupKFold or redundancy reduction; it
  either consumes an explicit `PartitionPlan` or delegates split generation to
  BioSieve.
- **Protected test.** Ordinary CV cannot serve as both the hyperparameter
  selection data and the unbiased final performance report. Tuned/benchmark
  workflows select on train/validation, refit, and evaluate only on a
  protected final test.
- **Explicit `positive_class`.** A two-column probability matrix is never
  interpreted by column position alone; the positive class is resolved
  through the fitted estimator's class order.
- **Fingerprints and versioned artifacts.** Dataset/partition fingerprints and
  the artifact schema version travel with every persisted model; checksums
  are verified before deserialization.
- **NumPy is the boundary towards estimators.** `saber` accepts NumPy arrays,
  Polars DataFrames, and pandas DataFrames as dataset input, but no estimator
  (scikit-learn, XGBoost, LightGBM) ever receives a DataFrame directly.
- **Polars null, not NaN, until the NumPy boundary.** Missing values stay as
  Polars null through the tabular layer and only become `NaN` at the final
  conversion to NumPy; they are never silently imputed before that point.

**Feature freeze until the first release.** The set of scientific methods
(algorithms, metrics, optimizers) is closed: do not add new ones until the
release. The code itself is not frozen — refactors, cleanups and fixes are
welcome as long as the invariants above hold and the test suite passes. A
change that alters numerical results needs a test that pins the new behavior.

## Core vs. Visualization

The scientific core (`saber/`) has no dependency on plotting libraries
(Matplotlib, Plotly, etc.) and must never import one. It returns structured,
traceable result objects only. All visualization lives externally, in the
marimo examples under `examples/`, which consume those result objects — they
never recompute statistics themselves.
