# Saber

[![PyPI](https://img.shields.io/pypi/v/saberlib?style=flat-square)](https://pypi.org/project/saberlib/)
[![PyVersions](https://img.shields.io/pypi/pyversions/saberlib?style=flat-square)](https://github.com/kren-ai-lab/saber)
[![Tests](https://img.shields.io/github/actions/workflow/status/kren-ai-lab/saber/tests.yml?style=flat-square)](https://github.com/kren-ai-lab/saber/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/kren-ai-lab/saber?style=flat-square)

Saber is a domain-agnostic framework for **classical supervised machine learning** on numerical tabular representations. It unifies model discovery, leakage-safe validation, BioSieve partitioning, hyperparameter optimization, benchmarking, persistence, configuration, and a polished CLI behind one scientific execution model.

---

## Why Saber?

A machine-learning result is more than `model.fit(X, y)`. Scientific workflows also need stable sample identity, explicit partitions, preprocessing that cannot leak across folds, comparable tuning and validation semantics, traceable benchmark runs, and artifacts that can be inspected and reused later.

Saber is built around those requirements.

| Principle | What it means in practice |
|---|---|
| **One execution path** | Training, validation, tuning, and benchmarking build estimators through the same registry/factory contract. |
| **Leakage-safe by construction** | Imputation and scaling are fitted inside each training fold. |
| **Explicit partitions** | Existing splits are preserved exactly; unpartitioned data are delegated to **BioSieve**. |
| **Structured results** | Metrics, OOF predictions, failures, runtimes, tuning histories, and provenance remain reusable data. |
| **Reproducible artifacts** | Feature schemas, versions, checksums, parameters, dataset/partition fingerprints, and environment metadata travel with persisted models. |
| **Domain agnostic** | `saber` consumes prepared numerical features; representation generation and redundancy reduction stay upstream. |

## What is in scope?

**Supported**

- binary classification;
- multiclass classification;
- single-target regression;
- NumPy arrays, Polars DataFrames, and pandas DataFrames;
- scikit-learn estimators plus optional XGBoost and LightGBM providers;
- explicit holdout/CV memberships and BioSieve-generated partitions;
- leakage-safe numerical preprocessing;
- Grid, Random, successive-halving, and Optuna optimization;
- multi-representation / multi-partition benchmarking;
- out-of-fold predictions and long-form result tables;
- auditable model and benchmark artifacts;
- Python API, YAML/JSON configuration, and CLI workflows.

**Deliberately out of scope**

Deep learning, representation learning, sequence/molecule-specific feature generation, redundancy reduction, AutoML, multilabel/multi-output learning, explainability frameworks, dashboards, and experiment-tracking services. See the formal [scope contract](docs/scope.md).

---

## Installation

Saber supports Python 3.11 to 3.14. Install the `saberlib` distribution:

```bash
pip install saberlib
# or
uv add saberlib
```

Optional capabilities are explicit:

```bash
pip install "saberlib[biosieve]"   # partition generation
pip install "saberlib[optuna]"     # Optuna tuning
pip install "saberlib[xgboost]"    # XGBoost provider
pip install "saberlib[lightgbm]"   # LightGBM provider
pip install "saberlib[all]"        # all optional runtime integrations
```

For development and executable notebooks, see [DEVELOPMENT.md](DEVELOPMENT.md):

```bash
uv sync --all-extras
```

Check the environment at any time:

```bash
saber doctor
```

---

## Quick start

The smallest scientifically explicit workflow is: create a `DatasetBundle`, provide a reproducible `PartitionPlan`, and call the high-level API.

```python
# docs-test: quickstart
import numpy as np
from sklearn.datasets import make_classification

import saber
from saber.datasets import DatasetBundle, PartitionPlan

X, y = make_classification(
    n_samples=120,
    n_features=12,
    n_informative=7,
    random_state=42,
)

sample_ids = [f"sample_{i}" for i in range(len(y))]
dataset = DatasetBundle(X=X, y=y, sample_ids=sample_ids)

# Example of an externally prepared 4-fold assignment.
folds = np.arange(len(y)) % 4
plan = PartitionPlan.from_predefined_folds(
    sample_ids=dataset.sample_ids,
    fold_assignments=folds,
    dataset_fingerprint=dataset.fingerprint,
)

result = saber.validate(
    dataset=dataset,
    algorithm="logistic_regression",
    partition_plan=plan,
    metrics=("mcc", "balanced_accuracy", "roc_auc"),
    random_state=42,
)

print(result.aggregate_metrics)
assert result.oof_prediction is not None
assert result.metadata["oof_complete"] is True
```

`X` can also be a Polars DataFrame, which is the tabular type Saber uses internally:

```python
import polars as pl

features = pl.DataFrame({f"f{i}": X[:, i] for i in range(X.shape[1])})
dataset = DatasetBundle(X=features, y=y, sample_ids=sample_ids)
```

Use `pl.read_csv(...)` to load a features file. pandas DataFrames are accepted too and are
converted to Polars once, at the boundary; pandas itself is not a runtime dependency.

If the dataset is **not partitioned**, delegate split generation to BioSieve:

```python
from saber.datasets import BioSievePartitionConfig

partitioning = BioSievePartitionConfig(
    strategy="stratified_kfold",
    params={"n_splits": 5, "seed": 42},
)

result = saber.validate(
    dataset=dataset,
    algorithm="random_forest",
    partitioning=partitioning,
    metrics=("mcc", "balanced_accuracy", "roc_auc"),
    random_state=42,
)
```

> Saber never substitutes its own splitter when BioSieve is requested. Redundancy reduction is also **not** performed by `saber`; if needed, it belongs upstream before `DatasetBundle` creation.

---

## One library, six public workflows

```text
prepared numerical data
        │
        ├── train      → final fitted model
        ├── validate   → held-out / OOF performance
        ├── evaluate   → metrics for predictions or persisted models
        ├── tune       → hyperparameter optimization
        ├── benchmark  → systematic experiment matrices
        └── predict    → reproducible artifact inference
```

The package root intentionally stays small:

```python
import saber

saber.train(...)
saber.validate(...)
saber.evaluate(...)
saber.tune(...)
saber.benchmark(...)
saber.predict(...)
```

`optimize` is an alias for `tune`.

---

## Hyperparameter optimization

Search spaces are backend-neutral and typed:

```python
from saber.core.search_space import SearchSpace, Categorical, Integer, LogFloat
from saber.tuning import TuningConfig

space = SearchSpace(
    name="svc",
    parameters={
        "C": LogFloat(1e-4, 1e2),
        "kernel": Categorical(["linear", "rbf"]),
    },
)

config = TuningConfig(
    optimizer="optuna",
    metrics=("mcc", "roc_auc"),
    refit_metric="mcc",
    n_trials=40,
    random_state=42,
)
```

Grid/Random/Halving/Optuna consume the same logical search-space contract. Preprocessing remains inside the searched pipeline, and tuning consumes the same explicit `PartitionPlan` used by validation.

Read [Hyperparameter optimization](docs/tuning.md).

---

## Benchmarking

`BenchmarkEngine` is for experiments where the scientific question is larger than “which classifier wins?”. It can cross:

```text
prepared representations
× partition scenarios
× algorithms
× random seeds
× tuned / untuned modes
```

and keeps every score connected to its run, split, configuration, runtime, and sample-level prediction.

```python
from saber.benchmark import BenchmarkConfig

benchmark = saber.benchmark(
    datasets={"representation_a": dataset_a, "representation_b": dataset_b},
    algorithms=("logistic_regression", "random_forest", "svc"),
    partitions={"cluster_disjoint": plan},
    config=BenchmarkConfig(
        metrics=("mcc", "balanced_accuracy"),
        seeds=(42, 123),
        modes=("untuned",),
        include_baselines=True,
    ),
)

leaderboard_data = benchmark.aggregate_metrics_frame()
predictions = benchmark.predictions_frame()
```

For tuned benchmarks, `saber` protects the final test membership and rejects ordinary-CV shortcuts that would report optimistically selected performance. Read [Benchmarking](docs/benchmarking.md).

---

## Persistence and reproducible inference

A model artifact is an auditable directory, not only a pickle:

```text
model_artifact/
├── manifest.json
├── model.joblib
├── environment.json
├── feature_schema.json
├── provenance.json
├── parameters.json
├── metrics.json
├── training_config.json
├── partition_plan.json      # when available
└── checksums.sha256
```

```python
artifact = saber.load_model("artifacts/final_model")
artifact.validate_features(X_new, feature_names=feature_names)
prediction = artifact.predict_result(
    X_new,
    feature_names=feature_names,
    sample_ids=sample_ids,
)
```

Checksums are verified before model deserialization. As with every joblib/pickle-based system, load artifacts only from trusted sources. Read [Persistence](docs/persistence.md).

---

## CLI 2.0

The CLI is a presentation layer over the same configuration/API engine:

```bash
saber --help
saber doctor
saber models list --task classification
saber models search forest --provider sklearn
saber models show random_forest
```

Run or inspect workflows:

```bash
saber validate experiment.yaml
saber benchmark study.yaml --dry-run
saber tune optimization.yaml --json
saber config show experiment.yaml
saber artifact verify artifacts/model
```

Human-readable Rich output is the default; `--json` provides machine-readable summaries for automation. Read the [CLI guide](docs/cli.md) and [configuration reference](docs/configuration.md).

---

## Executable scientific examples

The repository contains **13 advanced notebooks**. They are not screenshots: the test suite executes them in clean subprocesses, and each notebook contains assertions in addition to figures.

| Area | Example |
|---|---|
| Binary classification | imbalanced validation, 12 metrics, threshold diagnostics, error audit |
| Multiclass | macro/micro/weighted semantics, class reports, fold stability |
| Regression | leakage-safe imputation/scaling, residual analysis, difficult samples |
| Partitioning | BioSieve provenance and partition-strategy comparison |
| Optimization | Grid, Optuna, and optimizer comparison |
| Benchmarking | representation × partition × algorithm × seed matrices |
| Reporting | leaderboards, stability, runtimes, sample-level error audits |
| Persistence | checksum verification, reload, schema-safe inference |
| End-to-end | tuned/untuned multi-representation study with protected test |

Start at [`examples/README.md`](examples/README.md).

---

## Documentation map

| Document | Use it for |
|---|---|
| [Documentation index](docs/README.md) | complete map of the documentation |
| [Getting started](docs/getting_started.md) | first Python/API workflow |
| [Scope](docs/scope.md) | supported and intentionally unsupported problems |
| [Architecture](docs/architecture.md) | subsystem boundaries and invariants |
| [Data & partitions](docs/data_and_partitions.md) | `DatasetBundle`, `FeatureSchema`, `PartitionPlan`, BioSieve |
| [Metrics & evaluation](docs/metrics_and_evaluation.md) | metric semantics, probabilities, OOF results |
| [Tuning](docs/tuning.md) | optimizers, typed spaces, protected-test rules |
| [Benchmarking](docs/benchmarking.md) | experiment matrices and analysis-ready outputs |
| [Persistence](docs/persistence.md) | artifacts, checksums, schema-safe inference |
| [Configuration](docs/configuration.md) | YAML/JSON workflows |
| [CLI](docs/cli.md) | commands, JSON mode, dry runs, exit codes |
| [Python API](docs/api_reference.md) | high-level functions and important contracts |
| [Usage recipes](docs/usage.md) | compact workflow-oriented recipes |

---

## Scientific guarantees and non-guarantees

Saber helps enforce **workflow correctness**, not scientific validity by fiat. It can prevent overlap inside declared partitions, keep preprocessing inside folds, preserve sample identity, protect a final test in supported workflows, and retain provenance. It cannot decide whether a representation is biologically appropriate, whether a partition strategy answers the right scientific question, or whether the supplied data contain upstream leakage.

Use the partitioning strategy, metrics, and experimental design that match the scientific claim you want to make.

---

## Project status

Saber is currently **Alpha**. The scientific architecture, public workflows, advanced executable demos, and CLI are already extensively tested; packaging/distribution hardening remains separate from the scientific core.

Repository: <https://github.com/kren-ai-lab/saber>

---

## Citing and license

If you use Saber in published work, please cite it using
[`CITATION.cff`](https://github.com/kren-ai-lab/saber/blob/main/CITATION.cff), or the "Cite this repository" button on
GitHub. Saber is released under the [MIT license](https://github.com/kren-ai-lab/saber/blob/main/LICENSE).
