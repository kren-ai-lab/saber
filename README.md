# mlcore

`mlcore` is a domain-agnostic framework for **classical supervised machine learning** on numerical tabular representations.

The project is currently in **Alpha** while its execution, validation, benchmarking, persistence, and public API contracts are being consolidated.

## Scope

Supported targets are binary classification, multiclass classification, and single-target regression. The library deliberately excludes deep learning, representation learning, AutoML, domain-specific feature generation, and web/application layers.

See [`docs/scope.md`](docs/scope.md) for the formal scope contract and [`docs/architecture.md`](docs/architecture.md) for the architecture freeze.

## Installation model

Core installation:

```bash
pip install .
```

Optional estimator/optimization providers:

```bash
pip install '.[xgboost]'
pip install '.[lightgbm]'
pip install '.[optuna]'
pip install '.[all]'
```

Development installation:

```bash
pip install -e '.[dev]'
```

## Current implemented kernel

The existing codebase already contains the model registry, classical classification/regression estimators, a training engine, evaluation metrics, and Grid/Random/Halving/Optuna optimization foundations. The phased roadmap defines the work required to turn this kernel into a reproducible scientific library.

See [`docs/roadmap.md`](docs/roadmap.md).
