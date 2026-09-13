# mlcore

`mlcore` is a domain-agnostic framework for **classical supervised machine learning** on numerical tabular representations.

The project is currently in **Alpha** while release documentation and packaging are finalized.

## Scope

Supported targets are binary classification, multiclass classification, and single-target regression. The library deliberately excludes deep learning, representation learning, AutoML, domain-specific feature generation, and web/application layers.

See [`docs/scope.md`](docs/scope.md) for the formal scope contract and [`docs/architecture.md`](docs/architecture.md) for the current architecture.

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
