# Saber examples

Runnable marimo notebooks show complete `saber` workflows end to end. Each
example is an advanced executable demonstration, not a screenshot: it builds
a dataset, runs a public `saber` workflow, and reports/plots the results with
external libraries. Plotting and reporting stay outside the `saber` core.

## Catalog

| Example | Workflow | What it shows |
|---|---|---|
| [01_binary_classification.py](01_binary_classification.py) | `validate` | imbalanced binary validation, sample weights, extended metrics, OOF threshold diagnostics and error audit |
| [02_multiclass_classification.py](02_multiclass_classification.py) | `validate` | multiclass metric semantics, class-level report and fold stability |
| [03_model_comparison.py](03_model_comparison.py) | `benchmark` | representation × model × seed benchmark with rankings and runtime |
| [04_regression_workflow.py](04_regression_workflow.py) | `validate` | missing values, leakage-safe preprocessing, extended regression metrics and residual/error analysis |
| [05_hyperparameter_optimization.py](05_hyperparameter_optimization.py) | `tune` | grid search, multi-metric candidate report, untuned vs tuned protected test |
| [06_optuna_optimization.py](06_optuna_optimization.py) | `tune` | typed continuous spaces, convergence and trial diagnostics |
| [07_optimizer_comparison.py](07_optimizer_comparison.py) | `tune` | Grid vs Random vs Optuna on identical memberships and search domain |
| [08_biosieve_validation.py](08_biosieve_validation.py) | `validate` | live BioSieve integration when installed, provenance and fold diagnostics |
| [09_partition_strategy_comparison.py](09_partition_strategy_comparison.py) | `benchmark` | balanced vs group-blocked external partition regimes without reimplementing splitting in saber |
| [10_multi_representation_benchmark.py](10_multi_representation_benchmark.py) | `benchmark` | representations × partitions × algorithms × seeds benchmark matrix |
| [11_benchmark_reporting.py](11_benchmark_reporting.py) | `benchmark` | leaderboard, stability, sample-level error audit and CSV/Markdown report export |
| [12_model_persistence.py](12_model_persistence.py) | `train` / `save_model` / `load_model` | auditable artifact, checksum verification, reload and inference report |
| [13_data_centric_benchmark.py](13_data_centric_benchmark.py) | `benchmark` | representation comparison with untuned/tuned models and a protected final test |

## Configs

`examples/configs/` holds minimal example YAML configs used by the CLI docs
and `saber config validate`.

### Test mode

Each example reads `SABER_DEMO_TEST=1` to reduce dataset sizes/trial counts
while exercising the same workflows. Running an example normally (unset)
uses the fuller demonstration settings.

## Running

Each example is a [marimo](https://marimo.io) notebook stored as a plain
Python file. Run it as a script, or open it interactively:

```bash
uv sync --all-extras --group examples
MPLBACKEND=Agg uv run python examples/01_binary_classification.py   # script
uv run marimo edit examples/01_binary_classification.py             # notebook
```

Run the complete suite with `bash examples/run_ci_examples.sh`.

## Recommended learning paths

**First validation workflow**

1. [01_binary_classification.py](01_binary_classification.py)
2. [08_biosieve_validation.py](08_biosieve_validation.py)
3. [05_hyperparameter_optimization.py](05_hyperparameter_optimization.py)
4. [12_model_persistence.py](12_model_persistence.py)

**Data-centric benchmarking**

1. [09_partition_strategy_comparison.py](09_partition_strategy_comparison.py)
2. [03_model_comparison.py](03_model_comparison.py)
3. [10_multi_representation_benchmark.py](10_multi_representation_benchmark.py)
4. [11_benchmark_reporting.py](11_benchmark_reporting.py)
5. [13_data_centric_benchmark.py](13_data_centric_benchmark.py)

For conceptual documentation, start at [`../docs/README.md`](../docs/README.md).
