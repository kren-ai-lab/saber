# Saber executable examples

These notebooks are **advanced executable demonstrations**, not screenshots. They are executed in clean subprocesses by the notebook integration suite. Plotting/reporting stays outside the `saber` core.

## Classification
- `classification/01_binary_classification.ipynb` — imbalanced binary validation, sample weights, extended metrics, OOF threshold diagnostics and error audit.
- `classification/02_multiclass_classification.ipynb` — multiclass metric semantics, class-level report and fold stability.
- `classification/03_model_comparison.ipynb` — representation × model × seed benchmark with rankings and runtime.

## Regression
- `regression/01_regression_workflow.ipynb` — missing values, leakage-safe preprocessing, extended regression metrics and residual/error analysis.

## Validation and partitions
- `validation/01_biosieve_validation.ipynb` — live BioSieve integration when installed, provenance and fold diagnostics.
- `validation/02_partition_strategy_comparison.ipynb` — compare balanced vs group-blocked external partition regimes without reimplementing splitting in saber.

## Hyperparameter optimization
- `tuning/01_hyperparameter_optimization.ipynb` — grid search, multi-metric candidate report, untuned vs tuned protected test.
- `tuning/02_optuna_optimization.ipynb` — typed continuous spaces, convergence and trial diagnostics.
- `tuning/03_optimizer_comparison.ipynb` — Grid vs Random vs Optuna on identical memberships and search domain.

## Benchmarking and reporting
- `benchmark/01_multi_representation_multi_partition.ipynb` — representations × partitions × algorithms × seeds benchmark matrix.
- `reporting/01_benchmark_reporting.ipynb` — leaderboard, stability, sample-level error audit and CSV/Markdown report export.

## Persistence
- `persistence/01_model_persistence.ipynb` — auditable artifact, checksum verification, reload and inference report.

## End to end
- `end_to_end/01_data_centric_benchmark.ipynb` — representation comparison with untuned/tuned models and a protected final test.

### Test mode
The test suite sets `SABER_DEMO_TEST=1` to reduce dataset sizes/trial counts while exercising the same workflows. Running notebooks normally uses the fuller demonstration settings.


## Recommended learning paths

**First validation workflow**

1. `classification/01_binary_classification.ipynb`
2. `validation/01_biosieve_validation.ipynb`
3. `tuning/01_hyperparameter_optimization.ipynb`
4. `persistence/01_model_persistence.ipynb`

**Data-centric benchmarking**

1. `validation/02_partition_strategy_comparison.ipynb`
2. `classification/03_model_comparison.ipynb`
3. `benchmark/01_multi_representation_multi_partition.ipynb`
4. `reporting/01_benchmark_reporting.ipynb`
5. `end_to_end/01_data_centric_benchmark.ipynb`

For conceptual documentation, start at [`../docs/index.md`](../docs/index.md).
