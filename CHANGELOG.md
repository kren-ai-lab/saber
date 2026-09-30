# Changelog

All notable changes to Saber are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## 0.1.0 (unreleased)

First public release, published on PyPI as `saberlib`.

- Classical supervised classification and regression with a static algorithm catalog.
- Leakage-safe preprocessing, explicit and BioSieve-generated partitions, protected-test tuning.
- Grid, random, successive-halving and Optuna optimization; multi-representation benchmarking.
- Reproducible model and benchmark artifacts with checksums, fingerprints and environment metadata.
- The `saber` command-line interface.
- Tabular results (`BenchmarkResult.*_frame()`, `OptimizationResult.history_frame()`, loaded
  benchmark tables) are Polars DataFrames; nested cells in result tables are stable JSON strings.
- pandas DataFrames are accepted as inputs and converted once to Polars; this works with pandas 2
  and 3, and does not require pyarrow. pandas itself is not a runtime dependency.
- A result-table column that mixes scalar types across rows (for example a search-space column
  holding both `"sqrt"` and `0.5`) is stored as `String`.
- Custom preprocessing (`PreprocessingConfig(transformer=...)`) receives a Polars frame, so
  selecting columns by name works as usual; `sklearn.compose.make_column_selector` is pandas-only
  and is therefore not supported — select columns by name instead.
- Binary tuning scores `precision`, `recall`, `f1` and `roc_auc` for the positive class, the same
  quantity evaluation reports; `saber.tune()` and the tune workflow accept `positive_class`, and
  tuned benchmark runs select with the benchmark's `positive_class`.
- A stepped `Float` domain must have a range divisible by its step, so every optimizer searches the
  same grid.
- Sample identifiers must share one scalar type (`str`, `int`, `float`, or `bool`); mixing types
  across `sample_ids` is rejected.
- Datasets loaded from CSV are parsed with Polars, including correctly rounded floats. A partition
  plan built in Python from `pandas.read_csv` defaults may carry a different dataset fingerprint
  than the same CSV loaded through saber — read with `polars.read_csv` or
  `pandas.read_csv(..., float_precision="round_trip")` to match it.
- In CSV/TSV inputs, empty cells and `NA`, `N/A`, `n/a`, `NaN`, `nan`, `NULL`, `null`, `None` and
  `#N/A` are read as missing. Rarer pandas spellings such as `<NA>` or `-1.#QNAN` are read as text.
- Persisted/exported CSV tables are written by Polars (for example `0.00001`, `true`); values
  round-trip through saber's own readers.
- A multi-character `sep` in a dataset config is a configuration error.
- Fixed: benchmark `optimization_history_frame()` tables now carry the `parameters` column (it was
  always empty before).
- Algorithms live in a static `saber.ALGORITHMS` catalog with `saber.get_algorithm(name)`; the
  model registry, `AlgorithmRegistry`, name aliases, `EstimatorFactory`, `saber.optimize`,
  `PublicAPIError` and the scorers module were removed.
- `validate`, `tune` and `benchmark` are plain functions; the engine classes were removed.
- The CLI is reduced to `saber run`, `saber models` and `saber artifact`; the other commands
  (including `list`) were removed.
- Algorithm names: models offered for both tasks end in `_classifier`/`_regressor`, single-task
  models use the plain model name. Renamed:

  | Old | New |
  |-----|-----|
  | `adaboost`, `bagging`, `decision_tree`, `extra_tree`, `extra_trees`, `gaussian_process`, `gradient_boosting`, `hist_gradient_boosting`, `knn`, `radius_neighbors`, `random_forest` | same name + `_classifier` |
  | `xgbrf_regressor` | `xgb_rf_regressor` |
  | `huber_regression` | `huber_regressor` |
  | `gamma_regression` | `gamma_regressor` |
  | `lasso_regressor` | `lasso` |
  | `lars_regressor` | `lars` |
  | `lasso_lars_regressor` | `lasso_lars` |

- Artifact functions are `save_model`, `load_model`, `save_benchmark` and `load_benchmark`
  (`saber.persistence` and top-level `saber`); the `*_artifact` names were removed.
