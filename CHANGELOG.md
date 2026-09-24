# Changelog

All notable changes to Saber are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## 0.1.0 (unreleased)

First public release, published on PyPI as `saberlib`.

- Classical supervised classification and regression with a shared registry/factory execution path.
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
- Sample identifiers must share one scalar type (`str`, `int`, `float`, or `bool`); mixing types
  across `sample_ids` is rejected.
- Datasets loaded from CSV are parsed with Polars, including correctly rounded floats. A partition
  plan built in Python from `pandas.read_csv` defaults may carry a different dataset fingerprint
  than the same CSV loaded through saber — read with `polars.read_csv` or
  `pandas.read_csv(..., float_precision="round_trip")` to match it.
- Persisted/exported CSV tables are written by Polars (for example `0.00001`, `true`); values
  round-trip through saber's own readers.
- A multi-character `sep` in a dataset config is a configuration error.
- Fixed: benchmark `optimization_history_frame()` tables now carry the `parameters` column (it was
  always empty before).
