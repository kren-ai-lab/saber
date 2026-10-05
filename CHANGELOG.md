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
- API signatures:
  - `save_model(path, result, *, dataset, partition_plan=None, metadata=None, overwrite=False)`
    takes a `TrainResult` or `OptimizationResult`; model, algorithm, parameters, feature schema,
    positive class, random state and preprocessing come from the result. `train()` no longer
    saves (`artifact_path`, `artifact_overwrite`, `partition_plan` and `metadata` were removed).
    A tuned artifact stores its cross-validated scores in `selection_scores.json` (they chose the
    hyperparameters and are not a final performance estimate); `metrics.json` is gone. Artifact
    schema version is `2.0`; `1.0` artifacts are rejected.
  - `validate`/`tune` take one `partition_plan` (a `PartitionPlan` or `BioSievePartitionConfig`);
    `benchmark` takes `partitions` (a plan, a mapping of label to plan, or a
    `BioSievePartitionConfig`, generated from the first dataset). `partitioning`,
    `partitioning_reference` and `biosieve_extra_columns` were removed; extra columns are
    `BioSievePartitionConfig.extra_columns`.
  - `benchmark(datasets=...)` accepts a `DatasetBundle` or a mapping of label to `DatasetBundle`;
    `BenchmarkDataset` and `BenchmarkPartition` were removed.
  - `metrics`, `evaluation_role`, `require_complete`, `positive_class`, `return_estimators` and
    `random_state` are keyword arguments of `validate`/`tune`/`benchmark` (where they apply);
    `TuningConfig` lost `metrics`/`random_state` and `BenchmarkConfig` lost `metrics`,
    `evaluation_role`, `require_complete` and `return_estimators`. Tuned benchmark runs select on
    the benchmark metrics (refit on `TuningConfig.refit_metric`, default the first metric) with
    each benchmark seed as the tuning seed.
  - `predict(model, X, *, sample_ids=None, positive_class=None)` with `X` a `DatasetBundle`,
    array or DataFrame, and `evaluate(model, dataset, *, metrics=None, positive_class=None)`.
    `predict` lost `dataset=`, `feature_names=` and `strict_environment=` (use `load_model`), and
    raises `ValidationContractError` for bad inputs. `LoadedModelArtifact.predict_result` is private.
  - `PartitionPlan.holdout`, `PartitionPlan.from_predefined_folds`, `partition_plan_from_frame`
    and `load_partition_plan` take `dataset=` instead of `dataset_fingerprint=`.
- Config schema 2.0 (`schema_version: "2.0"`; `1.0` configs are rejected). Keys mirror the Python
  API:
  - `output: DIR` is a directory string (`output.path`, `output.directory` and `output.summary`
    were removed). Every workflow writes `summary.json` and fixed-name tables into it: `validate`
    `metrics.csv` and `predictions.csv`, `tune` `optimization_history.csv`, `evaluate`
    `metrics.csv`, `predict` `predictions.csv`. For `benchmark` the output directory is the
    `save_benchmark` artifact; the `benchmark` `artifact` block and `include_object` were removed.
  - `train`/`tune` save the model to `artifact: PATH`; `evaluate`/`predict` read `model: PATH`.
    A top-level `overwrite: true` lets both `artifact` and `output` replace existing paths; by
    default an existing path is refused before the workflow runs.
  - Dataset keys are `path`, `target_col`, `sample_id_col`, `feature_cols`, `group_col`,
    `sample_weight_col` and `sep` (were `target`, `sample_id`, `features`, `groups`,
    `sample_weight`). Datasets may be `.csv`, `.tsv`, `.txt` (tab-separated) or `.parquet`, for
    labeled and unlabeled workflows alike.
  - `partition` is `partition_plan` (keys: `path`, `sample_id_col`, `role_col`, `split_col`,
    `fold_col`, `always_train_value`). `partitioning` takes `strategy`, `params`,
    `seq_col`, `cluster_col`, `date_col` and `extra_columns`, a list of dataset columns read into
    `BioSievePartitionConfig.extra_columns`; columns named by any of these are never used as
    features. `biosieve_extra_columns` was removed.
  - `metrics`, `evaluation_role`, `require_complete`, `positive_class` and `random_state` are
    top-level keys in every workflow that takes them; `tuning` holds `TuningConfig` fields only and
    `benchmark` holds `BenchmarkConfig` fields only (no `metrics`, `metadata`, `evaluation_role`,
    `require_complete`, `return_estimators`, nor `tuning.metrics`/`tuning.random_state`).
    `return_estimators` is not a config key.
  - `benchmark` takes `datasets` only (one representation is a one-entry mapping) and its
    `benchmark` block is optional; `partitioning_reference` requires `partitioning`. A `train`
    `partition_plan` (artifact provenance) requires `artifact`.
- Output formats:
  - `PredictionResult.to_frame(y_true=None)` is the one prediction table: `sample_id`, `y_true`
    (if given), `y_pred`, `probability__{class}` per class, and `decision_score` (binary, oriented
    toward `positive_class`) or `decision_score__{class}` (multiclass). Validate/predict
    `predictions.csv` and `BenchmarkResult.predictions_frame()` (which adds run identity, `split`
    and `evaluation_role`) use it; the `prediction` and one-dimensional `probability` columns were
    removed, and validate `predictions.csv` gains `y_true`.
  - `ValidationResult.metrics_frame()` returns long-form metrics (`level`, `split`,
    `evaluation_role`, `metric`, `score`, `std`, `min`, `max`, `n`, `fit_seconds`): one
    `aggregate` row per metric (mean, sample std, min, max and integer `n` of finite fold scores)
    and one `fold` row per split. Validate `metrics.csv` is this table, and benchmark metrics tables
    carry the same columns plus run identity and `elapsed_seconds`. `ValidationResult.metric_summary`
    was removed.
  - `FoldValidationResult` has `split` (was `split_name`) and `metrics` (was `evaluation.metrics`);
    `evaluation` was removed, `prediction` stays.
  - `BenchmarkResult.failures_frame()` and the benchmark artifact's `failures.csv` were removed:
    failed runs are the `runs_frame()`/`runs.csv` rows with `status == "failed"`.
  - `BenchmarkRun.dataset_label` and the benchmark tables' `dataset` column were removed; they
    always equalled `representation`.
  - Every workflow result (`TrainResult`, `EvaluationResult`, `ValidationResult`,
    `OptimizationResult`, `BenchmarkResult`, `PredictionResult`) has `to_dict()`. `summary.json` and
    the `saber run --json` summary are `{"workflow", ...result.to_dict()}` with `algorithm`
    (`algorithms`), `task`, `dataset_fingerprint`, `partition_fingerprint` (validate/tune) and
    `metrics` (validate fold means, tune selection scores, evaluate scores); `saber run --json`
    adds `"status": "ok"`.
  - `OptimizationResult.best_scores`, the history and `optimization_history.csv`
    report metrics in their natural direction (RMSE is positive); selection is unchanged.
    `display_score`, `display_scores` and `metric` (always `refit_metric`) were removed.
  - Write-only metadata was removed: `ValidationResult.metadata` keeps only the dataset and
    partition fingerprints (OOF availability is `oof_prediction is not None`), `FoldValidationResult.metadata`
    is gone, `OptimizationResult.metadata` drops its count, timing and partition-source keys, and
    `PredictionResult.metadata` keeps only `algorithm`. `DatasetBundle.generated_sample_ids` was removed.
  - `OptimizationResult.best_score` and `refit` were removed: use `best_scores[refit_metric]` and
    `best_model is not None`. `to_dict()` carries them as `metrics` and `refit_metric`.
  - `BenchmarkRun.targets` moved to `BenchmarkResult.targets` (stored once; all representations
    share the same targets).
  - `SearchSpace.name` was removed (it repeated the algorithm name): `SearchSpace(parameters)`.
