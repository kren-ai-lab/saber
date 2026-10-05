# Persistence

A saved model is a directory with the fitted pipeline (`model.joblib`), its
feature schema, parameters, provenance (dataset and partition
fingerprints, class order, positive class), environment versions and a
SHA-256 checksum file.

## Save

```python
import saber

trained = saber.train(dataset=dataset, algorithm="random_forest_classifier", random_state=42)
saber.save_model("artifacts/model", trained, dataset=dataset)
```

`save_model(path, result, *, dataset, partition_plan=None, metadata=None,
overwrite=False)` takes a `train` or `tune` result; the model, algorithm,
parameters, feature schema and positive class come from it. The artifact schema
version is `"2.0"`. A tuned artifact also carries `selection_scores.json` with
the `refit_metric` and the selection scores of the best configuration. These are
scores from model selection, not final performance.

## Load and predict

```python
artifact = saber.load_model("artifacts/model")
prediction = saber.predict(artifact, X_new, sample_ids=sample_ids)
```

Loading verifies the checksums before deserializing. Prediction checks that
the features match the saved schema, including their order. Differences in
library versions are reported as warnings; pass `strict_environment=True` to
`saber.load_model` to make them errors. A `DatasetBundle` or DataFrame `X`
carries its feature names; NumPy input is only checked for width.
`saber.evaluate(artifact, dataset)` scores a loaded model.

`saber.inspect_artifact()` reads the manifest without loading the model and
checks the files and checksums (`verify=False` skips the check). It is also
available as `saber artifact inspect|verify`.

Checksums prove the files weren't altered, not that they're safe: joblib uses
pickle, so load artifacts only from sources you trust.

## Benchmarks

`saber.save_benchmark()` writes the benchmark tables (runs, metrics,
predictions, tuning history) as CSV files plus metadata; failed runs are the
`runs.csv` rows with `status == "failed"`. Saving the
full Python `BenchmarkResult` is optional and can produce much larger
artifacts.

See [`12_model_persistence.py`](../examples/12_model_persistence.py).
