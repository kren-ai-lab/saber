# Persistence

A saved model is a directory with the fitted pipeline (`model.joblib`), its
feature schema, parameters, metrics, provenance (dataset and partition
fingerprints, class order, positive class), environment versions and a
SHA-256 checksum file.

## Save

```python
import saber

trained = saber.train(dataset=dataset, algorithm="random_forest_classifier", random_state=42)
saber.save_model(
    "artifacts/model",
    model=trained.model,
    algorithm=trained.spec.name,
    task=trained.spec.task,
    dataset=dataset,
    provider=trained.spec.provider,
    parameters=trained.parameters,
)
```

## Load and predict

```python
artifact = saber.load_model("artifacts/model")
prediction = artifact.predict_result(X_new, feature_names=feature_names, sample_ids=sample_ids)
```

Loading verifies the checksums before deserializing. Prediction checks that
the features match the saved schema, including their order. Differences in
library versions are reported as warnings; pass `strict_environment=True` to
make them errors.

`saber.inspect_artifact()` reads the manifest without loading the model, and
`saber.verify_artifact()` checks the files and checksums. Both are also
available as `saber artifact inspect|verify`.

Checksums prove the files weren't altered, not that they're safe: joblib uses
pickle, so load artifacts only from sources you trust.

## Benchmarks

`saber.save_benchmark()` writes the benchmark tables (runs, metrics,
predictions, failures, tuning history) as CSV files plus metadata. Saving the
full Python `BenchmarkResult` is optional and can produce much larger
artifacts.

See [`12_model_persistence.py`](../examples/12_model_persistence.py).
