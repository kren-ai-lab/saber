# Persistence and reproducibility

`Saber` persists model artifacts as auditable directories with structured metadata and checksums.

## Model artifacts

A typical artifact contains:

```text
artifact/
├── manifest.json
├── model.joblib
├── environment.json
├── feature_schema.json
├── provenance.json
├── parameters.json
├── metrics.json
├── training_config.json
├── partition_plan.json      # optional
└── checksums.sha256
```

The manifest format has its own schema version, independent of the `saber` package version.

## Saving

Use `saber.save_model()` / `save_model_artifact()` after final fitting. Provenance can include algorithm/provider, parameters, metrics, dataset fingerprint, partition fingerprint, class order, positive class, and training configuration.

## Loading and inference

```python
artifact = saber.load_model("artifacts/model")

prediction = artifact.predict_result(
    X_new,
    feature_names=feature_names,
    sample_ids=sample_ids,
)
```

The persisted `FeatureSchema` checks names and order before prediction. A DataFrame with the same columns in a different order is intentionally rejected unless it matches the original schema.

## Integrity and compatibility

`verify_artifact()` validates schema and SHA-256 checksums. `inspect_artifact()` can inspect the manifest without loading the model. Environment metadata records core and installed optional-provider versions; strict compatibility checking can be requested at load time.

Checksums establish **integrity**, not **trust**. `joblib` uses pickle semantics, so deserialize only artifacts from trusted sources.

## Benchmark artifacts

Benchmark persistence is table-first by default: runs, metrics, predictions, failures, and optimization history are written as analysis-ready files. The entire Python `BenchmarkResult` can optionally be stored, but doing so may create much larger artifacts.

See [`examples/persistence/01_model_persistence.ipynb`](../examples/persistence/01_model_persistence.ipynb).
