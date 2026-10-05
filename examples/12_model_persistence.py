import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Persistence: auditable model artifact and reproducible inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    This notebook trains a final model, saves it as an artifact, verifies its checksums, reads its manifest and reloads it. The reloaded model must reproduce the original predictions, which then feed a short inference report. Everything goes to a temporary directory, so the notebook can run any number of times.
    """)
    return


@app.cell
def _():
    import os

    import numpy as np
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"

    from pathlib import Path
    import tempfile
    from sklearn.datasets import make_classification
    from saber import train, save_model, predict, load_model, inspect_artifact, evaluate, DatasetBundle

    return (
        DEMO_TEST,
        DatasetBundle,
        Path,
        evaluate,
        inspect_artifact,
        load_model,
        make_classification,
        np,
        plt,
        predict,
        save_model,
        tempfile,
        train,
    )


@app.cell
def _(
    DEMO_TEST,
    DatasetBundle,
    Path,
    evaluate,
    inspect_artifact,
    load_model,
    make_classification,
    np,
    predict,
    save_model,
    tempfile,
    train,
):
    X, y = make_classification(
        n_samples=140 if DEMO_TEST else 280, n_features=12, n_informative=8, class_sep=1.1, random_state=101
    )
    ids = [f"persist_{i:04d}" for i in range(len(y))]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"f{i}" for i in range(X.shape[1])])
    trained = train(dataset=dataset, algorithm="logistic_regression", random_state=42)
    with tempfile.TemporaryDirectory(prefix="saber-example-") as tmp:
        artifact_path = Path(tmp) / "model_artifact"
        save_model(artifact_path, trained, dataset=dataset)
        manifest = inspect_artifact(artifact_path)  # verifies checksums before reading the manifest
        loaded = load_model(artifact_path)
        files = sorted(p.name for p in artifact_path.iterdir())
    pred = predict(loaded, dataset.X, sample_ids=dataset.sample_ids)
    evaluation = evaluate(
        loaded, dataset, metrics=("accuracy", "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "log_loss")
    )
    exact_roundtrip = np.array_equal(trained.model.predict(dataset.X), pred.predictions)
    print("Artifact files:", files)
    print("Manifest type:", manifest.artifact_type)
    print({k: round(v, 4) for k, v in evaluation.metrics.items()})
    return evaluation, exact_roundtrip, files, manifest, pred, y


@app.cell
def _(evaluation, mo, plt, pred, y):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    proba = pred.positive_probabilities()
    for cls in (0, 1):
        axes[0].hist(proba[y == cls], alpha=0.55, bins=18, label=str(cls))
    axes[0].legend()
    axes[0].set(title="Reloaded probability output", xlabel="P(positive)")
    metric_items = sorted(evaluation.metrics.items(), key=lambda kv: kv[1])
    axes[1].barh([k for k, _ in metric_items], [v for _, v in metric_items])
    axes[1].set_title("Reloaded model metrics")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(exact_roundtrip, files, manifest):
    DEMO_CHECKS = {
        "roundtrip_exact": exact_roundtrip,
        "manifest_is_model": manifest.artifact_type == "model",
        "artifact_is_auditable": {
            "manifest.json",
            "model.joblib",
            "environment.json",
            "feature_schema.json",
            "checksums.sha256",
        }.issubset(files),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
