import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Persistence — auditable model artifact and reproducible inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Train a final pipeline, persist it as an auditable artifact, inspect the manifest/environment/checksums, reload it, reproduce predictions, and generate an inference report. This demo uses a temporary directory so it is safe to execute repeatedly.
    """)
    return


@app.cell
def _():
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"

    def balanced_fold_labels(y, n_splits):
        """Demo-only external memberships. Production partition generation belongs to BioSieve."""
        y = np.asarray(y)
        folds = np.empty(len(y), dtype=int)
        for label in np.unique(y):
            idx = np.flatnonzero(y == label)
            folds[idx] = np.arange(len(idx)) % n_splits
        return folds

    from pathlib import Path
    import tempfile
    from sklearn.datasets import make_classification
    from saber import train, load_model, inspect_artifact, verify_artifact, evaluate
    from saber.datasets import DatasetBundle

    return (
        DEMO_TEST,
        DatasetBundle,
        Path,
        evaluate,
        inspect_artifact,
        load_model,
        make_classification,
        np,
        pl,
        plt,
        tempfile,
        train,
        verify_artifact,
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
    pl,
    tempfile,
    train,
    verify_artifact,
):
    X,y=make_classification(n_samples=140 if DEMO_TEST else 280,n_features=12,n_informative=8,class_sep=1.1,random_state=101)
    ids=[f"persist_{i:04d}" for i in range(len(y))]
    dataset=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"f{i}" for i in range(X.shape[1])])
    tmp=tempfile.mkdtemp(prefix="saber-example-")
    artifact_path=Path(tmp)/"model_artifact"
    trained=train(dataset=dataset,algorithm="logistic_regression",random_state=42,artifact_path=artifact_path)
    before=trained.model.predict(dataset.X)
    verified=verify_artifact(artifact_path)
    manifest=inspect_artifact(artifact_path)
    loaded=load_model(artifact_path)
    pred=loaded.predict_result(dataset.X,feature_names=dataset.feature_names,sample_ids=dataset.sample_ids)
    evaluation=evaluate(dataset=dataset,model=loaded,metrics=("accuracy","balanced_accuracy","mcc","roc_auc","pr_auc","log_loss"))
    files=sorted(p.name for p in artifact_path.iterdir())
    report=pl.DataFrame({"sample_id":ids,"y_true":y,"y_pred":pred.predictions,"p_positive":pred.positive_probabilities()})
    exact_roundtrip=np.array_equal(before,pred.predictions)
    print("Artifact files:",files)
    print("Manifest type:",manifest.artifact_type)
    print({k: round(v, 4) for k, v in evaluation.metrics.items()})
    return (
        dataset,
        evaluation,
        exact_roundtrip,
        files,
        manifest,
        pred,
        report,
        verified,
    )


@app.cell
def _(evaluation, mo, pl, plt, report):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for cls in sorted(report.get_column("y_true").unique().to_list()):
        subset = report.filter(pl.col("y_true")==cls).get_column("p_positive").to_numpy()
        axes[0].hist(subset,alpha=.55,bins=18,label=str(cls))
    axes[0].legend(); axes[0].set(title="Reloaded probability output",xlabel="P(positive)")
    metric_items = sorted(evaluation.metrics.items(), key=lambda kv: kv[1])
    axes[1].barh([k for k,_ in metric_items],[v for _,v in metric_items]); axes[1].set_title("Reloaded model metrics")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(
    FIGURE_COUNT,
    dataset,
    evaluation,
    exact_roundtrip,
    files,
    manifest,
    pred,
    verified,
):
    DEMO_CHECKS={
        "checksums_verified": bool(verified),
        "roundtrip_exact": exact_roundtrip,
        "manifest_available": getattr(manifest,"artifact_type",None)=="model",
        "structured_prediction": pred.sample_ids.shape[0]==dataset.n_samples,
        "artifact_is_auditable": {"manifest.json","model.joblib","environment.json","feature_schema.json","checksums.sha256"}.issubset(files),
        "evaluation_complete": set(evaluation.metrics)>={"accuracy","mcc","roc_auc"},
        "figure_created":FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
