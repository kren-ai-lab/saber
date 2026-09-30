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
    # Regression — leakage-safe preprocessing, error analysis and stability
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Exercise regression with missing values, fold-local imputation/scaling, a broad metric panel, OOF residual diagnostics, fold stability and worst-sample reporting.
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

    from sklearn.datasets import make_regression
    from saber import validate, DatasetBundle, PartitionPlan

    return (
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        make_regression,
        np,
        pl,
        plt,
        validate,
    )


@app.cell
def _(DEMO_TEST, DatasetBundle, PartitionPlan, make_regression, np):
    rng=np.random.default_rng(55)
    n_samples=150 if DEMO_TEST else 360
    X,y=make_regression(n_samples=n_samples,n_features=16,n_informative=10,noise=18.0,random_state=55)
    # Introduce missingness only in X; fold-local imputation must handle it.
    mask=rng.random(X.shape)<0.025
    X[mask]=np.nan
    ids=[f"reg_{i:04d}" for i in range(len(y))]
    dataset=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"f{i:02d}" for i in range(X.shape[1])])
    plan=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=np.arange(len(y))%5,
                                             dataset_fingerprint=dataset.fingerprint)
    return dataset, ids, plan, y


@app.cell
def _(dataset, ids, pl, plan, validate, y):
    metrics=("rmse","mae","median_ae","r2","explained_variance","pearson","spearman")
    result=validate(dataset=dataset,algorithm="ridge_regressor",partition_plan=plan,metrics=metrics,random_state=42)
    oof=result.oof_prediction
    assert oof is not None
    report = pl.DataFrame({"sample_id": ids, "y_true": y, "y_pred": oof.predictions}).with_columns(
        (pl.col("y_true") - pl.col("y_pred")).alias("residual"),
    ).with_columns(
        pl.col("residual").abs().alias("absolute_error"),
        pl.col("y_true").qcut(4, labels=["Q1", "Q2", "Q3", "Q4"]).alias("target_quartile"),
    )
    quartile_error = (
        report.group_by("target_quartile")
        .agg(
            pl.col("absolute_error").mean().alias("mean"),
            pl.col("absolute_error").median().alias("median"),
            pl.col("absolute_error").max().alias("max"),
        )
        .sort("target_quartile")
    )
    worst = report.sort("absolute_error", descending=True).head(10)
    fold_metrics = pl.DataFrame([{"fold":f.split_name,**f.evaluation.metrics} for f in result.folds])
    print({k: round(result.aggregate_metrics[k], 4) for k in sorted(result.aggregate_metrics)})
    quartile_error
    return fold_metrics, metrics, oof, quartile_error, report, result, worst


@app.cell
def _(fold_metrics, mo, np, oof, plt, report, y):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    y_true = report.get_column("y_true").to_numpy()
    y_pred = report.get_column("y_pred").to_numpy()
    residual = report.get_column("residual").to_numpy()
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    axes[0,0].scatter(y_true,y_pred,s=18,alpha=.7); lo=min(y.min(),oof.predictions.min()); hi=max(y.max(),oof.predictions.max()); axes[0,0].plot([lo,hi],[lo,hi],"--"); axes[0,0].set(title="OOF observed vs predicted",xlabel="Observed",ylabel="Predicted")
    axes[0,1].scatter(y_pred,residual,s=18,alpha=.7); axes[0,1].axhline(0,linestyle="--"); axes[0,1].set(title="Residual structure",xlabel="Predicted",ylabel="Residual")
    axes[1,0].hist(residual,bins=20); axes[1,0].set(title="Residual distribution",xlabel="Residual")
    fold_cols = ["rmse", "mae"]
    grouped_bar(
        axes[1,1],
        fold_metrics.get_column("fold").to_list(),
        {col: fold_metrics.get_column(col).to_numpy() for col in fold_cols},
    )
    axes[1,1].set_title("Fold error stability")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(FIGURE_COUNT, dataset, metrics, np, quartile_error, result, worst):
    DEMO_CHECKS={
        "complete_oof": result.metadata["oof_complete"] is True,
        "all_metrics": set(metrics)==set(result.aggregate_metrics),
        "missing_values_exercised": np.isnan(dataset.X).any(),
        "quartile_report": len(quartile_error)==4,
        "worst_samples_traceable": worst.get_column("sample_id").is_unique().all(),
        "finite_metrics": all(np.isfinite(v) for v in result.aggregate_metrics.values()),
        "figure_created": FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
