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
    # Regression: leakage-safe preprocessing, error analysis and stability
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Ridge regression runs on data with missing values, and imputation and scaling are fitted inside each training fold. The report covers a wide metric panel, OOF residuals, fold stability, error by target quartile and the ten worst samples.
    """)
    return


@app.cell
def _():
    import os

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"

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
    rng = np.random.default_rng(55)
    n_samples = 150 if DEMO_TEST else 360
    X, y = make_regression(n_samples=n_samples, n_features=16, n_informative=10, noise=18.0, random_state=55)
    # Introduce missingness only in X; fold-local imputation must handle it.
    mask = rng.random(X.shape) < 0.025
    X[mask] = np.nan
    ids = [f"reg_{i:04d}" for i in range(len(y))]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"f{i:02d}" for i in range(X.shape[1])])
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids, fold_assignments=np.arange(len(y)) % 5, dataset=dataset
    )
    return dataset, ids, plan, y


@app.cell
def _(dataset, ids, pl, plan, validate, y):
    metrics = ("rmse", "mae", "median_ae", "r2", "explained_variance", "pearson", "spearman")
    result = validate(
        dataset=dataset, algorithm="ridge_regressor", partition_plan=plan, metrics=metrics, random_state=42
    )
    oof = result.oof_prediction
    assert oof is not None
    report = (
        pl.DataFrame({"sample_id": ids, "y_true": y, "y_pred": oof.predictions})
        .with_columns(residual=pl.col("y_true") - pl.col("y_pred"))
        .with_columns(
            absolute_error=pl.col("residual").abs(),
            target_quartile=pl.col("y_true").qcut(4, labels=["Q1", "Q2", "Q3", "Q4"]),
        )
    )
    quartile_error = (
        report.group_by("target_quartile")
        .agg(
            mean=pl.col("absolute_error").mean(),
            median=pl.col("absolute_error").median(),
            max=pl.col("absolute_error").max(),
        )
        .sort("target_quartile")
    )
    worst = report.sort("absolute_error", descending=True).head(10)
    fold_metrics = (
        result.metrics_frame()
        .filter(pl.col("level") == "fold")
        .pivot("metric", index="split", values="score")
        .rename({"split": "fold"})
    )
    print({k: round(result.aggregate_metrics[k], 4) for k in sorted(result.aggregate_metrics)})
    quartile_error, worst
    return fold_metrics, metrics, quartile_error, report, result


@app.cell
def _(fold_metrics, mo, plt, report):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].scatter(report["y_true"], report["y_pred"], s=18, alpha=0.7)
    axes[0, 0].axline((0, 0), slope=1, linestyle="--")
    axes[0, 0].set(title="OOF observed vs predicted", xlabel="Observed", ylabel="Predicted")
    axes[0, 1].scatter(report["y_pred"], report["residual"], s=18, alpha=0.7)
    axes[0, 1].axhline(0, linestyle="--")
    axes[0, 1].set(title="Residual structure", xlabel="Predicted", ylabel="Residual")
    axes[1, 0].hist(report["residual"], bins=20)
    axes[1, 0].set(title="Residual distribution", xlabel="Residual")
    for col in ("rmse", "mae"):
        axes[1, 1].plot(fold_metrics["fold"], fold_metrics[col], "o-", label=col)
    axes[1, 1].legend()
    axes[1, 1].set_title("Fold error stability")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(metrics, np, quartile_error, result):
    DEMO_CHECKS = {
        "all_metrics": set(metrics) == set(result.aggregate_metrics),
        "quartile_report": len(quartile_error) == 4,
        "finite_metrics": all(np.isfinite(v) for v in result.aggregate_metrics.values()),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
