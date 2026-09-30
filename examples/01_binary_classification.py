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
    # Binary classification — full OOF diagnostic report
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal and experimental design
    This notebook treats binary classification as a small scientific study rather than a one-score example. We use an imbalanced prepared matrix, explicit upstream fold memberships, sample weights, fold-local preprocessing, a broad metric panel, OOF diagnostics, threshold sensitivity analysis, calibration-style diagnostics, and an error report. No split generation is implemented inside `saber`.
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

    from sklearn.datasets import make_classification
    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                 matthews_corrcoef, f1_score, recall_score, confusion_matrix)
    from saber import validate, DatasetBundle, PartitionPlan

    return (
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        auc,
        balanced_fold_labels,
        confusion_matrix,
        f1_score,
        make_classification,
        matthews_corrcoef,
        np,
        pl,
        plt,
        precision_recall_curve,
        recall_score,
        roc_curve,
        validate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepared data and explicit memberships
    The minority class receives higher sample weight to exercise weight propagation through the validation pipeline. The fold assignments emulate an upstream/BioSieve artifact.
    """)
    return


@app.cell
def _(
    DEMO_TEST,
    DatasetBundle,
    PartitionPlan,
    balanced_fold_labels,
    make_classification,
    np,
    pl,
):
    n_samples = 180 if DEMO_TEST else 420
    X, y = make_classification(
        n_samples=n_samples, n_features=18, n_informative=10, n_redundant=3,
        weights=[0.72, 0.28], class_sep=1.15, flip_y=0.025, random_state=42,
    )
    ids = np.asarray([f"bin_{i:04d}" for i in range(len(y))], dtype=object)
    weights = np.where(y == 1, 1.8, 1.0)
    dataset = DatasetBundle(
        X=X, y=y, sample_ids=ids, sample_weight=weights,
        feature_names=[f"feature_{i:02d}" for i in range(X.shape[1])],
        metadata={"representation": "prepared_numeric"},
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=balanced_fold_labels(y, 5),
        dataset_fingerprint=dataset.fingerprint,
        metadata={"source": "demo_external_memberships", "strategy": "stratified_like_5fold"},
    )
    class_report = (
        pl.DataFrame({"label": y, "weight": weights})
        .group_by("label")
        .agg(pl.len().alias("count"), pl.col("weight").sum().alias("weighted_mass"))
        .sort("label")
    )
    class_report
    return dataset, ids, plan, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Validation with an extended metric panel
    The model returns probabilities, so we can evaluate ranking, calibration-related and threshold metrics together with class-label metrics.
    """)
    return


@app.cell
def _(dataset, pl, plan, validate):
    metrics = (
        "accuracy", "balanced_accuracy", "precision", "recall", "sensitivity",
        "specificity", "f1", "mcc", "roc_auc", "pr_auc", "log_loss", "brier_score",
    )
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=metrics,
        random_state=42,
        model_params={"max_iter": 2000},
    )
    oof = result.oof_prediction
    assert oof is not None
    proba = oof.positive_probabilities()
    fold_metrics = pl.DataFrame([{"fold": f.split_name, **f.evaluation.metrics} for f in result.folds])
    aggregate = {k: round(result.aggregate_metrics[k], 4) for k in sorted(result.aggregate_metrics)}
    print(aggregate)
    fold_metrics.head()
    return fold_metrics, metrics, oof, proba, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Threshold sensitivity and sample-level error analysis
    Threshold selection is intentionally an external analysis step. We do **not** optimize the threshold on these OOF data and then claim unbiased performance. The sweep is diagnostic: it shows how sensitivity, specificity, F1 and MCC move when the decision threshold changes.
    """)
    return


@app.cell
def _(
    confusion_matrix,
    f1_score,
    ids,
    matthews_corrcoef,
    np,
    oof,
    pl,
    proba,
    recall_score,
    y,
):
    def threshold_row(threshold):
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        return {
            "threshold": threshold,
            "mcc": matthews_corrcoef(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": specificity,
        }
    thresholds = np.linspace(0.10, 0.90, 17)
    threshold_report = pl.DataFrame([threshold_row(t) for t in thresholds])
    error_report = pl.DataFrame(
        {"sample_id": ids, "y_true": y, "y_pred": oof.predictions, "p_positive": proba}
    ).with_columns(
        (pl.col("y_true") != pl.col("y_pred")).alias("error"),
        pl.max_horizontal(pl.col("p_positive"), 1 - pl.col("p_positive")).alias("confidence"),
    )
    worst_errors = error_report.filter(pl.col("error")).sort("confidence", descending=True).head(10)
    threshold_report.head(), worst_errors
    return threshold_report, worst_errors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Visual report
    The four panels summarize fold stability, ranking behavior, threshold trade-offs and the distribution of OOF probabilities.
    """)
    return


@app.cell
def _(
    auc,
    fold_metrics,
    mo,
    np,
    plt,
    precision_recall_curve,
    proba,
    result,
    roc_curve,
    threshold_report,
    y,
):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fold_cols = ["balanced_accuracy", "mcc", "roc_auc", "pr_auc"]
    grouped_bar(
        axes[0, 0],
        fold_metrics.get_column("fold").to_list(),
        {col: fold_metrics.get_column(col).to_numpy() for col in fold_cols},
    )
    axes[0, 0].set_title("Fold-level stability"); axes[0, 0].set_ylim(-0.05, 1.05)

    fpr, tpr, _ = roc_curve(y, proba)
    precision, recall, _ = precision_recall_curve(y, proba)
    axes[0, 1].plot(fpr, tpr, label=f"ROC AUC={auc(fpr, tpr):.3f}")
    axes[0, 1].plot(recall, precision, label=f"PR AUC={result.aggregate_metrics['pr_auc']:.3f}")
    axes[0, 1].set(xlabel="x-axis rate", ylabel="y-axis rate", title="OOF ranking curves")
    axes[0, 1].legend()

    threshold_x = threshold_report.get_column("threshold").to_numpy()
    for col in ("mcc", "f1", "sensitivity", "specificity"):
        axes[1, 0].plot(threshold_x, threshold_report.get_column(col).to_numpy(), label=col)
    axes[1, 0].axvline(0.5, linestyle="--"); axes[1, 0].set_title("Threshold sensitivity"); axes[1, 0].legend()
    for cls in (0, 1):
        axes[1, 1].hist(proba[y == cls], bins=18, alpha=0.6, label=f"class {cls}")
    axes[1, 1].set(title="OOF probability distribution", xlabel="P(positive)"); axes[1, 1].legend()
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT = 1
    return (FIGURE_COUNT,)


@app.cell
def _(
    FIGURE_COUNT,
    dataset,
    metrics,
    np,
    result,
    threshold_report,
    worst_errors,
):
    DEMO_CHECKS = {
        "five_folds": result.n_splits == 5,
        "complete_oof": result.metadata["oof_complete"] is True,
        "all_metrics_present": set(metrics) == set(result.aggregate_metrics),
        "sample_weights_exercised": dataset.sample_weight is not None,
        "threshold_report_complete": len(threshold_report) == 17,
        "errors_traceable": set(worst_errors.columns) >= {"sample_id", "y_true", "y_pred", "p_positive"},
        "finite_metrics": all(np.isfinite(v) for v in result.aggregate_metrics.values()),
        "figure_created": FIGURE_COUNT == 1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
