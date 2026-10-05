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
    # Binary classification: OOF diagnostic report
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal and experimental design
    This notebook evaluates an imbalanced binary problem with more than one score. It uses a prepared feature matrix, fold memberships defined upstream, sample weights, preprocessing fitted inside each fold and a wide metric panel. The analysis covers out-of-fold (OOF) predictions, a threshold sweep, calibration metrics and an error report. `saber` does not generate the splits.
    """)
    return


@app.cell
def _():
    import os

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"

    from sklearn.datasets import make_classification
    from sklearn.metrics import (
        roc_curve,
        auc,
        precision_recall_curve,
        matthews_corrcoef,
        f1_score,
        recall_score,
    )
    from saber import validate, DatasetBundle, PartitionPlan

    return (
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        auc,
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
    The minority class gets a higher sample weight, which checks that weights reach every step of validation. The fold assignments stand in for memberships produced upstream, for example by BioSieve.
    """)
    return


@app.cell
def _(DEMO_TEST, DatasetBundle, PartitionPlan, make_classification, np, pl):
    n_samples = 180 if DEMO_TEST else 420
    X, y = make_classification(
        n_samples=n_samples,
        n_features=18,
        n_informative=10,
        n_redundant=3,
        weights=[0.72, 0.28],
        class_sep=1.15,
        flip_y=0.025,
        random_state=42,
    )
    ids = np.asarray([f"bin_{i:04d}" for i in range(len(y))], dtype=object)
    weights = np.where(y == 1, 1.8, 1.0)
    dataset = DatasetBundle(
        X=X,
        y=y,
        sample_ids=ids,
        sample_weight=weights,
        feature_names=[f"feature_{i:02d}" for i in range(X.shape[1])],
        metadata={"representation": "prepared_numeric"},
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=np.arange(len(y)) % 5,
        dataset=dataset,
        metadata={"source": "demo_external_memberships", "strategy": "round_robin_5fold"},
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
    The model returns probabilities, so the panel can include ranking, calibration and threshold metrics next to the class-label metrics.
    """)
    return


@app.cell
def _(dataset, pl, plan, validate):
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
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
    fold_metrics = (
        result.metrics_frame()
        .filter(pl.col("level") == "fold")
        .pivot("metric", index="split", values="score")
        .rename({"split": "fold"})
    )
    aggregate = {k: round(result.aggregate_metrics[k], 4) for k in sorted(result.aggregate_metrics)}
    print(aggregate)
    fold_metrics.head()
    return fold_metrics, metrics, oof, proba, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Threshold sensitivity and sample-level error analysis
    `saber` leaves threshold selection to the analyst. The sweep below is a diagnostic that shows how sensitivity, specificity, F1 and MCC change with the decision threshold. Choosing a threshold on these same OOF predictions and reporting its score would overstate performance. The error table lists the misclassified samples the model was most confident about.
    """)
    return


@app.cell
def _(f1_score, ids, matthews_corrcoef, np, oof, pl, proba, recall_score, y):
    def threshold_row(threshold):
        pred = (proba >= threshold).astype(int)
        return {
            "threshold": threshold,
            "mcc": matthews_corrcoef(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
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
    return (threshold_report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Visual report
    The four panels show per-fold metrics, the ROC and precision-recall curves, the threshold sweep and the OOF probabilities for each class.
    """)
    return


@app.cell
def _(
    auc,
    fold_metrics,
    mo,
    plt,
    precision_recall_curve,
    proba,
    result,
    roc_curve,
    threshold_report,
    y,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col in ("balanced_accuracy", "mcc", "roc_auc", "pr_auc"):
        axes[0, 0].plot(fold_metrics["fold"], fold_metrics[col], "o-", label=col)
    axes[0, 0].legend()
    axes[0, 0].set_title("Fold-level stability")
    axes[0, 0].set_ylim(-0.05, 1.05)

    fpr, tpr, _ = roc_curve(y, proba)
    precision, recall, _ = precision_recall_curve(y, proba)
    axes[0, 1].plot(fpr, tpr, label=f"ROC AUC={auc(fpr, tpr):.3f}")
    axes[0, 1].plot(recall, precision, label=f"PR AUC={result.aggregate_metrics['pr_auc']:.3f}")
    axes[0, 1].set(xlabel="x-axis rate", ylabel="y-axis rate", title="OOF ranking curves")
    axes[0, 1].legend()

    for col in ("mcc", "f1", "sensitivity", "specificity"):
        axes[1, 0].plot(threshold_report["threshold"], threshold_report[col], label=col)
    axes[1, 0].axvline(0.5, linestyle="--")
    axes[1, 0].set_title("Threshold sensitivity")
    axes[1, 0].legend()
    for cls in (0, 1):
        axes[1, 1].hist(proba[y == cls], bins=18, alpha=0.6, label=f"class {cls}")
    axes[1, 1].set(title="OOF probability distribution", xlabel="P(positive)")
    axes[1, 1].legend()
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(metrics, np, result, threshold_report):
    DEMO_CHECKS = {
        "five_folds": result.n_splits == 5,
        "all_metrics_present": set(metrics) == set(result.aggregate_metrics),
        "threshold_report_complete": len(threshold_report) == 17,
        "finite_metrics": all(np.isfinite(v) for v in result.aggregate_metrics.values()),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
