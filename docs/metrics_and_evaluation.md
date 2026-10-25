# Metrics and evaluation

Predictions come back as a `PredictionResult` with `predictions`,
`probabilities`, `decision_scores`, `classes`, `positive_class`, `sample_ids`
and `metadata`. Metrics are computed from it, and only the ones you request.

## Classification

| Metric | Requires |
|---|---|
| `accuracy`, `balanced_accuracy`, `mcc` | predictions |
| `precision`, `recall`, `sensitivity`, `specificity`, `f1` | predictions and, for binary, the positive class |
| `roc_auc`, `pr_auc` | probabilities or decision scores |
| `log_loss`, `brier_score` | probabilities |

For binary problems Saber finds the positive class's probability column
through the model's class order, so it works even when that class isn't
column 1 of `predict_proba`. The default positive class is the last one in
sorted order; pass `positive_class=` to pick another.

For multiclass problems, `precision`, `recall` and `f1` use weighted averaging;
`_macro`, `_micro` and `_weighted` variants (for example `f1_macro`) are also
available.

## Regression

`mae`, `median_ae`, `mse`, `rmse`, `mape`, `r2`, `explained_variance`,
`pearson`, `spearman`.

## Metrics for tuning

Tuning accepts a subset: `accuracy`, `balanced_accuracy`, `precision`,
`recall`, `f1`, `mcc` and `roc_auc` for classification; `mae`, `mse`, `rmse`,
`median_ae`, `r2` and `explained_variance` for regression. Losses are reported
with their natural sign.

## Undefined metrics

A fold can end up with a single class. Accuracy is still defined, but ROC-AUC
isn't, and Saber raises an error instead of returning a misleading number.

## Out-of-fold predictions

For complete, non-repeated cross-validation, `result.oof_prediction` is a
`PredictionResult` in the original sample order. Use it for thresholds,
calibration or error analysis without refitting.

## Result tables

`PredictionResult.to_frame(y_true=None)` returns one row per sample:
`id`, `y_true` (when given), `y_pred`, `probability__{class}` per class
(when the model has probabilities), and `decision_score` (binary, oriented
toward the positive class) or `decision_score__{class}` (multiclass).

`ValidationResult.metrics_frame()` is long form. Each metric has one
`level == "aggregate"` row, where `score` is the mean of the finite fold scores
and `std` (sample), `min`, `max` and `n` (number of finite folds) describe them,
followed by one `level == "fold"` row per split with its `split`,
`evaluation_role`, `score` and `fit_seconds`. Each fold is also available as
`result.folds[i]` with `split`, `metrics` and `prediction`.
