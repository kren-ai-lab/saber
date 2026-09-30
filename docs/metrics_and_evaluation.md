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

For binary problems the positive class is resolved through the model's class
order, never by assuming column 1 of `predict_proba`. Pass `positive_class=`
when the default (the last class in sorted order) isn't the one you mean.

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
