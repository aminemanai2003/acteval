# API guide

## High-level functions

### `evaluate(y_true, y_pred, *, task, exposure=None, sample_weight=None, metrics=None)`

Returns `EvaluationResult`. With `metrics=None`, ActEval uses task defaults.
Metrics can be registry names, supported tail aliases, or `MetricSpec` values.

### `compare(y_true, predictions, *, task, exposure=None, sample_weight=None, metrics=None)`

`predictions` maps unique model labels to arrays. Returns `ComparisonResult`.
Every model is evaluated with identical observations, weighting, task, and
metric specifications.

## Result objects

`EvaluationResult` provides:

- `metrics`: label-to-value mapping;
- `metadata`: weighting and per-metric parameter metadata;
- `summary()`;
- `to_dict()`;
- `to_dataframe()`.

`ComparisonResult` provides:

- `results`: model-to-`EvaluationResult` mapping;
- `summary()`;
- `to_dict()`;
- `to_dataframe()`;
- `rank(metric)`.

There is intentionally no universal model ranking.

## Diagnostic tables

`calibration_by_quantile()` returns `CalibrationTable` and
`lift_by_quantile()` returns `LiftTable`. Both expose `bins`,
`requested_bins`, `effective_bins`, `to_dict()`, and `to_dataframe()`.

## Plotting

Plotting functions return Matplotlib axes and accept an existing axis:

- `plot_calibration()`;
- `plot_lift()`;
- `plot_residuals()`;
- `plot_tail_diagnostics()`.

Metric calculation does not import Matplotlib.
