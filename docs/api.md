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

## Predictive distributions

All adapters implement the `PredictiveDistribution` protocol:

- `n_observations`;
- `cdf(x)`;
- `quantile(q)`;
- `sample(n, random_state=...)`;
- `log_prob(y)`;
- `mean()`, `variance()`, and `entropy()`.

`evaluate_distribution()` returns an `EvaluationResult` and defaults to CRPS,
predictive variance, and 90% central-interval coverage and width.

`compare_distributions()` returns `ComparisonResult` and uses the same metric
specification and metric-specific ranking rules as point predictions.

`EmpiricalDistribution` is explicitly discrete: sample frequencies define its
probability mass and exact Shannon entropy; unseen values have log probability
`-inf`. `TweedieDistribution` uses a numerical series for its CDF and mixed
mass/density log probability, deterministic Monte Carlo quantiles, and a seeded
Monte Carlo estimate of `-E[log_prob(X)]`. Entropies across different base
measures are not directly comparable.

## Decision-aware results

`DecisionEvaluation` contains model financial loss, benchmark financial loss,
absolute regret, optional relative regret, and benchmark/loss metadata.

- `pricing_regret()` uses explicit asymmetric costs of underpricing and
  overpricing.
- `loss_ratio_impact()` returns the realized aggregate ratio and its signed and
  absolute displacement from a caller-selected target.
- `reserve_shortfall()` and `capital_shortfall()` report aggregate, mean,
  frequency, and conditional-mean insufficiency.
- `select_reinsurance_option()` selects among immutable `ReinsuranceOption`
  quotes under a documented projected cost.
- `reinsurance_decision_regret()` compares realized
  `premium + min(loss, retention)` costs to a named option.

`decision_regret()` is the low-level extension point. Callers provide a loss
function returning finite nonnegative financial loss per observation. Regret
can be negative and is never interpreted across different loss functions.
