# Metric reference

All metrics accept finite one-dimensional arrays. Unless stated otherwise,
weights are nonnegative and effective weight is

```text
w_i = sample_weight_i * exposure_i
```

when both are present. Zero-weight observations do not affect calculations.

## Accuracy

### MAE

```text
sum_i w_i |y_i - p_i| / sum_i w_i
```

MAE has the target's unit and does not measure calibration or ordering.

### RMSE

```text
sqrt(sum_i w_i (y_i - p_i)^2 / sum_i w_i)
```

RMSE emphasizes large individual errors but is not an observed-tail metric.

### Poisson, Gamma, and Tweedie deviance

ActEval delegates mean deviance calculations to scikit-learn after validating
domains. Poisson permits nonnegative observations and requires positive
predictions. Gamma requires positive observations and predictions.

Tweedie power is always explicit for direct calls. Powers between 0 and 1 are
undefined. The pure-premium default report uses the documented compound
Poisson-Gamma power `1.5` and records it as result metadata; users should
override it when their fitted model uses a different power.

## Calibration

### Aggregate A/E

```text
sum_i w_i y_i / sum_i w_i p_i
```

The target is 1. Values above 1 indicate aggregate underprediction.

### Calibration by prediction quantile

Prediction cut points use linearly interpolated weighted quantiles. Equal
prediction values are never split, so fewer bins can be returned than
requested. Each bin reports count, effective weight, exposure, weighted mean
prediction, weighted mean observation, and A/E.

### Weighted calibration error

For bin weight `W_b`, observed mean `O_b`, and predicted mean `P_b`:

```text
sum_b W_b |O_b - P_b| / sum_b W_b
```

The metric has the target's unit. It is not classification ECE.

## Discrimination

### Gini

Observations are ordered by predictions from low to high and a weighted Lorenz
curve is integrated using trapezoids. Equal prediction scores are aggregated
before integration, making the result invariant to input order within ties.
Weighted observed loss must be positive.

### Normalized Gini

```text
Gini(y, prediction) / Gini(y, y)
```

A perfect ordering equals 1. Constant predictions equal 0. The metric is
undefined if perfect-ordering Gini is zero.

### Lift

Scalar lift is the weighted observed mean in the highest predicted fraction
divided by the portfolio observed mean. `lift_by_quantile()` returns this
relative observed risk for every prediction bin. Ties remain together.

## Observed-tail diagnostics

Tail membership is always defined using the observation:

```text
y_i > threshold
```

The boundary is strict. Supply either an absolute threshold or a quantile, not
both. When neither is supplied, the default observed quantile is 0.95.

- Tail MAE and RMSE apply their conventional formulas within selected rows.
- Tail A/E is weighted actual divided by weighted expected in selected rows.
- `large_loss_bias()` is the reciprocal expected-to-actual ratio retained for
  users who prefer the bias direction used in some actuarial presentations.

Selection on observed outcomes makes these retrospective diagnostics. They are
not estimates of a predictive tail probability or a proper scoring rule.
