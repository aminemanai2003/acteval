# ActEval

Evaluate actuarial predictive models beyond predictive accuracy.

ActEval is a model-agnostic Python framework for non-life insurance model
evaluation. It accepts prediction arrays instead of fitted model objects, so it
works after GLMs, scikit-learn pipelines, XGBoost, CatBoost, or neural networks.

ActEval reports accuracy, calibration, discrimination, and observed-tail
behavior separately. It does not create an arbitrary overall score or claim
that one model is universally best.

## Installation

Until the first PyPI release, install the package directly from GitHub:

```bash
python -m pip install "acteval-insurance @ git+https://github.com/aminemanai2003/acteval.git"
```

For plots:

```bash
python -m pip install "acteval-insurance[plot] @ git+https://github.com/aminemanai2003/acteval.git"
```

The distribution is named `acteval-insurance` because `acteval` is occupied by
an unrelated project on PyPI. The import name remains `acteval`.

## Quick start

```python
import acteval as ae

result = ae.evaluate(
    y_true=[0.0, 0.4, 1.0, 2.0, 4.0, 7.0],
    y_pred=[0.1, 0.5, 0.9, 1.8, 3.6, 6.4],
    exposure=[1.0, 0.5, 1.2, 0.8, 1.5, 2.0],
    task="claim_frequency",
    metrics=["rmse", "poisson_deviance", "ae_ratio", "normalized_gini"],
)

print(result.summary())
```

Task defaults provide a broader report, including 95% observed-tail metrics.

## Model comparison

```python
comparison = ae.compare(
    y_true=y,
    predictions={
        "GLM": glm_predictions,
        "CatBoost": catboost_predictions,
        "XGBoost": xgb_predictions,
    },
    exposure=exposure,
    task="claim_frequency",
)

print(comparison.to_dataframe())
print(comparison.rank(metric="poisson_deviance"))
```

`rank()` uses a metric's documented direction. Target metrics such as A/E are
ranked by distance from 1. Rankings remain metric-specific.

## Accuracy can disagree with tail calibration

The example below deliberately creates two models:

- Model A makes moderate errors on many ordinary risks but predicts large
  observed outcomes accurately.
- Model B improves ordinary-risk predictions and overall RMSE while
  underpredicting the observed tail.

```python
import numpy as np
import acteval as ae

y = np.r_[np.tile([0.5, 1.0, 1.5, 1.0, 0.5], 19), np.repeat(10.0, 5)]
model_a = np.r_[y[:95] + 0.5, np.repeat(10.0, 5)]
model_b = np.r_[y[:95], np.repeat(9.0, 5)]

tradeoff = ae.compare(
    y,
    {"Model A": model_a, "Model B": model_b},
    task="claim_frequency",
    metrics=["rmse", "poisson_deviance", "tail_ae_95"],
)
print(tradeoff.to_dataframe())
```

Model B has lower overall RMSE and deviance, while Model A has tail A/E equal
to 1. The appropriate choice depends on the actuarial objective.

## Input and exposure contract

`y_true` and `y_pred` must be finite, one-dimensional, nonnegative arrays on
the same scale.

- For claim frequency, use frequency rates for both arrays and provide policy
  exposure as `exposure`.
- For pure premium, use pure-premium rates for both arrays and provide exposure
  when portfolio-volume weighting is desired.
- For severity, use claim severities. Exposure is optional and usually
  unnecessary; claim-level `sample_weight` is normally more meaningful.
- If both are supplied, effective weight is `sample_weight * exposure`.

ActEval does not silently convert raw claim counts into rates.

## Parameterized metrics

Use `MetricSpec` whenever a parameter should be explicit and reproducible:

```python
result = ae.evaluate(
    y,
    predictions,
    task="pure_premium",
    metrics=[
        ae.MetricSpec("tweedie_deviance", {"power": 1.7}),
        ae.MetricSpec("tail_mae", {"quantile": 0.99}, label="tail_mae_99"),
    ],
)
```

Tail aliases such as `tail_mae_95`, `tail_rmse_99`, and `tail_ae_95` are also
accepted. Parameter values are retained in result metadata.

## Calibration, discrimination, and tail diagnostics

```python
calibration = ae.calibration_by_quantile(y, predictions, n_bins=10)
lift = ae.lift_by_quantile(y, predictions, n_bins=10)

print(calibration.to_dataframe())
print(lift.to_dataframe())

ae.plot_calibration(y, predictions)
ae.plot_lift(y, predictions)
ae.plot_residuals(y, predictions)
ae.plot_tail_diagnostics(y, predictions, quantile=0.95)
```

## Supported MVP metrics

| Metric | Category | Interpretation |
|---|---|---|
| `mae` | accuracy | Lower is better |
| `rmse` | accuracy | Lower is better |
| `poisson_deviance` | accuracy | Lower; frequency only |
| `gamma_deviance` | accuracy | Lower; positive severity only |
| `tweedie_deviance` | accuracy | Lower; explicit power required |
| `ae_ratio` | calibration | Target is 1 |
| `weighted_calibration_error` | calibration | Lower is better |
| `gini` | discrimination | Higher is better |
| `normalized_gini` | discrimination | Perfect ordering is 1 |
| `lift` | discrimination | Higher means stronger top-group concentration |
| `tail_mae` | tail risk | Lower is better |
| `tail_rmse` | tail risk | Lower is better |
| `tail_ae_ratio` | tail risk | Target is 1 |

Use `ae.list_metrics()` for machine-readable registry metadata. Exact formulas
and limitations are in [the metric reference](docs/metric-reference.md).

## Development

```bash
git clone https://github.com/aminemanai2003/acteval.git
cd acteval
python -m venv .venv
python -m pip install -e ".[dev]"
ruff check .
mypy src/acteval
pytest
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[implementation audit](docs/plan-audit.md).

## Roadmap

- v0.1: point-prediction accuracy, calibration, discrimination, tail
  diagnostics, comparisons, and plotting.
- v0.2: carefully specified predictive-distribution scores and uncertainty
  diagnostics.
- v0.3: researched decision-aware actuarial consequences such as pricing or
  capital regret. No decision metric will be added without a defensible loss
  function and benchmark.

## License

Apache-2.0.
