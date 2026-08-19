# ActEval

[![PyPI version](https://img.shields.io/pypi/v/acteval-insurance.svg)](https://pypi.org/project/acteval-insurance/)
[![Python versions](https://img.shields.io/pypi/pyversions/acteval-insurance.svg)](https://pypi.org/project/acteval-insurance/)
[![CI](https://github.com/aminemanai2003/acteval/actions/workflows/ci.yml/badge.svg)](https://github.com/aminemanai2003/acteval/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/acteval-insurance.svg)](https://github.com/aminemanai2003/acteval/blob/main/LICENSE)
[![Typed](https://img.shields.io/badge/typing-typed-blue.svg)](https://peps.python.org/pep-0561/)

**Model-agnostic evaluation for actuarial predictive models.**

ActEval evaluates prediction arrays—not fitted model objects—across accuracy,
calibration, discrimination, probabilistic quality, uncertainty, observed-tail
risk, and financial decisions. It works with outputs from GLMs, scikit-learn,
XGBoost, CatBoost, neural networks, or any other modelling stack.

The project is designed for non-life insurance pricing workflows. It keeps
actuarial objectives separate and never creates an arbitrary universal model
score.

## Why ActEval?

A model with lower RMSE can still have worse aggregate calibration, weaker
large-loss behavior, or a less favorable pricing consequence. ActEval makes
those trade-offs visible through explicit metrics and reproducible metadata.

| Capability | Included diagnostics |
|---|---|
| Point predictions | MAE, RMSE, Poisson/Gamma/Tweedie deviance |
| Calibration | A/E, calibration by risk quantile, weighted calibration error |
| Discrimination | Gini, normalized Gini, lift |
| Tail risk | Observed-tail MAE, RMSE, A/E, large-loss bias |
| Predictive distributions | CRPS, log, Brier, quantile, and interval scores |
| Uncertainty | Coverage, width, variance, entropy, bootstrap intervals |
| Model comparison | Metric-specific ranking and paired bootstrap differences |
| Monitoring | Segment reports, temporal validation, prediction drift/PSI |
| Decisions | Pricing regret, loss ratio, reserve/capital shortfall, reinsurance |
| Reporting | DataFrame, dictionary, CSV, JSON, HTML, and plot export |

## Installation

ActEval requires Python 3.11 or newer.

```bash
python -m pip install acteval-insurance
```

Install the optional plotting support with:

```bash
python -m pip install "acteval-insurance[plot]"
```

The distribution name is `acteval-insurance` because `acteval` was already
occupied on PyPI. The import remains concise:

```python
import acteval as ae
```

## Quick start

ActEval accepts ordinary NumPy-compatible arrays and returns structured result
objects.

```python
import acteval as ae

y_true = [0.0, 0.4, 1.0, 2.0, 4.0, 7.0]
y_pred = [0.1, 0.5, 0.9, 1.8, 3.6, 6.4]
exposure = [1.0, 0.5, 1.2, 0.8, 1.5, 2.0]

result = ae.evaluate(
    y_true,
    y_pred,
    exposure=exposure,
    input_scale="rate",
    task="claim_frequency",
    context={"model_id": "frequency-glm-v4", "split": "holdout", "split_seed": 42},
)

print(result.to_dataframe())
```

Task defaults avoid making portfolio-specific tail thresholds or Tweedie-power
assumptions. Select those metrics explicitly and record their parameters:

```python
result = ae.evaluate(
    y_true,
    y_pred,
    task="claim_frequency",
    metrics=[
        "rmse",
        "poisson_deviance",
        "ae_ratio",
        "normalized_gini",
        "tail_ae_95",
    ],
)
```

Parameterized metrics use `MetricSpec`, keeping every assumption in result
metadata:

```python
result = ae.evaluate(
    y_true,
    y_pred,
    task="pure_premium",
    metrics=[
        ae.MetricSpec("tweedie_deviance", {"power": 1.7}),
        ae.MetricSpec("tail_mae", {"quantile": 0.99}, label="tail_mae_99"),
    ],
)
```

## Compare models

```python
comparison = ae.compare(
    y_true,
    {
        "GLM": glm_predictions,
        "Gradient boosting": boosting_predictions,
    },
    exposure=exposure,
    input_scale="rate",
    task="claim_frequency",
)

print(comparison.to_dataframe())
print(comparison.rank("poisson_deviance"))
```

Rankings are metric-specific. Target metrics such as A/E are ranked by distance
from their target; ActEval does not declare one model universally best.

## Quantify sampling uncertainty

Version 1.0 includes the inference layer introduced for the v0.4 roadmap.
Rows, predictions, exposures, and weights are resampled jointly.

```python
intervals = ae.bootstrap_evaluate(
    y_true,
    y_pred,
    exposure=exposure,
    input_scale="rate",
    task="claim_frequency",
    metrics=["rmse", "ae_ratio", "normalized_gini", "tail_ae_95"],
    n_resamples=2_000,
    confidence_level=0.95,
    random_state=42,
)

print(intervals.to_dataframe())
```

For model comparisons, paired resampling evaluates every model on the same
bootstrap rows. Negative `objective_delta` favors the candidate model after
accounting for whether a metric is minimized, maximized, or has a target.

```python
paired = ae.paired_bootstrap_compare(
    y_true,
    {"Current GLM": glm_predictions, "Candidate": boosting_predictions},
    reference="Current GLM",
    task="claim_frequency",
    metrics=["poisson_deviance", "ae_ratio", "normalized_gini"],
    n_resamples=2_000,
    random_state=42,
)
```

Confidence intervals are descriptive sampling-uncertainty estimates. Paired
comparisons are not automatically adjusted for multiple testing.

## Segment and temporal monitoring

The v0.5 monitoring layer evaluates portfolio slices without changing the
meaning of the underlying metrics.

```python
segments = ae.evaluate_by_segment(
    y_true,
    y_pred,
    segment_labels,
    task="claim_frequency",
    exposure=exposure,
    input_scale="rate",
    metrics=["ae_ratio", "normalized_gini", "tail_ae_95"],
)

timeline = ae.evaluate_over_time(
    y_true,
    y_pred,
    accounting_period,
    task="claim_frequency",
    exposure=exposure,
    input_scale="rate",
    metrics=["poisson_deviance", "ae_ratio"],
)

drift = ae.prediction_drift(
    reference_predictions,
    current_predictions,
    n_bins=10,
)
```

Prediction drift uses fixed, weighted reference-quantile bins and reports PSI
contributions. ActEval intentionally applies no universal PSI alert threshold.

## Predictive distributions

Built-in vectorized adapters represent one predictive distribution per
observation:

- `PoissonDistribution(mu)`
- `NegativeBinomialDistribution(mean, dispersion)`
- `GammaDistribution(mean, shape)`
- `LognormalDistribution(meanlog, sdlog)`
- `TweedieDistribution(mean, power, dispersion)` for `1 < power < 2`
- `EmpiricalDistribution(samples)` for joint or independent scenario draws

```python
poisson = ae.PoissonDistribution(mu=frequency_predictions)

distribution_result = ae.evaluate_distribution(
    claim_counts,
    poisson,
    task="claim_frequency",
    exposure=exposure,
    input_scale="rate",
    metrics=[
        ae.MetricSpec("crps", {"n_samples": 5_000, "random_state": 42}),
        "log_score",
        ae.MetricSpec("interval_score", {"coverage": 0.90}),
    ],
)
```

Samples have shape `(n_samples, n_observations)`. Scalar quantiles have shape
`(n_observations,)`; vector quantiles have shape
`(n_quantiles, n_observations)`.

## Decision-aware evaluation

Financial decisions always expose their loss function and named benchmark.
Regret is reported in the financial loss function's unit.

```python
premiums = ae.premium_from_distribution(
    severity_distribution,
    profit_loading=0.08,
    expense_ratio=0.20,
)

pricing = ae.pricing_regret(
    y_true=realized_loss,
    premium=premiums,
    benchmark_premium=current_tariff,
    underpricing_cost=2.0,
    overpricing_cost=1.0,
    benchmark_name="current tariff",
)
```

ActEval also provides loss-ratio impact, reserve and capital shortfall, and
quoted stop-loss reinsurance selection. These are explicit decision models,
not interchangeable measures of predictive accuracy.

## Reports and exports

Result objects support DataFrames, dictionaries, printable summaries, and
standalone HTML reports:

```python
result.save_html("reports/frequency-evaluation.html")
comparison.save_html("reports/model-comparison.html")

ae.export_table(comparison, "reports/model-comparison.csv")
ae.export_table(comparison, "reports/model-comparison.json")

axis = ae.plot_calibration(y_true, y_pred, exposure=exposure)
ae.save_plot(axis, "reports/calibration.png", dpi=180)
```

HTML reports contain no JavaScript or remote assets and can be archived for
offline review.

## Input contract

- `y_true` and `y_pred` are finite, one-dimensional, nonnegative arrays on the
  same scale.
- Frequency and pure-premium rates should be supplied with policy exposure.
- Severity observations are normally claim-level; `sample_weight` is often
  more meaningful than exposure.
- When both are present, effective weight is `sample_weight * exposure`.
- ActEval does not silently convert claim counts to rates.
- Observed-tail diagnostics select rows using realized outcomes and are
  retrospective—not predictive tail probabilities.

## Documentation

- [API guide](https://github.com/aminemanai2003/acteval/blob/main/docs/api.md)
- [Metric reference](https://github.com/aminemanai2003/acteval/blob/main/docs/metric-reference.md)
- [Bootstrap inference](https://github.com/aminemanai2003/acteval/blob/main/docs/inference.md)
- [Monitoring](https://github.com/aminemanai2003/acteval/blob/main/docs/monitoring.md)
- [Reporting](https://github.com/aminemanai2003/acteval/blob/main/docs/reporting.md)
- [Decision reference](https://github.com/aminemanai2003/acteval/blob/main/docs/decision-reference.md)
- [API stability policy](https://github.com/aminemanai2003/acteval/blob/main/docs/stability.md)
- [Migrating from 0.3 to 1.0](https://github.com/aminemanai2003/acteval/blob/main/docs/migration-1.0.md)

## Development

```bash
git clone https://github.com/aminemanai2003/acteval.git
cd acteval
python -m venv .venv
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/acteval
pytest
python -m build
```

Contributions are welcome. Read [CONTRIBUTING.md](https://github.com/aminemanai2003/acteval/blob/main/CONTRIBUTING.md)
and the [security policy](https://github.com/aminemanai2003/acteval/blob/main/SECURITY.md)
before opening a pull request or reporting a vulnerability.

## Versioning and license

ActEval follows Semantic Versioning from 1.0 onward. Public compatibility and
deprecation guarantees are documented in the stability policy.

Licensed under the [Apache License 2.0](https://github.com/aminemanai2003/acteval/blob/main/LICENSE).
