# ActEval

Evaluate actuarial predictive models beyond predictive accuracy.

ActEval is an early-stage, model-agnostic Python framework for non-life
insurance model evaluation. It accepts prediction arrays rather than fitted
model objects, so it can sit after GLMs, scikit-learn pipelines, gradient
boosting models, or neural networks.

> **Project status:** the foundation milestone is implemented: validation, a
> lightweight metric registry, MAE, RMSE, Poisson deviance, Gamma deviance,
> and aggregate actual-to-expected calibration. The high-level `evaluate()`
> and `compare()` APIs are planned for the next milestones.

## Why ActEval?

A model can improve average error while becoming materially worse for large
losses or for portfolio-level calibration. ActEval will keep accuracy,
calibration, discrimination, uncertainty, and tail-risk diagnostics separate
so users can see those trade-offs instead of hiding them in a composite score.

## Installation for development

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

The distribution is named `acteval-insurance` because the `acteval` name is
already occupied on PyPI. The import remains concise:

```python
import acteval as ae

y_true = [0.10, 0.25, 0.40]
y_pred = [0.12, 0.20, 0.35]
exposure = [1.0, 0.5, 2.0]

print(ae.rmse(y_true, y_pred, sample_weight=exposure))
print(ae.ae_ratio(y_true, y_pred, exposure=exposure))
print(ae.poisson_deviance(y_true, y_pred, sample_weight=exposure))
```

When `exposure` is supplied to `ae_ratio`, both actual and expected values are
exposure-weighted. Therefore `y_true` and `y_pred` must be on the same scale
(for example, annualized claim frequencies). If observations are raw claim
counts while predictions are frequencies, convert the counts to frequencies
before calling the function.

## Current metrics

```python
for metric in ae.list_metrics():
    print(metric.name, metric.category, metric.higher_is_better)
```

| Metric | Category | Direction |
|---|---|---|
| `mae` | accuracy | lower is better |
| `rmse` | accuracy | lower is better |
| `poisson_deviance` | accuracy | lower is better |
| `gamma_deviance` | accuracy | lower is better |
| `ae_ratio` | calibration | target is 1 |

No arbitrary overall score is produced. A metric only supports claims about
the objective it measures.

## Development checks

```bash
ruff check .
mypy src/acteval
pytest
python -m build
```

See [the audited implementation plan](docs/plan-audit.md) for scope decisions
and the next milestones.

## License

Apache-2.0.
