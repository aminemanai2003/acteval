# ActEval

ActEval is an insurance-oriented Python toolkit for evaluating caller-supplied
prediction arrays. It keeps accuracy, calibration, discrimination, tail,
uncertainty, and realized-consequence objectives separate instead of collapsing
them into a universal score.

## Install

```bash
python -m pip install acteval-insurance
```

Add plotting or numerical Tweedie distribution support only when needed:

```bash
python -m pip install "acteval-insurance[plot,tweedie]"
```

## First evaluation

```python
import acteval as ae

result = ae.evaluate(
    y_true=[0.0, 0.4, 1.0, 2.0, 4.0],
    y_pred=[0.1, 0.5, 0.9, 1.8, 3.6],
    exposure=[1.0, 0.5, 1.2, 0.8, 1.5],
    input_scale="rate",
    task="claim_frequency",
    context={"model_id": "frequency-glm-v4", "split": "holdout"},
)

print(result.to_dataframe())
```

Start with the [API guide](api.md) for input and result contracts, then use the
[metric reference](metric-reference.md) to choose diagnostics that match the
prediction functional and portfolio question.

## What this documentation covers

- Point and predictive-distribution evaluation
- Explicit exposure-scale handling
- Metric-specific model comparison
- Bootstrap sampling uncertainty
- Segment, temporal, and prediction-drift reports
- Immutable result metadata and exports
- Illustrative realized-consequence helpers

## Scope

ActEval is not a complete model-validation or governance system. It cannot
detect data leakage, validate a train/test split, choose portfolio thresholds,
or replace feature-aware conditional calibration. The
[compatibility policy](stability.md) describes software guarantees, not
actuarial fitness or production readiness.
