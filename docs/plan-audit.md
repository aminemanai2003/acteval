# Implementation plan audit

The supplied plan has a strong scientific direction: it separates evaluation
from training, avoids a universal score, makes calibration and observed-tail
behavior first-class concerns, and defers decision-aware metrics until their
definitions are defensible.

## Decisions implemented in v0.1

1. **Use a `src/` layout.** Tests exercise the installed package layout.
2. **Separate distribution and import names.** The distribution is
   `acteval-insurance`; the Python import remains `acteval`.
3. **Use a same-scale target contract.** ActEval does not infer whether a
   frequency vector contains counts or rates. Frequency and pure-premium users
   provide rates when using exposure weighting.
4. **Define effective weight.** Exposure is portfolio volume. When both inputs
   are supplied, effective weight is `sample_weight * exposure`.
5. **Allow MAE and RMSE for all MVP tasks.** They remain scale-dependent
   diagnostics and are never presented as actuarially complete objectives.
6. **Reuse scikit-learn deviance implementations.** ActEval owns validation,
   metadata, and interpretation rather than duplicating formulas.
7. **Make Tweedie power reproducible.** Direct calls require power. The
   pure-premium default is the documented power 1.5 and records it in metadata.
8. **Keep prediction ties together.** Weighted risk-quantile cut points can
   produce fewer effective bins than requested.
9. **Define calibration error explicitly.** It is the bin-weighted absolute
   difference between weighted observed and predicted means and is not called
   ECE.
10. **Fix a Gini convention.** Scores sort low to high; score ties aggregate
    before weighted Lorenz integration. Normalization uses the observation
    ordering.
11. **Define observed tails strictly.** Tail rows satisfy `y_true > cutoff`.
    Weighted quantiles use cumulative-weight linear interpolation.
12. **Keep A/E direction consistent.** Both aggregate and tail A/E are actual
    divided by expected. `large_loss_bias` provides the reciprocal direction.
13. **Record parameter metadata.** `MetricSpec` preserves powers, quantiles,
    thresholds, bin counts, and caller labels in results.

## Implemented v0.1 slices

- Packaging, validation, registry, task defaults, tests, and CI.
- Conventional and actuarial deviance metrics.
- Aggregate and prediction-quantile calibration.
- Gini, normalized Gini, and lift diagnostics.
- Observed-tail MAE, RMSE, A/E, and large-loss bias.
- `EvaluationResult`, `ComparisonResult`, `evaluate()`, and `compare()`.
- Calibration, lift, residual, and tail plots.
- Synthetic trade-off script/notebook and metric/API documentation.
- Wheel/sdist builds and trusted-publishing release workflow.

## Later layers implemented after the MVP

Version 0.2 adds vectorized predictive distributions, explicit sample shapes,
reproducible sample-based scores, proper log/Brier/quantile/interval scores, and
uncertainty diagnostics. Entropy has no optimization direction.

Version 0.3 adds explicit financial decision rules and named benchmarks for
pricing, loss ratios, reserve shortfall, capital shortfall, and quoted
stop-loss reinsurance choices. Regret remains a difference in financial loss
and is never combined across decision objectives.
