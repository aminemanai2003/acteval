# Changelog

## 0.3.0 - 2026-08-12

- Added generic benchmarked financial decision regret.
- Added premium decisions, asymmetric pricing regret, and loss-ratio impact.
- Added reserve and capital shortfall amount/frequency diagnostics.
- Added distribution quantile decisions.
- Added quoted stop-loss reinsurance option selection under explicit
  premium, retained-loss, and capital-cost assumptions.
- Added realized reinsurance decision regret against a named option.
- Completed numerical Tweedie CDF/log-density evaluation and reproducible
  mixed-measure entropy estimation.
- Defined empirical predictive samples as discrete distributions with exact
  probability masses and Shannon entropy.
- Added immutable metadata modules for every supported actuarial task.

## 0.2.0 - 2026-08-12

- Added vectorized Poisson, Negative Binomial, Gamma, Lognormal, empirical,
  and compound Poisson-Gamma Tweedie predictive distributions.
- Added sample-based CRPS, log score, threshold Brier score, quantile score,
  and central interval score.
- Added prediction-interval coverage and width diagnostics.
- Added predictive variance and entropy diagnostics without universal ranking
  directions.
- Added `evaluate_distribution()` and `compare_distributions()`.

## 0.1.0 - 2026-08-12

- Added validated, exposure-aware actuarial point-prediction inputs.
- Added MAE, RMSE, Poisson, Gamma, and explicit-power Tweedie deviance.
- Added aggregate and quantile calibration diagnostics.
- Added weighted Gini, normalized Gini, and lift diagnostics.
- Added observed-tail MAE, RMSE, A/E, and large-loss bias.
- Added metric registry and parameterized metric specifications.
- Added evaluation and model-comparison result objects.
- Added calibration, lift, residual, and tail plots.
- Added Python 3.11-3.13 CI and package-build validation.
